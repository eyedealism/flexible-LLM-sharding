import gc
import ctypes
from time import time, sleep
import threading
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load as safetensors_load

max_token_len = 4096  # Llama token max length


# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


class DeviceManager:
    """
	Thread-safe class to manage multiple GPUs.
	parameter "max_layer_num" control how many layers can be store in RAM (must >= 1) in data_parallel mode.
	If args.data_parallel, then multiple GPUs will split the data; otherwise, GPUs will split the model.
	"""

    def __init__(self, args, devices, max_layer_num=1, ):
        assert max_layer_num >= 1, "Must set max_layer_num >= 1 to allow loading a layer"
        self.args = args
        self.devices = devices
        if args.data_parallel:  # only read the layers to cpu once to reduce IO load
            self.max_layer_num = max_layer_num
            self.layer2state_dict = {}
            self.layer2used_device = dict()
            self.waiting_queue = []
            self.condition = threading.Condition()
            self.lock = threading.Lock()
        else:  # the activation dictionary is shared among GPUs
            self.shared_activation_dict = dict()
            self.prompt2layer = defaultdict(lambda: -1)

    def get_state_dict(self, device, layer, ):
        with self.condition:
            while layer not in self.layer2state_dict:  # layer is not loaded yet
                print(f'{device} waiting {layer}')
                self.condition.wait()

            with self.lock:  # only allow one thread to modify the device manager
                result = self.layer2state_dict[layer]
                self.layer2used_device[layer].add(device)
                if len(self.layer2used_device[layer]) == len(self.devices):  # all devices have used this layer
                    del self.layer2state_dict[layer]
                    del self.layer2used_device[layer]
                    self.clean_waiting_queue()
                return result

    def send_device_request(self, device, layer, ):  # request to load a layer to RAM
        with self.lock:  # only allow one thread to modify the device manager
            if layer not in self.layer2used_device and layer not in self.waiting_queue:  # new request
                self.waiting_queue.append(layer)
                self.clean_waiting_queue()

    def clean_waiting_queue(self):
        # try to load a new layer
        with self.condition:
            while len(self.layer2state_dict) < self.max_layer_num and len(self.waiting_queue) > 0:
                working_layer = self.waiting_queue.pop(0)
                with open(f'{self.args.model_path}/{working_layer}.safetensors', 'rb') as f:
                    self.layer2state_dict[working_layer] = safetensors_load(f.read())
                self.layer2used_device[working_layer] = set()
                self.condition.notify_all()


# Class for sharded llama
class ShardedLlama:
    def __init__(self, args, device_manager, device="cuda:0", dtype=torch.float16):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed shard by shard, and the GPU memory is freed after each shard.
        The intermediate activations could be saved in vRAM, RAM, or even disk (slower due to the exact IO load),
        according to the argument 'storage_location'.

        Parameters
        ----------
        args : arguments of control parameters
        device : str, by default "cuda:0"
        dtype : torch.dtype, by default torch.float16
        """

        # Save parameters
        self.args = args
        self.device = device
        self.dtype = dtype
        self.device_manager = device_manager

        # Create model
        self.config = AutoConfig.from_pretrained(args.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.init_model()
        self.layer_names = ["model.embed_tokens"] + [f"model.layers.{i}"
                                                     for i in range(len(self.model.model.layers))] + ["model.norm", "lm_head"]

    def init_model(self):
        # Load meta model (no memory used)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
            self.model.tie_weights()
        self.layers = [self.model.model.embed_tokens] + list(self.model.model.layers) + [self.model.model.norm,
                                                                                         self.model.lm_head]

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.device, value=buffer, dtype=self.dtype)

    def load_layer_to_gpu(self, layer_name):
        if self.args.data_parallel:
            self.device_manager.send_device_request(self.device, layer_name, )
            state_dict = self.device_manager.get_state_dict(self.device, layer_name, )
        else:
            with open(f'{self.args.model_path}/{layer_name}.safetensors', 'rb') as f:
                state_dict = safetensors_load(f.read())  # .to(self.device)
        for param_name, param in state_dict.items():
            assert param.dtype != torch.int8, "int8 not supported (need to add fp16_statistics)"
            set_module_tensor_to_device(self.model, param_name, self.device, value=param, dtype=self.dtype)
        gc.collect()

    def __call__(self, inputs):
        # Reboot the model to make sure buffers are loaded and memory is clean
        del self.model
        clean_memory()
        self.init_model()

        input_len = len(inputs)
        num_gpu = len(self.device_manager.devices)
        gpu_rank = self.device_manager.devices.index(self.device)

        # split the model into N shards so that the layer number in each shard <= layer_num_per_shard
        if self.args.data_parallel:  # each gpu loads all the shards and has individual activation_dict/prompt2layer/suffix_eos
            num_shards = np.ceil(len(self.layers) / self.args.layer_num_per_shard)
            model_shards = np.array_split(np.arange(len(self.layers)), num_shards)
            activation_dict = dict()
            prompt2layer = defaultdict(lambda: -1)
            suffix_eos = []
        else:  # each gpu loads a part of the shards
            num_shards = np.ceil(np.ceil(len(self.layers) / self.args.layer_num_per_shard) / num_gpu) * num_gpu
            all_shards = np.array_split(np.arange(len(self.layers)), num_shards)
            model_shards = list(map(tuple, all_shards[gpu_rank::num_gpu]))
            activation_dict = self.device_manager.shared_activation_dict
            prompt2layer = self.device_manager.prompt2layer
            if gpu_rank == 0:
                self.device_manager.suffix_eos = suffix_eos = []

        def store_intermediate_activation(prompt_idx, layer_idx, prefix, suffix,
                                          storage_location='gpu', activation_dict=None):
            """ if layer_idx >= total_layer_num-2, prefix is None. """
            if storage_location == 'gpu':
                activation_dict[prompt_idx] = (prefix, suffix)
            elif storage_location == 'cpu':
                if prefix is not None:
                    activation_dict[prompt_idx] = (prefix.cpu(), suffix.cpu())
                else:
                    activation_dict[prompt_idx] = (None, suffix.cpu())
            else:  # disk
                if gpu_rank == num_gpu - 1 or self.args.data_parallel:  # write to disk
                    np.save(
                        f'{self.args.disk_folder}/suffix{gpu_rank if self.args.data_parallel else ""}-{prompt_idx:05d}.npy',
                        suffix.cpu().numpy(), )
                    if layer_idx < len(self.layers) - 2:
                        np.save(
                            f'{self.args.disk_folder}/prefix{gpu_rank if self.args.data_parallel else ""}-{prompt_idx:05d}.npy',
                            prefix.cpu().numpy(), )
                else:  # save in shared dict
                    while len(activation_dict) >= self.args.max_activation_in_cpu:
                        sleep(1)
                    if prefix is not None:
                        activation_dict[prompt_idx] = (prefix.cpu(), suffix.cpu())
                    else:
                        activation_dict[prompt_idx] = (None, suffix.cpu())
            prompt2layer[prompt_idx] = layer_idx

        def fetch_intermediate_activation(prompt_idx, layer_idx, storage_location='gpu',
                                          activation_dict=None):
            while prompt2layer[prompt_idx] != layer_idx - 1:
                sleep(1)
            if storage_location in ['gpu', 'cpu']:
                prefix, suffix = activation_dict[prompt_idx]
                suffix = suffix.to(self.device)
                if prefix is not None:
                    prefix = prefix.to(self.device)
            else:  # disk
                if gpu_rank == 0 or self.args.data_parallel:  # read from disk
                    suffix = torch.tensor(np.load(
                        f'{self.args.disk_folder}/suffix{gpu_rank if self.args.data_parallel else ""}-{prompt_idx:05d}.npy', )).to(
                        self.device)
                    if layer_idx < len(self.layers) - 1:
                        prefix = torch.tensor(np.load(
                            f'{self.args.disk_folder}/prefix{gpu_rank if self.args.data_parallel else ""}-{prompt_idx:05d}.npy', )).to(
                            self.device)
                    else:
                        prefix = None
                else:  # fetch from shared dict
                    prefix, suffix = activation_dict[prompt_idx]
                    del activation_dict[prompt_idx]
                    suffix = suffix.to(self.device)
                    if prefix is not None:
                        prefix = prefix.to(self.device)
            return prefix, suffix

        # start inference
        output_scores = []

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.finfo(self.dtype).min * torch.ones((max_token_len, max_token_len), dtype=self.dtype)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...].to(self.device)
        position_ids = torch.arange(max_token_len, dtype=torch.long, device=self.device)[None, :]

        load_weights_time = 0
        with torch.inference_mode():

            model_shards_iter = model_shards if len(model_shards) * 200 < input_len else tqdm(model_shards, leave=True,
                                                                                              desc=f'{self.device} | shards')
            for layer_idx_in_shard in model_shards_iter:
                # Load the current shard (multiple layers) to GPU
                start = time()
                for layer_idx in layer_idx_in_shard:
                    self.load_layer_to_gpu(self.layer_names[layer_idx])
                load_weights_time += time() - start

                # loop over all input prompts
                prompt_iter = tqdm(range(input_len), leave=True,
                                   desc=f'Stage {model_shards.index(layer_idx_in_shard)} | {self.device}') if len(
                    model_shards) * 200 < input_len else range(input_len)
                for prompt_idx in prompt_iter:

                    # Run one shard (multiple layers) on this input prompt
                    for layer_idx in layer_idx_in_shard:
                        layer_name, layer = self.layer_names[layer_idx], self.layers[layer_idx]

                        # fetch prefix and suffix for the first layer in each shard
                        if layer_idx == 0:  # fetch from input prompts
                            prompt_prefix, prompt_suffix = inputs[prompt_idx][0], inputs[prompt_idx][1]
                            prefix = \
                                self.tokenizer(prompt_prefix, return_tensors="pt", return_attention_mask=False,
                                               truncation=True, max_length=max_token_len)["input_ids"].to(
                                    self.device)
                            suffix = \
                                self.tokenizer(prompt_suffix, return_tensors="pt", return_attention_mask=False,
                                               truncation=True, max_length=max_token_len, padding=True)[
                                    "input_ids"][
                                :,
                                1:].to(self.device)
                            suffix_eos.append((suffix != self.tokenizer.pad_token_id).sum(1) - 1)  # -1 for zero-index
                            inputs[prompt_idx] = None
                        elif layer_idx == layer_idx_in_shard[0]:  # read stored activations
                            prefix, suffix = fetch_intermediate_activation(prompt_idx, layer_idx,
                                                                           storage_location=self.args.storage_location,
                                                                           activation_dict=activation_dict)

                        # perform layer-wise inference
                        if layer_name == "model.embed_tokens":  # first layer
                            prefix = layer(prefix)
                            suffix = layer(suffix)
                        elif "model.layers." in layer_name:  # middle layers
                            len_p, len_s, n_suffixes = prefix.shape[1], suffix.shape[1], suffix.shape[0]
                            # Run prefix
                            prefix, (k_cache, v_cache) = layer(prefix, use_cache=True, attention_mask=None)

                            # Run suffix
                            pos = position_ids[:, len_p:len_p + len_s].expand(n_suffixes, -1)
                            attn = attention_mask[:, :, -len_s:, -len_p - len_s:].expand(n_suffixes, -1, -1, -1)
                            kv_cache = (k_cache.expand(n_suffixes, -1, -1, -1), v_cache.expand(n_suffixes, -1, -1, -1))
                            suffix = layer(suffix, past_key_value=kv_cache,
                                           position_ids=pos, attention_mask=attn)[0]

                        elif layer_name == "model.norm":  # second last layer
                            # Only keep the last token
                            prefix = None
                            new_token_pos = suffix_eos[prompt_idx].cpu() if self.args.data_parallel else \
                                self.device_manager.suffix_eos[prompt_idx].cpu()
                            suffix = layer(suffix[torch.arange(suffix.shape[0]), new_token_pos][:, None])
                        else:  # last layer
                            logits = layer(suffix)[:, 0]
                            scores = torch.softmax(logits, axis=-1).detach().cpu().numpy()
                            output_scores.append(np.expand_dims(scores, axis=1))
                            del prompt2layer[prompt_idx]

                        # store mediate tensors for the last layer in each shard (except the last layer of the model)
                        if layer_idx == layer_idx_in_shard[-1] and layer_idx != len(self.layers) - 1:
                            store_intermediate_activation(prompt_idx, layer_idx, prefix, suffix,
                                                          storage_location=self.args.storage_location,
                                                          activation_dict=activation_dict)

                # Remove multiple layers from memory (including buffers)
                for layer_idx in layer_idx_in_shard:
                    self.layers[layer_idx].to("meta")
                clean_memory()

        print(f'{self.device} loaded {sum(map(len, model_shards))} layers in {load_weights_time:.0f}s')
        return output_scores
