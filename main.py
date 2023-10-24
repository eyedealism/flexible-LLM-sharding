import copy
import os
import numpy as np
import pickle
import copy
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import torch
from transformers import AutoTokenizer
from utils import DeviceManager, ShardedLlama


def run_sharded_llama(device, prompt_list, device_manager, args):
    model = ShardedLlama(args, device_manager, device=device)
    if not args.gpu_parallel and device != device_manager.devices[0]:  # don't need actual prompts
        prompt_list = [None] * len(prompt_list)

    batch_ends = [len(prompt_list) // args.num_batch * i for i in range(1, args.num_batch)] + [len(prompt_list)]
    batch_ranges = list(zip([0] + batch_ends[:-1], batch_ends))
    llama_outputs = []
    for batch_start, batch_end in batch_ranges:
        llama_outputs += model(prompt_list[batch_start: batch_end])

    return llama_outputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./")
    parser.add_argument('--prompt_pickle', type=str, required=True, help="Path to the input prompt pickle file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the LLM output scores file")
    parser.add_argument('--num_batch', type=int, default=1)
    parser.add_argument('--layer_num_per_shard', type=int,
                        default=1)  # suggest to use 1 for gpu_parallel mode; and as largse as vRAM allows for gpu_series mode
    parser.add_argument('--storage_location', type=str, default='cpu',
                        help="'gpu': use vRAM to store intermediate activations, 'cpu': use RAM, 'disk': use disk")
    parser.add_argument('--max_activation_in_cpu', type=int, default=100)
    parser.add_argument('--gpu_parallel', type=bool, default=False, required=False,
                        # only matters if multiple GPUs are available
                        help="if gpu_parallel, multiple GPUs will run in parallel with the same layers; if False, multiple GPUs will run in series and load a portion of the layers")
    parser.add_argument('--disk_folder', type=str, default='./temp',
                        help="folder path for writing files of intermediate activations in 'disk' mode")
    parser.add_argument('--num_gen_token', type=int, default=1, help="how many new tokens to be generated")
    # parser.add_argument('--temperature', type=float, default=0,
    #                     help="temperature to perform token sampling")
    args = parser.parse_args()
    print(args)

    if args.storage_location == 'disk':
        os.makedirs(args.disk_folder, exist_ok=True)

    with open(args.prompt_pickle, 'rb') as file:
        original_input_prompts = pickle.load(file)

    input_prompts = copy.deepcopy(original_input_prompts)
    num_device = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(num_device)]
    device_manager = DeviceManager(args, devices)
    run_model_wrapped = partial(run_sharded_llama, device_manager=device_manager, args=args)
    # loop K times to generate K new tokens
    output_scores = []
    for i_new in range(args.num_gen_token):
        if args.gpu_parallel:  # devices run in parallel
            with ThreadPoolExecutor() as executor:
                input_prompts = np.array(input_prompts, dtype=object)
                outputs = list(executor.map(run_model_wrapped, devices, np.array_split(input_prompts, num_device), ))
        else:  # devices run in series
            with ThreadPoolExecutor() as executor:
                outputs = list(executor.map(run_model_wrapped, devices, [input_prompts] * num_device, ))
        outputs = sum(outputs, [])

        if i_new == 0:
            output_scores = outputs
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            tokenizer.pad_token = tokenizer.eos_token
        else:
            output_scores = [np.concatenate((old, new), axis=1) for old, new in zip(output_scores, outputs)]

        # add the newly generated token at the end of the prompts
        for prompt_idx in range(len(input_prompts)):
            prefix, suffix = original_input_prompts[prompt_idx]
            new_tokens = np.argmax(output_scores[prompt_idx], axis=-1)
            new_suffix = tuple([s+tokenizer.decode(t) for s, t in zip(suffix, new_tokens)])
            input_prompts[prompt_idx] = (prefix, new_suffix)

    # save the updated prompts
    with open(args.prompt_pickle.replace('.pkl', '_updated.pkl'), 'wb') as file:
        pickle.dump(input_prompts, file)

    # save the results
    with open(args.output_file, 'wb') as file:
        pickle.dump(output_scores, file)




