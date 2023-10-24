# flexible-LLM-sharding
This repository is designed for individuals who want to run LLM locally with small vRAM (GPU memory) and RAM. It facilitates the efficient execution of unquantized LLM's (such as Llama2-70B) with vRAM&ge;6GB and RAM&ge;8GB, i.e. when standard offloading does not work.


**Basic Usage**

To use flexible LLM sharding, first you will need to download and convert HF weights into layerwise weight files by running prepare_weights.py. An example notebook splitting Llama2-7B model could be found [here](https://www.kaggle.com/code/junxhuang/split-llama2-7b-weights).

Then you could run LLM with flexible sharding by
```bash
git clone https://github.com/Robot-Eyes/flexible-LLM-sharding.git

python flexible-LLM-sharding/main.py --layer_num_per_shard 1 --storage_location cpu --num_batch 1 \
                                    --model_path ./llama2_checkpoints \
                                    --prompt_pickle input_prompts.pkl --output_file output_score.pkl

```

The flag _--layer_num_per_shard_ controls how many layers should fit into one GPU as a shard. The more layers in a shard, the more vRAM it requires. Setting to 1 will minimize vRAM usage.

The flag _--storage_location_ controls where to store the intermediate activations. It could be 'cpu' (store in RAM), 'gpu' (store in vRAM), or 'disk' (write files to a folder you choose). The speeds of these modes vary depending on hardwares.

The flag _--model_path_ is where the layer-wise weights are.

The flags _prompt_pickle_ and _--output_file_ specific the pickle file that has the input prompts, and the output pickle file of the scores.

The flag _--gpu_parallel_ will let multiple GPUs run in parallel, i.e. each GPU loads the whole model and works on a portion of the prompts. It doesn't matter if only one GPU is available.

A sample demo for a Kaggle competition could be found in [this notebook](https://www.kaggle.com/junxhuang/running-llm-with-flexible-sharding-technique), which shows that flexible-LLM-sharding technique has much lower vRAM requirement (6GB for 70B model) and faster inference speed than standard offloading.
