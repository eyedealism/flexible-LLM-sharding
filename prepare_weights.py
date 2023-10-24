import os
import shutil
import gc
from glob import glob
import json
from tqdm import tqdm
import torch
from safetensors.torch import save_file
import argparse


def split_into_layers(bin_dir, new_file_dir):
    os.makedirs(new_file_dir, exist_ok=True)
    for fn in glob(f'{bin_dir}/*'):
        if '.bin' not in fn:
            shutil.copy(fn, f"{new_file_dir}/{fn.split('/')[-1]}")

    with open(f'{bin_dir}/pytorch_model.bin.index.json', 'rb') as f:
        sublayer2shard = json.load(f)['weight_map']

    sublayer2layer = {k: '.'.join(k.replace('.weight', '').split('.')[:3]) for k in sublayer2shard}
    layer2sublayer = {layer: set() for layer in set(sublayer2layer.values())}
    for k, v in sublayer2layer.items():
        layer2sublayer[v].add(k)
    layer2shard = {layer: set(sublayer2shard[s])
                   for layer, subs in layer2sublayer.items()
                   for s in subs}
    layer_list = sorted(layer2shard, key=lambda l: (min(layer2shard[l]), len(layer2shard[l])))

    state_dict = {}
    loaded_shards = set()
    for layer in tqdm(layer_list):
        sublayers = layer2sublayer[layer]
        needed_shards = set(sublayer2shard[s] for s in sublayers)
        for new_shard in needed_shards - loaded_shards:
            loaded_shards.add(new_shard)
            state_dict.update(torch.load(f"{bin_dir}/{new_shard}", map_location='cpu'))

        # state dict of the current layer
        layer_state_dict = dict([(k, v) for k, v in state_dict.items() if k.startswith(f'{layer}.')])
        assert len(layer_state_dict) == len(
            sublayers), f"Should have {len(sublayers)} keys for {layer}, But get {len(layer_state_dict)} keys"
        save_file(layer_state_dict, f"{new_file_dir}/{layer}.safetensors")

        # Free memory
        for k in layer_state_dict.keys():
            del state_dict[k]
        del layer_state_dict
        gc.collect()


def main(args):
    split_into_layers(args.bin_dir, args.new_file_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split model weights into layers.')
    parser.add_argument('bin_dir', type=str, help='Path to the HF weights directory')
    parser.add_argument('new_file_dir', type=str, help='Path to the new layer-wise file directory')

    args = parser.parse_args()
    main(args)