import torch
from datasets.utils import texture_maps

def make_plot_maps(inputs, outputs):
    B, _, H, W = inputs.shape
    map_size = [H, W]

    def chain_maps(maps, i):
        render = inputs[i].unsqueeze(0)
        return (torch.cat([render, *[maps[key][i].unsqueeze(0).expand(1, 3, *map_size) for key in texture_maps]], 0) + 1) / 2

    maps = []

    for i in range(B):
        maps.append(chain_maps(outputs, i))

    return torch.cat(maps, 0)