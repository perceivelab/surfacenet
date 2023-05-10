import os
import random
from pathlib import Path

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from datasets.utils import textures_mapping, texture_maps


def MapTransform(load_size=256):
    return T.Compose([
        T.Resize(load_size),
        T.CenterCrop(load_size),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])


class SurfaceDataset(Dataset):
    def __init__(self, dset_dir, load_size=256):
        self.dset_dir = Path(dset_dir)

        self.materials = []

        self.materials = [{"name": x, "folder": self.dset_dir / x}
                          for x in os.listdir(self.dset_dir)]

        self.load_size = load_size
        self.transforms = MapTransform(load_size)

    def __len__(self):
        '''Dataset size'''
        return len(self.materials)

    def __getitem__(self, i):
        material = self.materials[i]
        folder = material["folder"]

        maps = {}

        for texture_map in texture_maps:
            src = folder/(texture_map+".jpg")
            image = Image.open(src).convert("RGB")

            if textures_mapping[texture_map] == 1:
                image = image.convert("L")

            maps[texture_map] = self.transforms(image)

        # pick random render among available renders
        # (useful wen using multiple renderings with different illumination)
        folder_render = folder/"render"
        render = random.sample(os.listdir(folder_render), 1)[0]
        render = Image.open(folder_render/render).convert("RGB")
        maps["Render"] = self.transforms(render)

        return maps

class PicturesDataset(Dataset):
    def __init__(self, dset_dir, load_size=256):

        self.dset_dir = Path(dset_dir)
        self.files = list(self.dset_dir.glob('*.jpg'))

        self.transforms = MapTransform(load_size)

    def __len__(self):
        '''Dataset size'''
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        
        img = Image.open(file).convert("RGB")
        img = self.transforms(img)

        return img