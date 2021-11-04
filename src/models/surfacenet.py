import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.resnet import BasicBlock as ResidualBlock
from torchvision.models.resnet import resnet50 as resnet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from datasets.utils import *

def upsample_conv(in_channels=256, out_channels=256):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels,
                           kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        ResidualBlock(out_channels, out_channels)
    )


def same_conv(in_channels=256, out_channels=256):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels,
                           kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        ResidualBlock(out_channels, out_channels)
    )


class Backbone(nn.Module):
    out_feat = 256

    def __init__(self):
        super(Backbone, self).__init__()

        encoder = resnet(replace_stride_with_dilation=[
                          False, True, True], pretrained=True)

        self.encoder = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,

            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )

        self.bottleneck = DeepLabHead(2048, 256)

        self.decoder = nn.ModuleList([
            upsample_conv(in_channels=256+3,
                          out_channels=self.out_feat),
            upsample_conv(in_channels=256+3,
                          out_channels=self.out_feat),
            upsample_conv(in_channels=256+3,
                          out_channels=self.out_feat),
            same_conv(in_channels=256+3,
                      out_channels=self.out_feat)
        ])

    def forward(self, x, **kwargs):
        feat = self.encoder(x)
        feat = self.bottleneck(feat)

        for layer in self.decoder:
            pooled_x = F.interpolate(
                x, feat.shape[-2], mode="bilinear", align_corners=False)
            feat = layer(torch.cat((feat, pooled_x), 1))

        return feat


class SurfaceNet(nn.Module):
    padding = 32

    def __init__(self, **kwargs):
        super(SurfaceNet, self).__init__()
        
        self.encoder = Backbone()

        self.out_feat = self.encoder.out_feat

        self.heads = nn.ModuleDict()

        # create one head for each requested map
        for map in texture_maps:
            out_features = textures_mapping[map]
            self.heads.add_module(map, self.__make_head(
                self.out_feat, out_features))

    def __make_head(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x, pad_image=False):
        _, _, h, w = x.shape

        x = F.pad(x, pad=(32, 32, 32, 32), mode="replicate") if pad_image else x

        out = self.encoder(x)

        maps = {}

        for key in self.heads.keys():
            out_map = self.heads[key](out)
            out_map = TF.center_crop(out_map, (h, w))
            maps[key] = out_map.clamp(-1, 1)

        return maps
