import torch
import torch.nn as nn
from torchvision.models import inception, resnet34 as resnet

from datasets.utils import *


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        # in_channels is computed as the sum of channels per map + the channesl for the rendering (3)
        in_channels = 3 + sum([textures_mapping[x] for x in texture_maps])

        self.main = NLayerDiscriminator(in_channels=in_channels, n_layers=4)

    def forward(self, x):
        out = self.main(x)

        return out


class ImageDiscriminator(nn.Module):
    def __init__(self):
        super(ImageDiscriminator, self).__init__()
        # in_channels is computed as the sum of channels per map + the channesl for the rendering (3)
        in_channels = 3 + sum([textures_mapping[x] for x in texture_maps])

        n_layers = 6

        self.main = NLayerDiscriminator(in_channels=in_channels, n_layers=n_layers, final_classifier=False)

        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(2),
            nn.Conv2d(512, 1, kernel_size=2)
        )


    def forward(self, x):
        out = self.main(x)
        out = self.classifier(out)
        return out

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_features=64, n_layers=3, norm_layer=nn.BatchNorm2d, final_classifier=True, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1

        sequence = [
            nn.Conv2d(in_channels, base_features, kernel_size=kernel_size,
                      stride=2, padding=padding),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(base_features * nf_mult_prev, base_features * nf_mult,
                          kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(base_features * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(base_features * nf_mult_prev, base_features * nf_mult,
                      kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
            norm_layer(base_features * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        if final_classifier:
            sequence += [nn.Conv2d(base_features * nf_mult, 1,
                               kernel_size=kernel_size, stride=1, padding=padding)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
