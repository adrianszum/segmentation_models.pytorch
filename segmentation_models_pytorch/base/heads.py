import torch.nn as nn
from .modules import Flatten, Activation


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


class ProjectHead(nn.Sequential):
    """
    Implements projection head for contrastive learning as per
    "Exploring Cross-Image Pixel Contrast for Semantic Segmentation"
    https://arxiv.org/abs/2101.11939
    https://github.com/tfzhou/ContrastiveSeg

    Provides high-dimensional L2-normalized pixel embeddings (256-d from 1x1 conv by default)
    """

    def __init__(self, in_channels: int, out_channels: int = 256, kernel_size: int = 1, upsampling: int = 1):
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        relu = nn.ReLU(inplace=True)
        conv2d_2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # TODO decide where to put upsampling
        super().__init__(upsampling, conv2d_1, relu, conv2d_2, relu)

    def forward(self, x):
        x = super().forward(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
