# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from functools import partial

from torch import nn
from einops import rearrange

from timm.models.layers import to_2tuple


def check_upstream_shape(x, img_size=(224, 224)):
    _, _, H, W = x.shape
    assert H == img_size[0] and W == img_size[1], \
        f"Input image size ({H}*{W}) doesn't match model ({img_size[0]}*{img_size[1]})."


def reshape2n(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def reshape2hw(x, hw=None):
    n = x.shape[1]
    if hw is None:
        hw = to_2tuple(int(n ** 0.5))
    assert n == hw[0] * hw[1], f"N={n} is not equal to H={hw[0]}*W={hw[1]}"
    return rearrange(x, 'b (h w) c -> b c h w', h=hw[0])


def downsample_conv(in_channels, out_channels, kernel_size=2, stride=2, padding=0, dilation=1, norm_layer=None):
    assert norm_layer is None, "only support default normalization"
    norm_layer = norm_layer or partial(nn.GroupNorm, num_groups=1, num_channels=out_channels)
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, bias=False),
                         norm_layer()
                         )


class Reshape2N(nn.Module):
    def __init__(self):
        super(Reshape2N, self).__init__()

    def forward(self, x):
        return reshape2n(x)


class Reshape2HW(nn.Module):
    def __init__(self, hw=None):
        super(Reshape2HW, self).__init__()
        self.hw = hw

    def forward(self, x):
        return reshape2hw(x, self.hw)


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, dilation=1, norm_layer=None):
        super(DownsampleConv, self).__init__()
        self.net = nn.Sequential(
            Reshape2HW(),
            downsample_conv(in_channels, out_channels, kernel_size, stride, padding, dilation, norm_layer),
            Reshape2N()
        )

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.net(x)

    def flops(self, input_shape):
        _, N, C = input_shape  # C == out_channels
        flops = 0
        flops += N * self.out_channels * self.in_channels * self.kernel_size**2
        flops += N * self.out_channels
        return flops
