# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from torch import nn
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BN_Activ_Conv(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=(1, 1), dilation=(1, 1), groups=1):
        super(BN_Activ_Conv, self).__init__()
        self.BN = nn.BatchNorm2d(out_channels)
        self.Activation = activation
        padding = [int((dilation[j] * (kernel_size[j] - 1) - stride[j] + 1) / 2) for j in range(2)]  # Same padding
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=False)

    def forward(self, img):
        img = self.BN(img)
        img = self.Activation(img)
        img = self.Conv(img)
        return img


class sMLPBlock(nn.Module):
    def __init__(self, W, H, channels):
        super().__init__()
        assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels)
        self.proj_h = nn.Conv2d(H, H, (1, 1))
        self.proh_w = nn.Conv2d(W, W, (1, 1))
        self.fuse = nn.Conv2d(channels*3, channels, (1,1), (1,1), bias=False)

    def forward(self, x):
        x = self.activation(self.BN(x))
        x_h = self.proj_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proh_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = self.fuse(torch.cat([x, x_h, x_w], dim=1))
        return x


class DWConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_merge = BN_Activ_Conv(channels, nn.GELU(), channels, (3, 3), groups=channels)

    def forward(self, img):
        img = self.conv_merge(img)
        return img


class sMLPNet(nn.Module):

    def __init__(self, in_chans=3, dim=80, alpha=3, num_classes=1000, patch_size=4, image_size=224, depths=[2,8,14,2], dp_rate=0.,
                 **kwargs):
        super(sMLPNet, self).__init__()
        '''
        (B,H,W,C): (B,(image_size// patch_size)**2,dim)
        '''

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = image_size // patch_size
        self.depths = depths

        self.to_patch_embedding = nn.ModuleList([])
        self.token_mix = nn.ModuleList([])
        self.channel_mix = nn.ModuleList([])
        self.drop_path = nn.ModuleList([])

        net_num_blocks = sum(self.depths)
        net_block_idx = 0
        for i in range(len(self.depths)):
            ratio = 2 ** i
            if i == 0:
                self.to_patch_embedding.append(nn.Sequential(nn.Conv2d(in_chans, dim, patch_size, patch_size, bias=False)))
            else:
                self.to_patch_embedding.append(nn.Sequential(nn.Conv2d(dim * ratio // 2, dim * ratio, 2, 2, bias=False)))

            for j in range(self.depths[i]):
                block_dpr = dp_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
                self.drop_path.append(DropPath(block_dpr) if block_dpr > 0. else nn.Identity())
                net_block_idx += 1

                self.channel_mix.append(nn.Sequential(
                                     Rearrange('b c h w -> b h w c'),
                                     nn.LayerNorm(dim*ratio),
                                     FeedForward(dim*ratio,dim*ratio*alpha),
                                     Rearrange('b h w c -> b c h w'))
                                     )

                self.token_mix.append(nn.Sequential(DWConvBlock(dim*ratio), sMLPBlock(self.num_patch//ratio, self.num_patch//ratio, dim * ratio)))

        self.batch_norm = nn.BatchNorm2d(dim*2**(len(self.depths)-1))

        self.mlp_head = nn.Sequential(
            nn.Linear(dim * 2**(len(self.depths)-1), num_classes)
        )

    def forward(self, x):

        shift = 0
        for i in range(len(self.depths)):
            x = self.to_patch_embedding[i](x)
            for j in range(self.depths[i]):
                x = x + self.drop_path[j+shift](self.token_mix[j+shift](x))
                x = x + self.drop_path[j+shift](self.channel_mix[j+shift](x))
            shift += self.depths[i]

        x = self.batch_norm(x)

        x = x.mean(dim=[2,3]).flatten(1)

        return self.mlp_head(x)
