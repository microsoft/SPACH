# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from functools import partial

import torch
from torch import nn
from timm.models.layers import DropPath
from einops.layers.torch import Reduce

from .layers import DWConv, SPATIAL_FUNC, ChannelMLP, STEM_LAYER
from .misc import reshape2n


class MixingBlock(nn.Module):
    def __init__(self, dim,
                 spatial_func=None, scaled=True, init_values=1e-4, shared_spatial_func=False,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop_path=0., cpe=True,
                 num_heads=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  # attn
                 in_features=None, hidden_features=None, drop=0.,  # mlp
                 channel_ratio=2.0
                 ):
        super(MixingBlock, self).__init__()

        spatial_kwargs = dict(act_layer=act_layer,
                              in_features=in_features, hidden_features=hidden_features, drop=drop,  # mlp
                              dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop  # attn
                              )

        self.valid_spatial_func = True

        if spatial_func is not None:
            if shared_spatial_func:
                self.spatial_func = spatial_func
            else:
                self.spatial_func = spatial_func(**spatial_kwargs)
            self.norm1 = norm_layer(dim)
            if scaled:
                self.gamma_1 = nn.Parameter(init_values * torch.ones(1, 1, dim), requires_grad=True)
            else:
                self.gamma_1 = 1.
        else:
            self.valid_spatial_func = False

        self.channel_func = ChannelMLP(in_features=dim, hidden_features=int(dim*channel_ratio), act_layer=act_layer,
                                       drop=drop)

        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.cpe = cpe
        if cpe:
            self.cpe_net = DWConv(dim)


    def forward(self, x):
        in_x = x
        if self.valid_spatial_func:
            x = x + self.drop_path(self.gamma_1 * self.spatial_func(self.norm1(in_x)))
        if self.cpe:
            x = x + self.cpe_net(in_x)

        x = x + self.drop_path(self.channel_func(self.norm2(x)))

        return x

    def flops(self, input_shape):
        _, N, C = input_shape
        flops = 0
        if self.valid_spatial_func:
            flops += self.spatial_func.flops(input_shape)
            flops += N * C * 2  # norm + skip
        if self.cpe:
            flops += self.cpe_net.flops(input_shape)

        flops += self.channel_func.flops(input_shape)
        flops += N * C * 2
        return flops


class Spach(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 img_size=224,
                 in_chans=3,
                 hidden_dim=384,
                 patch_size=16,
                 net_arch=None,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 stem_type='conv1',
                 scaled=True, init_values=1e-4, drop_path_rate=0., cpe=True, shared_spatial_func=False,  # mixing block
                 num_heads=12, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,  # attn
                 token_ratio=0.5, channel_ratio=2.0, drop_rate=0.,  # mlp
                 downstream=False,
                 **kwargs
                 ):
        super(Spach, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.downstream = downstream

        self.stem = STEM_LAYER[stem_type](
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=hidden_dim, downstream=downstream)
        self.norm1 = norm_layer(hidden_dim)

        block_kwargs = dict(dim=hidden_dim, scaled=scaled, init_values=init_values, cpe=cpe,
                            shared_spatial_func=shared_spatial_func, norm_layer=norm_layer, act_layer=act_layer,
                            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop,  # attn
                            in_features=self.stem.num_patches, hidden_features=int(self.stem.num_patches * token_ratio), channel_ratio=channel_ratio, drop=drop_rate)  # mlp

        self.blocks = self.make_blocks(net_arch, block_kwargs, drop_path_rate, shared_spatial_func)
        self.norm2 = norm_layer(hidden_dim)

        if not downstream:
            self.pool = Reduce('b n c -> b c', reduction='mean')
            self.head = nn.Linear(hidden_dim, self.num_classes)

        self.init_weights()

    def make_blocks(self, net_arch, block_kwargs, drop_path, shared_spatial_func):
        if shared_spatial_func:
            assert len(net_arch) == 1, '`shared_spatial_func` only support unitary spatial function'
            assert net_arch[0][0] != 'pass', '`shared_spatial_func` do not support pass'
            spatial_func = SPATIAL_FUNC[net_arch[0][0]](**block_kwargs)
        else:
            spatial_func = None
        blocks = []
        for func_type, depth in net_arch:
            for i in range(depth):
                blocks.append(MixingBlock(spatial_func=spatial_func or SPATIAL_FUNC[func_type], drop_path=drop_path,
                                          **block_kwargs))
        return nn.Sequential(*blocks)

    def init_weights(self):
        for n, m in self.named_modules():
            _init_weights(m, n)

    def forward_features(self, x):
        x = self.stem(x)
        x = reshape2n(x)
        x = self.norm1(x)

        x = self.blocks(x)
        x = self.norm2(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        shape = (1, self.stem.num_patches, self.hidden_dim)
        # stem
        flops += self.stem.flops()
        flops += sum(shape)
        # blocks
        flops += sum([i.flops(shape) for i in self.blocks])
        flops += sum(shape)
        # head
        flops += self.hidden_dim * self.num_classes
        return flops


def _init_weights(m, n: str):
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                if 'mlp' in n:
                    nn.init.normal_(m.bias, std=1e-6)
                else:
                    nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)