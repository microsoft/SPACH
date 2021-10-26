# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from functools import partial

from torch import nn
from einops.layers.torch import Reduce

from .spach import MixingBlock, _init_weights
from .layers import STEM_LAYER, SPATIAL_FUNC
from .misc import DownsampleConv, reshape2n


class SpachMS(nn.Module):
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
        super(SpachMS, self).__init__()
        assert len(net_arch) == 4
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.downstream = downstream

        self.stem = STEM_LAYER[stem_type](
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=hidden_dim, downstream=downstream)
        self.norm1 = norm_layer(hidden_dim)

        block_kwargs = dict(scaled=scaled, init_values=init_values, cpe=cpe,
                            shared_spatial_func=shared_spatial_func, norm_layer=norm_layer, act_layer=act_layer,
                            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop,  # attn
                            channel_ratio=channel_ratio, drop=drop_rate)  # mlp

        stage_modules = self.make_blocks(hidden_dim, self.stem.num_patches, net_arch, block_kwargs, drop_path_rate,
                                         shared_spatial_func, token_ratio)
        for stage in stage_modules:
            self.add_module(*stage)
        hidden_dim = hidden_dim * 8
        self.norm2 = norm_layer(hidden_dim)

        if not downstream:
            self.pool = Reduce('b n c -> b c', reduction='mean')
            self.head = nn.Linear(hidden_dim, self.num_classes)

        self.init_weights()

    def make_blocks(self, dim, seq_len, net_arch, block_kwargs, drop_path, shared_spatial_func, token_ratio):
        stages = []
        num_blocks = sum(sum([depth for _, depth in stage_arch]) for stage_arch in net_arch)
        block_idx = 0

        for stage_idx, stage_arch in enumerate(net_arch):
            stage_name = f'layer{stage_idx + 1}'
            blocks = []
            if stage_idx > 0:
                down_kwargs = dict(in_channels=dim, out_channels=dim * 2)
                downsample = DownsampleConv(**down_kwargs)
                blocks.append(downsample)
                dim = dim * 2
                seq_len = seq_len // 4

            block_kwargs.update(dict(dim=dim, in_features=seq_len, hidden_features=int(seq_len * token_ratio)))

            if stage_idx > 0 and shared_spatial_func:
                assert len(stage_arch) == 1, '`shared_spatial_func` only support unitary spatial function'
                assert stage_arch[0][0] != 'pass', '`shared_spatial_func` do not support pass'
                spatial_func = SPATIAL_FUNC[stage_arch[0][0]](**block_kwargs)
            else:
                spatial_func = None

            for func_type, depth in stage_arch:
                for i in range(depth):
                    block_dpr = drop_path * block_idx / (num_blocks - 1)  # stochastic depth linear decay rule
                    blocks.append(MixingBlock(spatial_func=spatial_func or SPATIAL_FUNC[func_type], drop_path=block_dpr,
                                              **block_kwargs))
                    block_idx += 1
            stages.append((stage_name, nn.Sequential(*blocks)))

        return stages

    def init_weights(self):
        for n, m in self.named_modules():
            _init_weights(m, n)

    def forward_features(self, x):
        x = self.stem(x)
        x = reshape2n(x)
        x = self.norm1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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
        # layer1,2,3,4
        flops += sum([i.flops(shape) for i in self.layer1])
        shape = (1, self.stem.num_patches//4, self.hidden_dim*2)
        flops += sum([i.flops(shape) for i in self.layer2])
        shape = (1, self.stem.num_patches//16, self.hidden_dim*4)
        flops += sum([i.flops(shape) for i in self.layer3])
        shape = (1, self.stem.num_patches//64, self.hidden_dim*8)
        flops += sum([i.flops(shape) for i in self.layer4])
        flops += sum(shape)
        # head
        flops += self.hidden_dim * 8 * self.num_classes
        return flops