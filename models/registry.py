# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from timm.models.registry import register_model
from .smlp import sMLPNet
from .spach import Spach, SpachMS
from .shiftvit import ShiftViT


# sMLP
@register_model
def smlpnet_tiny(pretrained=False, **kwargs):
    model = sMLPNet(dim=80, alpha=3, patch_size=4, depths=[2,8,14,2], dp_rate=0.0, **kwargs)
    return model


@register_model
def smlpnet_small(pretrained=False, **kwargs):
    model = sMLPNet(dim=96, alpha=3, patch_size=4, depths=[2,10,24,2], dp_rate=0.2, **kwargs)
    return model


@register_model
def smlpnet_base(pretrained=False, **kwargs):
    model = sMLPNet(dim=112, alpha=3, patch_size=4, depths=[2,10,24,2], dp_rate=0.3, **kwargs)
    return model


# SPACH
@register_model
def spach_xxs_patch16_224_mlp(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=16, hidden_dim=384, token_ratio=0.5, num_heads=12, channel_ratio=2.0)
    cfgs['net_arch'] = [('mlp', 12)]
    cfgs.update(kwargs)
    model = Spach(**cfgs)
    return model


@register_model
def spach_xxs_patch16_224_conv(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=16, hidden_dim=384, token_ratio=0.5, num_heads=12, channel_ratio=2.0)
    cfgs['net_arch'] = [('pass', 12)]
    cfgs.update(kwargs)
    model = Spach(**cfgs)
    return model


@register_model
def spach_xxs_patch16_224_attn(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=16, hidden_dim=192, token_ratio=0.5, num_heads=6, channel_ratio=2.0)
    cfgs['net_arch'] = [('attn', 12)]
    cfgs.update(kwargs)
    model = Spach(**cfgs)
    return model


@register_model
def spach_xs_patch16_224_mlp(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=16, hidden_dim=384, token_ratio=0.5, num_heads=12, channel_ratio=2.0)
    cfgs['net_arch'] = [('mlp', 24)]
    cfgs.update(kwargs)
    model = Spach(**cfgs)
    return model


@register_model
def spach_xs_patch16_224_conv(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=16, hidden_dim=384, token_ratio=0.5, num_heads=12, channel_ratio=2.0)
    cfgs['net_arch'] = [('pass', 24)]
    cfgs.update(kwargs)
    model = Spach(**cfgs)
    return model


@register_model
def spach_xs_patch16_224_attn(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=16, hidden_dim=384, token_ratio=0.5, num_heads=12, channel_ratio=2.0)
    cfgs['net_arch'] = [('attn', 12)]
    cfgs.update(kwargs)
    model = Spach(**cfgs)
    return model


@register_model
def spach_s_patch16_224_mlp(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=16, hidden_dim=512, token_ratio=0.5, num_heads=16, channel_ratio=3.0)
    cfgs['net_arch'] = [('mlp', 24)]
    cfgs.update(kwargs)
    model = Spach(**cfgs)
    return model


@register_model
def spach_s_patch16_224_conv(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=16, hidden_dim=512, token_ratio=0.5, num_heads=16, channel_ratio=3.0)
    cfgs['net_arch'] = [('pass', 24)]
    cfgs.update(kwargs)
    model = Spach(**cfgs)
    return model


@register_model
def spach_s_patch16_224_attn(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=16, hidden_dim=512, token_ratio=0.5, num_heads=16, channel_ratio=3.0)
    cfgs['net_arch'] = [('attn', 12)]
    cfgs.update(kwargs)
    model = Spach(**cfgs)
    return model


@register_model
def spach_ms_xxs_patch4_224_conv(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=64, token_ratio=0.5, num_heads=2, channel_ratio=2.0)
    cfgs['net_arch'] = [[('pass', 2)], [('pass', 2)], [('pass', 6)], [('pass', 2)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


@register_model
def spach_ms_xxs_patch4_224_mlp(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=64, token_ratio=0.5, num_heads=2, channel_ratio=2.0)
    cfgs['net_arch'] = [[('pass', 2)], [('mlp', 2)], [('mlp', 6)], [('mlp', 2)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


@register_model
def spach_ms_xxs_patch4_224_attn(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=32, token_ratio=0.5, num_heads=1, channel_ratio=2.0)
    cfgs['net_arch'] = [[('pass', 2)], [('attn', 2)], [('attn', 6)], [('attn', 2)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


@register_model
def spach_ms_xs_patch4_224_conv(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=96, token_ratio=0.5, num_heads=3, channel_ratio=2.0)
    cfgs['net_arch'] = [[('pass', 3)], [('pass', 4)], [('pass', 12)], [('pass', 3)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


@register_model
def spach_ms_xs_patch4_224_mlp(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=96, token_ratio=0.5, num_heads=3, channel_ratio=2.0)
    cfgs['net_arch'] = [[('pass', 3)], [('mlp', 4)], [('mlp', 12)], [('mlp', 3)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


@register_model
def spach_ms_xs_patch4_224_attn(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=64, token_ratio=0.5, num_heads=2, channel_ratio=2.0)
    cfgs['net_arch'] = [[('pass', 3)], [('attn', 4)], [('attn', 12)], [('attn', 3)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


@register_model
def spach_ms_s_patch4_224_conv(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=128, token_ratio=0.5, num_heads=4, channel_ratio=3.0)
    cfgs['net_arch'] = [[('pass', 3)], [('pass', 4)], [('pass', 12)], [('pass', 3)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


@register_model
def spach_ms_s_patch4_224_mlp(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=128, token_ratio=0.5, num_heads=4, channel_ratio=3.0)
    cfgs['net_arch'] = [[('pass', 3)], [('mlp', 4)], [('mlp', 12)], [('mlp', 3)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


@register_model
def spach_ms_s_patch4_224_attn(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=96, token_ratio=0.5, num_heads=3, channel_ratio=3.0)
    cfgs['net_arch'] = [[('pass', 3)], [('attn', 4)], [('attn', 12)], [('attn', 3)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


@register_model
def spach_ms_xs_patch4_224_hybrid(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=96, token_ratio=0.5, num_heads=3, channel_ratio=2.0)
    cfgs['net_arch'] = [[('pass', 3)], [('pass', 4)], [('pass', 2), ('attn', 10)], [('pass', 1), ('attn', 2)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


@register_model
def spach_ms_s_patch4_224_hybrid(pretrained=False, **kwargs):
    cfgs = dict(img_size=224, patch_size=4, hidden_dim=128, token_ratio=0.5, num_heads=4, channel_ratio=3.0)
    cfgs['net_arch'] = [[('pass', 3)], [('pass', 2), ('attn', 2)], [('pass', 2), ('attn', 10)], [('pass', 1), ('attn', 2)]]
    cfgs.update(kwargs)
    model = SpachMS(**cfgs)
    return model


# shift vit
@register_model
def shiftvit_light_tiny(**kwargs):
    model = ShiftViT(embed_dim=96, depths=(2, 2, 6, 2), mlp_ratio=4, drop_path_rate=0.2, n_div=12)
    return model


@register_model
def shiftvit_r4_tiny(**kwargs):
    model = ShiftViT(embed_dim=96, depths=(2, 2, 12, 3), mlp_ratio=4, drop_path_rate=0.2, n_div=12)
    return model


@register_model
def shiftvit_r2_tiny(**kwargs):
    model = ShiftViT(embed_dim=96, depths=(6, 8, 18, 6), mlp_ratio=2, drop_path_rate=0.2, n_div=12)
    return model


@register_model
def shiftvit_light_small(**kwargs):
    model = ShiftViT(embed_dim=96, depths=(2, 2, 18, 2), mlp_ratio=4, drop_path_rate=0.4, n_div=12)
    return model


@register_model
def shiftvit_r4_small(**kwargs):
    model = ShiftViT(embed_dim=96, depths=(2, 6, 24, 4), mlp_ratio=4, drop_path_rate=0.4, n_div=12)
    return model


@register_model
def shiftvit_r2_small(**kwargs):
    model = ShiftViT(embed_dim=96, depths=(10, 18, 36, 10), mlp_ratio=2, drop_path_rate=0.4, n_div=12)
    return model


@register_model
def shiftvit_light_base(**kwargs):
    model = ShiftViT(embed_dim=128, depths=(2, 2, 18, 2), mlp_ratio=4, drop_path_rate=0.5, n_div=16)
    return model


@register_model
def shiftvit_r4_base(**kwargs):
    model = ShiftViT(embed_dim=128, depths=(4, 6, 22, 4), mlp_ratio=4, drop_path_rate=0.5, n_div=16)
    return model


@register_model
def shiftvit_r2_base(**kwargs):
    model = ShiftViT(embed_dim=128, depths=(10, 18, 36, 10), mlp_ratio=2, drop_path_rate=0.6, n_div=16)
    return model
