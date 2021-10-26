from torch import nn
from einops import rearrange

from ..misc import Reshape2HW, Reshape2N


class SpatialAttention(nn.Module):
    """Spatial Attention"""
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., **kwargs):
        super(SpatialAttention, self).__init__()
        head_dim = dim // num_heads

        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (three heads head_c) -> three b heads n head_c", three=3, heads=self.num_heads)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))  # B, head, N, N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  # B, head, N, C
        out = rearrange(out, "b heads n head_c -> b n (heads head_c)")

        out = self.proj(out)
        out = self.proj_drop(out)

        return out

    def flops(self, input_shape):
        _, N, C = input_shape
        flops = 0
        # qkv
        flops += 3 * C * C * N
        # q@k
        flops += N ** 2 * C
        # attn@v
        flops += N ** 2 * C
        # proj
        flops += C * C * N
        return flops

class SpatialMLP(nn.Module):
    """Spatial MLP"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., **kwargs):
        super(SpatialMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.transpose(1, 2)
        return x

    def flops(self, input_shape):
        _, N, C = input_shape
        flops = 0
        flops += (N + 1) * self.hidden_features * C
        flops += (self.hidden_features + 1) * self.out_features * C
        return flops


class DWConv(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(DWConv, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(Reshape2HW(),
                                 nn.Conv2d(dim, dim, kernel_size, 1, padding, groups=dim),
                                 Reshape2N())


    def forward(self, x):
        x = self.net(x)
        return x

    def flops(self, input_shape):
        _, N, C = input_shape
        flops = N * self.dim * (3 * 3 + 1)
        return flops


SPATIAL_FUNC = {'attn': SpatialAttention, 'mlp': SpatialMLP, 'pass': None}
