from torch import nn

from timm.models.layers import to_2tuple

from ..misc import check_upstream_shape


class PatchEmbed(nn.Module):
    """1-conv patch embedding layer"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, downstream=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.downstream = downstream
        self.img_size = img_size
        self.patch_size = patch_size
        self.stem_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.stem_shape[0] * self.stem_shape[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.out_size = None

        # for flops
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x):
        if not self.downstream:
            check_upstream_shape(x, self.img_size)
        x = self.proj(x)
        return x

    def flops(self, input_shape=None):
        flops = self.num_patches * self.embed_dim * (sum(self.patch_size) * self.in_chans + 1)  # Ho*Wo*Co*(K^2*Ci+1)
        return flops


class Conv4PatchEmbed(nn.Module):
    """4-conv patch embedding layer"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, downstream=False, hidden_chans=64):
        super(Conv4PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.downstream = downstream
        self.img_size = img_size
        self.patch_size = patch_size
        self.stem_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.stem_shape[0] * self.stem_shape[1]

        sub_patch_size = (patch_size[0]//2, patch_size[1]//2)

        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hidden_chans),
            nn.ReLU(),
            nn.Conv2d(hidden_chans, hidden_chans, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_chans),
            nn.ReLU(),
            nn.Conv2d(hidden_chans, hidden_chans, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_chans),
            nn.ReLU(),
            nn.Conv2d(hidden_chans, embed_dim, kernel_size=sub_patch_size, stride=sub_patch_size)
        )

        # for flops
        self.inside_num_patches = self.num_patches * sum(sub_patch_size)
        self.in_chans = in_chans
        self.new_patch_size = sub_patch_size
        self.embed_dim = embed_dim
        self.hidden_chans = hidden_chans

    def forward(self, x):
        if not self.downstream:
            check_upstream_shape(x, self.img_size)
        x = self.proj(x)
        return x

    def flops(self, input_shape=None):
        flops = 0
        flops += self.inside_num_patches * self.hidden_chans * self.in_chans * 7 * 7  # Ho*Wo*Co*K^2*Ci+1
        flops += self.inside_num_patches * self.hidden_chans

        flops += self.inside_num_patches * self.hidden_chans * self.hidden_chans * 3 * 3
        flops += self.inside_num_patches * self.hidden_chans

        flops += self.inside_num_patches * self.hidden_chans * self.hidden_chans * 3 * 3
        flops += self.inside_num_patches * self.hidden_chans

        flops += self.num_patches * self.embed_dim * (sum(self.new_patch_size)*self.hidden_chans + 1)  # Ho*Wo*Co*(K^2*Ci+1)

        return flops


STEM_LAYER = {'conv1': PatchEmbed, 'conv4': Conv4PatchEmbed}
