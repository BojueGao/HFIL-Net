# -*- coding: utf-8 -*-
"""
@author: gaohaoran@Dalian Minzu University
@software: PyCharm
@file: HFIL-Net.py
@time: 2023/11/23 7:23
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C) 堆叠到一起形成一个长条
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每一个头的通道维数
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])

        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1---Important！！！
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # 输入此的x是整图
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:

            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print('FFN',x.shape)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans  # define in_chans == 3
        self.embed_dim = embed_dim  # Swin-B.embed_dim ==128,(T is 96)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # dim 3->128
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints,尺寸固定，下有断言
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=384, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=128, depths=None, num_heads=None,
                 window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if depths is None:
            depths = [2, 2, 6, 2]
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            # self.layers 中应该是 4 个
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        layer_features = []
        x = self.patch_embed(x)
        B, L, C = x.shape
        layer_features.append(x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous())

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
            B, L, C = x.shape
            xl = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
            layer_features.append(xl)
        x = self.norm(x)  # B L C
        B, L, C = x.shape
        x = x.view(B, int(np.sqrt(L)), int(np.sqrt(L)), -1).permute(0, 3, 1, 2).contiguous()
        layer_features[-1] = x

        return layer_features

    def forward(self, x):
        outs = self.forward_features(x)

        return outs

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class HFILNet(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(HFILNet, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])

        # ---------- Uni-modal Cross-Scale Interaction Block---------------------------------------------------
        self.cross_scales1 = UCIB1(1024, 512)
        self.cross_scales2 = UCIB2(512, 256)
        self.cross_scales3 = UCIB2(256, 128)

        self.f_cs1_conv = conv3x3_bn_relu(512, 64)
        self.f_cs2_conv = conv3x3_bn_relu(256, 64)
        self.f_cs3_conv = conv3x3_bn_relu(128, 64)
        self.f_cs4_conv = conv3x3_bn_relu(512, 64)
        self.f_cs5_conv = conv3x3_bn_relu(256, 64)
        self.f_cs6_conv = conv3x3_bn_relu(128, 64)

        # ---------- Cross-Modal Same-Scale Interaction Block --------------------------------------------------
        self.cross_modal1 = CSIB(1024)
        self.cross_modal2 = CSIB(512)
        self.cross_modal3 = CSIB(256)

        self.f_cm1_conv = conv3x3_bn_relu(1024, 64)
        self.f_cm2_conv = conv3x3_bn_relu(512, 64)
        self.f_cm3_conv = conv3x3_bn_relu(256, 64)
        self.f_cm4_conv = conv3x3_bn_relu(1024, 64)
        self.f_cm5_conv = conv3x3_bn_relu(512, 64)
        self.f_cm6_conv = conv3x3_bn_relu(256, 64)

        # ---------- Cross-Modal Cross-Scale Interaction Block --------------------------------------------------
        self.cross_scales_modal1 = CCIB(512)
        self.cross_scales_modal2 = CCIB(256)
        self.cross_scales_modal3 = CCIB(128)

        self.f_csm1_conv = conv3x3_bn_relu(512, 64)
        self.f_csm2_conv = conv3x3_bn_relu(256, 64)
        self.f_csm3_conv = conv3x3_bn_relu(128, 64)
        self.f_csm4_conv = conv3x3_bn_relu(512, 64)
        self.f_csm5_conv = conv3x3_bn_relu(256, 64)
        self.f_csm6_conv = conv3x3_bn_relu(128, 64)

        # ---------- Multi-modal information Adaptive Fusion ----------------------------------------------------
        self.f_fuse1_conv = conv3x3_bn_relu(192, 64)
        self.f_fuse2_conv = conv3x3_bn_relu(256, 64)
        self.f_fuse3_conv = conv3x3_bn_relu(256, 64)
        self.f_fuse4_conv = conv3x3_bn_relu(192, 64)
        self.f_fuse5_conv = conv3x3_bn_relu(256, 64)
        self.f_fuse6_conv = conv3x3_bn_relu(256, 64)
        self.conv128_64_1 = conv3x3_bn_relu(128, 64)
        self.conv128_64_2 = conv3x3_bn_relu(128, 64)
        self.conv128_64_3 = conv3x3_bn_relu(128, 64)

        # ---------- Advanced Semantic Guidance Aggregation ---------------------------------------------------
        self.rgb_fuse1_CA = ChannelAttention(192)
        self.rgb_fuse2_CA = ChannelAttention(256)
        self.rgb_fuse3_CA = ChannelAttention(256)
        self.depth_fuse1_CA = ChannelAttention(192)
        self.depth_fuse2_CA = ChannelAttention(256)
        self.depth_fuse3_CA = ChannelAttention(256)
        self.fuse_fea1_conv = conv3x3_bn_relu(192, 64)
        self.fuse_fea2_conv = conv3x3_bn_relu(192, 64)
        self.fuse_fea3_conv = conv3x3_bn_relu(192, 64)

        # ---------- Feature Integrity Learning and Refinement ---------------------------------------------------
        self.pwv = FILR(64)

        self.pred_end_out1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.pred_end_out2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.pred_end_out3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.pred_end_pose = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.param1 = torch.nn.Parameter(torch.rand([1, 64, 1, 1]), requires_grad=True)
        self.param2 = torch.nn.Parameter(torch.rand([1, 64, 1, 1]), requires_grad=True)
        self.param3 = torch.nn.Parameter(torch.rand([1, 64, 1, 1]), requires_grad=True)

    def forward(self, rgb, depth):
        rgb_list = self.rgb_swin(rgb)
        depth_list = self.depth_swin(depth)

        r4 = rgb_list[0]  # (8,128,96,96)
        r3 = rgb_list[1]  # (8,256,48,48)
        r2 = rgb_list[2]  # (8,512,24,24)
        r1 = rgb_list[3]  # (8,1024,12,12)
        d4 = depth_list[0]  # (8,128,96,96)
        d3 = depth_list[1]  # (8,256,48,48)
        d2 = depth_list[2]  # (8,512,24,24)
        d1 = depth_list[3]  # (8,1024,12,12)

        r2_size = r2.size()[2:]  # (24,24)
        r3_size = r3.size()[2:]  # (48,48)
        r4_size = r4.size()[2:]  # (96,96)

        # ---------- Uni-modal Cross-Scale Interaction Block -----------------------------------------
        # RGB Branch
        rgb_fem1 = self.cross_scales1(r1)  # (8,512,12,12)
        rgb_fem1 = F.interpolate(rgb_fem1, size=r2_size, mode='bilinear')  # (8,512,24,24)
        f_rgb_cs1 = torch.add(r2, rgb_fem1)  # (8,512,24,24)
        f_rgb_cs1 = self.f_cs1_conv(f_rgb_cs1)  # (8,64,24,24)

        rgb_fem2 = self.cross_scales2(r2)  # (8,256,24,24)
        rgb_fem2 = F.interpolate(rgb_fem2, size=r3_size, mode='bilinear')  # (8,256,48,48)
        f_rgb_cs2 = torch.add(r3, rgb_fem2)  # (8,256,48,48)
        f_rgb_cs2 = self.f_cs2_conv(f_rgb_cs2)  # (8,64,48,48)

        rgb_fem3 = self.cross_scales3(r3)  # (8,128,48,48)
        rgb_fem3 = F.interpolate(rgb_fem3, size=r4_size, mode='bilinear')  # (8,128,96,96)
        f_rgb_cs3 = torch.add(r4, rgb_fem3)  # (8,128,96,96)
        f_rgb_cs3 = self.f_cs3_conv(f_rgb_cs3)  # (8,64,96,96)

        # Depth Branch
        depth_fem1 = self.cross_scales1(d1)  # (8,512,12,12)
        depth_fem1 = F.interpolate(depth_fem1, size=r2_size, mode='bilinear')  # (8,512,24,24)
        f_depth_cs1 = torch.add(d2, depth_fem1)  # (8,512,24,24)
        f_depth_cs1 = self.f_cs4_conv(f_depth_cs1)  # (8,64,24,24)

        depth_fem2 = self.cross_scales2(d2)  # (8,256,24,24)
        depth_fem2 = F.interpolate(depth_fem2, size=r3_size, mode='bilinear')  # (8,256,48,48)
        f_depth_cs2 = torch.add(d3, depth_fem2)  # (8,256,48,48)
        f_depth_cs2 = self.f_cs5_conv(f_depth_cs2)  # (8,64,48,48)

        depth_fem3 = self.cross_scales3(d3)  # (8,128,48,48)
        depth_fem3 = F.interpolate(depth_fem3, size=r4_size, mode='bilinear')  # (8,128,96,96)
        f_depth_cs3 = torch.add(d4, depth_fem3)  # (8,128,96,96)
        f_depth_cs3 = self.f_cs6_conv(f_depth_cs3)  # (8,64,96,96)

        # ---------- Cross-Modal Same-Scale Interaction Block -------------------------------------
        rgb_enh1 = self.cross_modal1(r1, d1)  # (8,1024,12,12)
        rgb_enh1 = F.interpolate(rgb_enh1, size=r2_size, mode='bilinear')  # (8,1024,24,24)
        f_rgb_cm1 = self.f_cm1_conv(rgb_enh1)  # (8,64,24,24)

        rgb_enh2 = self.cross_modal2(r2, d2)  # (8,512,24,24)
        rgb_enh2 = F.interpolate(rgb_enh2, size=r3_size, mode='bilinear')  # (8,512,48,48)
        f_rgb_cm2 = self.f_cm2_conv(rgb_enh2)  # (8,64,48,48)

        rgb_enh3 = self.cross_modal3(r3, d3)  # (8,256,48,48)
        rgb_enh3 = F.interpolate(rgb_enh3, size=r4_size, mode='bilinear')  # (8,256,96,96)
        f_rgb_cm3 = self.f_cm3_conv(rgb_enh3)  # (8,64,96,96)

        # ---------- Cross-Modal Cross-Scale Interaction Block --------------------------------------
        # RGB Branch
        rgb_csm1 = self.cross_scales_modal1(r1, d2)  # (8,512,24,24)
        rgb_csm2 = self.cross_scales_modal2(r2, d3)  # (8,256,48,48)
        rgb_csm3 = self.cross_scales_modal3(r3, d4)  # (8,128,96,96)
        f_rgb_csm1 = self.f_csm1_conv(rgb_csm1)  # (8,64,24,24)
        f_rgb_csm2 = self.f_csm2_conv(rgb_csm2)  # (8,64,48,48)
        f_rgb_csm3 = self.f_csm3_conv(rgb_csm3)  # (8,64,96,96)

        # Depth Branch
        depth_csm1 = self.cross_scales_modal1(d1, r2)  # (8,512,24,24)
        depth_csm2 = self.cross_scales_modal2(d2, r3)  # (8,256,48,48)
        depth_csm3 = self.cross_scales_modal3(d3, r4)  # (8,128,96,96)
        f_depth_csm1 = self.f_csm4_conv(depth_csm1)  # (8,64,24,24)
        f_depth_csm2 = self.f_csm5_conv(depth_csm2)  # (8,64,48,48)
        f_depth_csm3 = self.f_csm6_conv(depth_csm3)  # (8,64,96,96)

        # ---------- Advanced Semantic Guidance Aggregation --------------------------------------
        # RGB Branch
        rgb_fuse1 = torch.cat((f_rgb_cs1, f_rgb_cm1, f_rgb_csm1), dim=1)  # (8,192,24,24)
        rgb_fuse1_attention = self.rgb_fuse1_CA(rgb_fuse1)  # (8,192,24,24)  通道注意力
        rgb_fuse1 = rgb_fuse1 + rgb_fuse1 * rgb_fuse1_attention  # (8,192,24,24)  通道注意力
        rgb_fuse1 = self.f_fuse1_conv(rgb_fuse1)  # (8,64,24,24)
        rgb_fuse1_up = F.interpolate(rgb_fuse1, size=r3_size, mode='bilinear')  # (8,64,48,48)

        rgb_fuse2 = torch.cat((f_rgb_cs2, f_rgb_cm2, f_rgb_csm2, rgb_fuse1_up), dim=1)  # (8,256,48,48)
        rgb_fuse2_attention = self.rgb_fuse2_CA(rgb_fuse2)  # (8,256,24,24)
        rgb_fuse2 = rgb_fuse2 + rgb_fuse2 * rgb_fuse2_attention  # (8,256,24,24)
        rgb_fuse2 = self.f_fuse2_conv(rgb_fuse2)  # (8,64,48,48)
        rgb_fuse2_up = F.interpolate(rgb_fuse2, size=r4_size, mode='bilinear')  # (8,64,96,96)

        rgb_fuse3 = torch.cat((f_rgb_cs3, f_rgb_cm3, f_rgb_csm3, rgb_fuse2_up), dim=1)  # (8,256,96,96)
        rgb_fuse3_attention = self.rgb_fuse3_CA(rgb_fuse3)  # (8,256,24,24)
        rgb_fuse3 = rgb_fuse3 + rgb_fuse3 * rgb_fuse3_attention  # (8,256,24,24)
        rgb_fuse3 = self.f_fuse3_conv(rgb_fuse3)  # (8,64,96,96)

        # depth Branch
        depth_fuse1 = torch.cat((f_depth_cs1, f_rgb_cm1, f_depth_csm1), dim=1)  # (8,192,24,24)
        depth_fuse1_attention = self.depth_fuse1_CA(depth_fuse1)  # (8,192,24,24)
        depth_fuse1 = depth_fuse1 + depth_fuse1 * depth_fuse1_attention  # (8,192,24,24)
        depth_fuse1 = self.f_fuse4_conv(depth_fuse1)  # (8,64,24,24)
        depth_fuse1_up = F.interpolate(depth_fuse1, size=r3_size, mode='bilinear')  # (8,64,48,48)

        depth_fuse2 = torch.cat((f_depth_cs2, f_rgb_cm2, f_depth_csm2, depth_fuse1_up), dim=1)  # (8,256,48,48)
        depth_fuse2_attention = self.depth_fuse2_CA(depth_fuse2)  # (8,256,24,24)
        depth_fuse2 = depth_fuse2 + depth_fuse2 * depth_fuse2_attention  # (8,256,24,24)
        depth_fuse2 = self.f_fuse5_conv(depth_fuse2)  # (8,64,48,48)
        depth_fuse2_up = F.interpolate(depth_fuse2, size=r4_size, mode='bilinear')  # (8,64,96,96)

        depth_fuse3 = torch.cat((f_depth_cs3, f_rgb_cm3, f_depth_csm3, depth_fuse2_up), dim=1)  # (8,256,96,96)
        depth_fuse3_attention = self.depth_fuse3_CA(depth_fuse3)  # (8,192,24,24)
        depth_fuse3 = depth_fuse3 + depth_fuse3 * depth_fuse3_attention  # (8,192,24,24)
        depth_fuse3 = self.f_fuse6_conv(depth_fuse3)  # (8,64,96,96)

        # ---------- Multi-modal information Adaptive Fusion ----------------------------------------------------
        p1 = torch.sigmoid(self.param1)
        p2 = torch.sigmoid(self.param2)
        p3 = torch.sigmoid(self.param3)

        add_fea1 = rgb_fuse1 * p1 + depth_fuse1 * (1 - p1)  # (8,64,24,24)
        fuse_fea1 = torch.cat((rgb_fuse1, depth_fuse1, add_fea1), dim=1)  # (8,192,24,24)
        fuse_fea1 = self.fuse_fea1_conv(fuse_fea1)  # (8,64,24,24)

        add_fea2 = rgb_fuse2 * p2 + depth_fuse2 * (1 - p2)  # (8,64,48,48)
        fuse_fea2 = torch.cat((rgb_fuse2, depth_fuse2, add_fea2), dim=1)  # (8,192,48,48)
        fuse_fea2 = self.fuse_fea2_conv(fuse_fea2)  # (8,64,48,48)

        add_fea3 = rgb_fuse3 * p3 + depth_fuse3 * (1 - p3)  # (8,64,96,96)
        fuse_fea3 = torch.cat((rgb_fuse3, depth_fuse3, add_fea3), dim=1)  # (8,192,96,96)
        fuse_fea3 = self.fuse_fea3_conv(fuse_fea3)  # (8,64,96,96)

        # ---------- Feature Integrity Learning and Refinement ------------------------------------------------
        shape = rgb.size()[2:]  # shape:(384,384)
        end_out1, end_out2, end_out3, end_pose = self.pwv(fuse_fea1, fuse_fea2, fuse_fea3)
        pred1 = F.interpolate(self.pred_end_out1(end_out1), size=shape, mode='bilinear')  # (b,1,384,384)
        pred2 = F.interpolate(self.pred_end_out2(end_out2), size=shape, mode='bilinear')  # (b,1,384,384)
        pred3 = F.interpolate(self.pred_end_out3(end_out3), size=shape, mode='bilinear')  # (b,1,384,384)

        return pred1, pred2, pred3

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


# Cross-Modal Cross-Scale Interaction Block
class CCIB(nn.Module):
    def __init__(self, in_channel):
        super(CCIB, self).__init__()
        self.high_conv1 = nn.Sequential(
            nn.Conv2d(2 * in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.high_conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.low_conv1 = nn.Sequential(
            nn.Conv2d(2 * in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.low_conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()

        self.param = torch.nn.Parameter(torch.rand([1, in_channel, 1, 1]), requires_grad=True)

    def forward(self, high, low, high_prior=None, low_prior=None):
        high1 = self.high_conv1(high)  # high:(H/2, W/2, C); high1:(H/2, W/2, C/2)
        high2 = self.high_conv2(low)  # low:(H, W, C/2); high2:(H/2, W/2, C/2)
        _, _, h, w = low.size()  # h, w = (H, W)
        high = F.interpolate(high, size=(h, w), mode='bilinear', align_corners=True)  # high:(H, W)
        low1 = self.low_conv1(high)  # low1:(H, W, C/2)
        low2 = self.low_conv2(low)  # low2:(H, W, C/2)
        high2_down = self.SA1(high2)
        low1_down = self.SA2(low1)

        high1_e = high1 + high1 * high2_down  # (H/2, W/2, C/2)
        high2_e = high2 + high2 * high2_down  # (H/2, W/2, C/2)
        low1_e = low1 + low1 * low1_down  # (H, W, C/2)
        low2_e = low2 + low2 * low1_down  # (H, W, C/2)

        high_add = high1_e + high2_e  # (H/2, W/2, C/2)
        low_add = low1_e + low2_e  # (H, W, C/2)

        high_add = F.interpolate(high_add, size=(h, w), mode='bilinear', align_corners=True)  # (H, W, C/2)

        p = torch.sigmoid(self.param)
        x_last = high_add * p + low_add * (1 - p)

        return x_last


# Cross-Modal Same-Scale Interaction Block
class CSIB(nn.Module):
    def __init__(self, in_channel):
        super(CSIB, self).__init__()
        self.bn_1 = nn.BatchNorm2d(in_channel)
        self.bn_2 = nn.BatchNorm2d(in_channel)
        self.bn_3 = nn.BatchNorm2d(in_channel)
        self.bn_4 = nn.BatchNorm2d(in_channel)

        self.conv_qr = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv_kr = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv_vr = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv_rgb = nn.Conv2d(in_channel, in_channel, kernel_size=1)

        self.conv_qd = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv_kd = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv_vd = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.conv_depth = nn.Conv2d(in_channel, in_channel, kernel_size=1)

        self.conv_1 = nn.Conv2d(in_channel * 2, in_channel, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)

        self.rgb_attention = SpatialAttention()
        self.depth_attention = SpatialAttention()

        self.scale = in_channel ** -0.5

        # --------------------------- 12.0 -------------------------------
        self.param = torch.nn.Parameter(torch.rand([1, in_channel, 1, 1]), requires_grad=True)

    def forward(self, f1, f2):
        f1_r = self.bn_1(f1)
        f1_qr = self.conv_qr(f1_r)
        f1_kr = self.conv_kr(f1_r)
        f1_vr = self.conv_vr(f1_r)

        f2_d = self.bn_2(f2)
        f2_qd = self.conv_qd(f2_d)
        f2_kd = self.conv_kd(f2_d)
        f2_vd = self.conv_vd(f2_d)

        # QKV—RGB----------------------------------
        B, C, H, W = f1_qr.shape
        f1_qr = f1_qr.reshape(B, C, H * W) * self.scale
        f1_kr = f1_kr.reshape(B, C, H * W).transpose(1, 2)
        att_rgb = torch.bmm(f1_kr, f1_qr)
        att_rgb = self.softmax(att_rgb)
        f1_mul = torch.bmm(f2_vd.reshape(B, C, H * W), att_rgb).reshape(B, C, H, W)
        f1_end = f1 + self.relu(self.conv_rgb(self.bn_3(f1_mul)))

        # QKV—Depth----------------------------------
        f2_qd = f2_qd.reshape(B, C, H * W) * self.scale
        f2_kd = f2_kd.reshape(B, C, H * W).transpose(1, 2)
        att_depth = torch.bmm(f2_kd, f2_qd)
        att_depth = self.softmax(att_depth)
        f2_mul = torch.bmm(f1_vr.reshape(B, C, H * W), att_depth).reshape(B, C, H, W)
        f2_end = f2 + self.relu(self.conv_depth(self.bn_4(f2_mul)))

        param = torch.sigmoid(self.param)
        f = f1_end * param + f2_end * (1 - param)

        return f


# Uni-modal Cross-Scale Interaction Block
class UCIB1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UCIB1, self).__init__()
        self.down_sample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_sample4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_sum = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.up_sample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_sample4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, x):
        x = self.bn1(x)  # (8,1024,12,12)

        x_down2 = self.down_sample2(x)  # (8,1024,6,6)
        x_down4 = self.down_sample4(x)  # (8,1024,3,3)

        conv_x2 = self.up_sample2(self.relu(self.conv1(x_down2)))  # (8,1024,12,12)
        conv_x4 = self.up_sample4(self.relu(self.conv2(x_down4)))  # (8,1024,12,12)

        resl = x + conv_x2 + conv_x4  # (8,1024,12,12)
        resl1 = self.bn2(resl)  # (8,1024,12,12)
        resl2 = self.conv_sum(resl1)
        resl3 = self.relu(resl2)  # (8,out_channel,12,12)

        return resl3


class UCIB2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UCIB2, self).__init__()
        self.down_sample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_sample4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.down_sample8 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_sum = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.up_sample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_sample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up_sample8 = nn.UpsamplingBilinear2d(scale_factor=8)

    def forward(self, x):
        x = self.bn1(x)

        x_down2 = self.down_sample2(x)
        x_down4 = self.down_sample4(x)
        x_down8 = self.down_sample8(x)

        conv_x2 = self.up_sample2(self.relu(self.conv1(x_down2)))
        conv_x4 = self.up_sample4(self.relu(self.conv2(x_down4)))
        conv_x8 = self.up_sample8(self.relu(self.conv3(x_down8)))

        resl = x + conv_x2 + conv_x4 + conv_x8
        resl1 = self.bn2(resl)
        resl2 = self.conv_sum(resl1)
        resl3 = self.relu(resl2)

        return resl3


# Feature Integrity Learning and Refinement
eps = 1e-12


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
                nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
                nn.AdaptiveAvgPool1d,
                nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


class EmRouting2d(nn.Module):
    def __init__(self, A, B, caps_size, kernel_size=3, stride=1, padding=1, iters=3, final_lambda=1e-2):
        super(EmRouting2d, self).__init__()
        self.A = A
        self.B = B
        self.psize = caps_size
        self.mat_dim = int(caps_size ** 0.5)

        self.k = kernel_size
        self.kk = kernel_size ** 2
        self.kkA = self.kk * A

        self.stride = stride
        self.pad = padding

        self.iters = iters

        self.W = nn.Parameter(torch.FloatTensor(self.kkA, B, self.mat_dim, self.mat_dim))
        nn.init.kaiming_uniform_(self.W.data)

        self.beta_u = nn.Parameter(torch.FloatTensor(1, 1, B, 1))
        self.beta_a = nn.Parameter(torch.FloatTensor(1, 1, B))
        nn.init.constant_(self.beta_u, 0)
        nn.init.constant_(self.beta_a, 0)

        self.final_lambda = final_lambda
        self.ln_2pi = math.log(2 * math.pi)
        self.initialize()

    def m_step(self, v, a_in, r):
        # v: [b, l, kkA, B, psize]
        # a_in: [b, l, kkA]
        # r: [b, l, kkA, B, 1]
        b, l, _, _, _ = v.shape

        a = a_in.view(b, l, -1, 1, 1)

        # r: [b, l, kkA, B, 1]
        r = r * a_in.view(b, l, -1, 1, 1)
        # r_sum: [b, l, 1, B, 1]
        r_sum = r.sum(dim=2, keepdim=True)
        # coeff: [b, l, kkA, B, 1]
        coeff = r / (r_sum + eps)

        # mu: [b, l, 1, B, psize]
        mu = torch.sum(coeff * v, dim=2, keepdim=True)
        # sigma_sq: [b, l, 1, B, psize]
        sigma_sq = torch.sum(coeff * (v - mu) ** 2, dim=2, keepdim=True) + eps

        # [b, l, B, 1]
        r_sum = r_sum.squeeze(2)
        # [b, l, B, psize]
        sigma_sq = sigma_sq.squeeze(2)
        # [1, 1, B, 1] + [b, l, B, psize] * [b, l, B, 1]
        cost_h = (self.beta_u + torch.log(sigma_sq.sqrt())) * r_sum
        # cost_h = (torch.log(sigma_sq.sqrt())) * r_sum

        # [b, l, B]
        a_out = torch.sigmoid(self.lambda_ * (self.beta_a - cost_h.sum(dim=3)))
        # a_out = torch.sigmoid(self.lambda_*(-cost_h.sum(dim=3)))

        return a_out, mu, sigma_sq

    def e_step(self, v, a_out, mu, sigma_sq):
        b, l, _ = a_out.shape
        # v: [b, l, kkA, B, psize]
        # a_out: [b, l, B]
        # mu: [b, l, 1, B, psize]
        # sigma_sq: [b, l, B, psize]

        # [b, l, 1, B, psize]
        sigma_sq = sigma_sq.unsqueeze(2)

        ln_p_j = -0.5 * torch.sum(torch.log(sigma_sq * self.ln_2pi), dim=-1) \
                 - torch.sum((v - mu) ** 2 / (2 * sigma_sq), dim=-1)

        # [b, l, kkA, B]
        ln_ap = ln_p_j + torch.log(a_out.view(b, l, 1, self.B))
        # [b, l, kkA, B]
        r = torch.softmax(ln_ap, dim=-1)
        # [b, l, kkA, B, 1]
        return r.unsqueeze(-1)

    def forward(self, a_in, pose):
        # pose: [batch_size, A, psize]
        # a: [batch_size, A]
        batch_size = a_in.shape[0]

        # a: [b, A, h, w]
        # pose: [b, A*psize, h, w]
        b, _, h, w = a_in.shape

        # [b, A*psize*kk, l]
        pose = F.unfold(pose, self.k, stride=self.stride, padding=self.pad)
        l = pose.shape[-1]
        # [b, A, psize, kk, l]
        pose = pose.view(b, self.A, self.psize, self.kk, l)
        # [b, l, kk, A, psize]
        pose = pose.permute(0, 4, 3, 1, 2).contiguous()
        # [b, l, kkA, psize]
        pose = pose.view(b, l, self.kkA, self.psize)
        # [b, l, kkA, 1, mat_dim, mat_dim]
        pose = pose.view(batch_size, l, self.kkA, self.mat_dim, self.mat_dim).unsqueeze(3)

        # [b, l, kkA, B, mat_dim, mat_dim]
        pose_out = torch.matmul(pose, self.W)

        # [b, l, kkA, B, psize]
        v = pose_out.view(batch_size, l, self.kkA, self.B, -1)

        # [b, kkA, l]
        a_in = F.unfold(a_in, self.k, stride=self.stride, padding=self.pad)
        # [b, A, kk, l]
        a_in = a_in.view(b, self.A, self.kk, l)
        # [b, l, kk, A]
        a_in = a_in.permute(0, 3, 2, 1).contiguous()
        # [b, l, kkA]
        a_in = a_in.view(b, l, self.kkA)

        r = a_in.new_ones(batch_size, l, self.kkA, self.B, 1)
        for i in range(self.iters):
            # this is from open review
            self.lambda_ = self.final_lambda * (1 - 0.95 ** (i + 1))
            a_out, pose_out, sigma_sq = self.m_step(v, a_in, r)
            if i < self.iters - 1:
                r = self.e_step(v, a_out, pose_out, sigma_sq)

        # [b, l, B*psize]
        pose_out = pose_out.squeeze(2).view(b, l, -1)
        # [b, B*psize, l]
        pose_out = pose_out.transpose(1, 2)
        # [b, B, l]
        a_out = a_out.transpose(1, 2).contiguous()

        oh = ow = math.floor(l ** (1 / 2))

        a_out = a_out.view(b, -1, oh, ow)
        pose_out = pose_out.view(b, -1, oh, ow)

        return a_out, pose_out

    def initialize(self):
        weight_init(self)


class FILR(nn.Module):
    def __init__(self, channels):
        super(FILR, self).__init__()
        self.conv_trans = nn.Conv2d(channels * 3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_trans = nn.BatchNorm2d(64)

        self.num_caps = 8
        planes = 16

        self.conv_m = nn.Conv2d(64, self.num_caps, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv_pose = nn.Conv2d(64, self.num_caps * 16, kernel_size=5, stride=1, padding=1, bias=False)

        self.bn_m = nn.BatchNorm2d(self.num_caps)
        self.bn_pose = nn.BatchNorm2d(self.num_caps * 16)

        self.emrouting = EmRouting2d(self.num_caps, self.num_caps, 16, kernel_size=3, stride=2, padding=1)
        self.bn_caps = nn.BatchNorm2d(self.num_caps * planes)

        self.conv_rec = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_rec = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.fuse1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.fuse2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.CA = ChannelAttention(192)
        self.conv_fuse = nn.Sequential(nn.Conv2d(192, 192, kernel_size=1),
                                       nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))

        self.out1_SA = SpatialAttention()
        self.out2_SA = SpatialAttention()

    def forward(self, input1, input2, input3):
        _, _, h1, w1 = input1.size()  # (24, 24)
        _, _, h2, w2 = input2.size()  # (48, 48)
        _, _, h3, w3 = input3.size()  # (96, 96)

        input1_up = F.interpolate(input1, size=(h2, w2), mode='bilinear', align_corners=True)  # (8,64,48,48)
        input3_down = F.interpolate(input3, size=(h2, w2), mode='bilinear', align_corners=True)  # (8,64,48,48)

        input_fuse = torch.cat((input1_up, input2, input3_down), dim=1)  # (8,64*3,48,48)
        input_fuse = self.conv_fuse(input_fuse + input_fuse * self.CA(input_fuse))

        # primary caps
        m, pose = self.conv_m(input_fuse), self.conv_pose(input_fuse)  # m:(b,8,46,46); pose:(b,8*16,46,46)
        m, pose = torch.sigmoid(self.bn_m(m)), self.bn_pose(pose)  # m:(b,8,46,46); pose:(b,8*16,46,46)

        m, pose = self.emrouting(m, pose)  # m:(b,8,23,23); pose:(b,128,23,23)
        pose = self.bn_caps(pose)  # pose:(b,8*16,23,23)

        pose = F.relu(self.bn_rec(self.conv_rec(pose)), inplace=True)  # pose:(b,64,23,23)

        pose1 = F.interpolate(pose, size=(h1, w1), mode='bilinear', align_corners=True)  # (8,64,24,24)
        pose2 = F.interpolate(pose, size=(h2, w2), mode='bilinear', align_corners=True)  # (8,64,48,48)
        pose3 = F.interpolate(pose, size=(h3, w3), mode='bilinear', align_corners=True)  # (8,64,96,96)

        out1 = torch.cat((input1, pose1), dim=1)  # (8,128,24,24)
        out2 = torch.cat((input2, pose2), dim=1)  # (8,128,48,48)
        out3 = torch.cat((input3, pose3), dim=1)  # (8,128,96,96)

        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)  # (8,64.24,24)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)  # (8,64,48,48)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)  # (8,64,96,96)

        out1 = F.interpolate(out1, size=(h2, w2), mode='bilinear', align_corners=True)  # (8,64,48,48)\\
        out1 = out1 + out1 * self.out1_SA(out1)

        out2 = self.fuse1(out2 * out1) + out2  # (8,64,48,48)
        out2 = F.interpolate(out2, size=(h3, w3), mode='bilinear', align_corners=True)  # (8,64,96,96)
        out2 = out2 + out2 * self.out1_SA(out2)

        out3 = self.fuse2(out3 * out2) + out3  # (8,64,96,96)

        # return out3
        return out1, out2, out3

