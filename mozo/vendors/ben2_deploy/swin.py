# ------------------------------------------------------------------------
# BEN2 -- Background Erase Network
# Copyright (c) 2025 Prama LLC. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
#
# The Swin Transformer in this file derives from Microsoft's implementation
# (https://github.com/microsoft/Swin-Transformer, MIT), by way of BEN2.py.
# ------------------------------------------------------------------------
"""Swin-B backbone, returning five feature maps.

Extracted from ``BEN2.py`` lines 38-616. Three things changed, all of them removals:

* **``timm`` is gone.** ``to_2tuple`` is inlined, ``trunc_normal_`` is initialisation that the
  strict load overwrites before a single forward runs, and ``DropPath`` is reimplemented in eight
  lines below. Upstream lists ``timm>=1.0.10`` as a requirement; nothing it provides survives to
  inference, and the gate proves the removal changes no number.
* **No ``torch.utils.checkpoint``.** ``use_checkpoint`` is False at the one call site, and
  gradient checkpointing is a training facility.
* **No ``_freeze_stages``.** ``frozen_stages`` is -1 and this package does not train.

The window mask is rebuilt on every ``BasicLayer.forward``, as upstream does. It depends only on
``(H, W, window_size)`` and could be cached; it is not, because caching it is a change whose
payoff is unmeasured and whose risk is a stale mask at a second resolution.
"""

from __future__ import annotations

__all__ = ["SwinTransformer"]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False,
              scale_by_keep: bool = True) -> torch.Tensor:
    """Stochastic depth, per sample.

    Reimplemented from ``timm.layers.DropPath`` rather than imported, so this package depends on
    torch alone. At eval -- which is the only mode a deployment package runs in -- it returns *x*
    unchanged, so the body below never executes here. It is kept faithful anyway: a module that
    silently becomes an identity when someone calls ``.train()`` is a worse thing to ship than
    eight lines.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Stochastic depth wrapper. Carries no parameters, so the strict load never sees it."""

    def __init__(self, drop_prob: float | None = None, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob or 0.0
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    """Two linear layers with GELU between them."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """``(B, H, W, C)`` -> ``(num_windows * B, window_size, window_size, C)``."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """``(num_windows * B, window_size, window_size, C)`` -> ``(B, H, W, C)``."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window-based multi-head self attention with a relative position bias.

    The query is scaled *before* the matmul (``q = q * self.scale``), which is upstream's order
    and not the same in float as scaling the product. Followed exactly.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Pair-wise relative position index for the tokens inside one window. A buffer, so it is
        # in the checkpoint and the strict load checks it rather than trusting this arithmetic.
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class SwinTransformerBlock(nn.Module):
    """One Swin block: windowed attention, then an MLP, each with a residual."""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path_rate=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

        # Set by BasicLayer before each call. Upstream carries the resolution on the module
        # rather than passing it, and the assert below is what catches it going stale.
        self.H: int | None = None
        self.W: int | None = None

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    """Halve the resolution and double the width, by interleaving four strided views."""

    def __init__(self, dim, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # The order of these four is load-bearing: it decides which quarter of the 4C axis each
        # weight column multiplies. Reordering them loads cleanly and predicts nothing.
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)

        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    """One Swin stage: a run of blocks with alternating shift, then an optional merge."""

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, downsample=None) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path_rate=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        # Attention mask for the shifted blocks. Nine regions, numbered, differenced, and turned
        # into a -100/0 additive mask -- so tokens that the roll made neighbours without being
        # neighbours cannot attend to each other.
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        slices = (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        cnt = 0
        for h in slices:
            for w in slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            return x, H, W, x_down, (H + 1) // 2, (W + 1) // 2
        return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to patch embedding: one strided convolution, then a LayerNorm."""

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None) -> None:
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class SwinTransformer(nn.Module):
    """Swin-B, emitting five feature maps.

    ``ape`` is absent rather than False: upstream leaves it at its default, so
    ``absolute_pos_embed`` is never built and the checkpoint has no key for it. Carrying the flag
    would mean carrying a branch that no published weight can reach.
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True, out_indices=(0, 1, 2, 3)) -> None:
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None))

        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        for i_layer in out_indices:
            self.add_module(f"norm{i_layer}", norm_layer(self.num_features[i_layer]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return ``(patch_embed, stage0, stage1, stage2, stage3)`` -- five maps, not four.

        The list is seeded with the patch-embedding output *before* the stages run, which is why
        the decoder's first two projections both take 128 channels. Dropping the seed would make
        the tuple the length a reader expects and shift every projection by one stage.
        """
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        outs = [x.contiguous()]
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            x_out, H, W, x, Wh, Ww = self.layers[i](x, Wh, Ww)
            if i in self.out_indices:
                x_out = getattr(self, f"norm{i}")(x_out)
                outs.append(x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous())

        return tuple(outs)
