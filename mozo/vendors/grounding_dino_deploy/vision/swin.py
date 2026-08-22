# SPDX-License-Identifier: Apache-2.0
"""The Swin Transformer image backbone, as Grounding DINO builds it.

Windowed self-attention over a four-stage pyramid, each stage halving the resolution and doubling
the width. Extracted from upstream's ``backbone/swin_transformer.py``, which is itself Microsoft's
Swin adapted for detection.

What is not carried: absolute position embeddings (``ape`` is False in both published configs, so
the branch never runs), gradient checkpointing (training only), stage freezing (training only),
dilation (never enabled), and ``forward_raw`` (a debugging entry point).

Only stages 1, 2 and 3 are returned. Stage 0 is computed -- the pyramid is sequential, so it must
be -- but its output is not normalised or emitted, which is why the checkpoint carries ``norm1``,
``norm2`` and ``norm3`` and no ``norm0``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SwinTransformer"]


def window_partition(x: Tensor, window: int) -> Tensor:
    """Cut ``(B, H, W, C)`` into ``(B * windows, window, window, C)``."""
    batch, height, width, channels = x.shape
    x = x.view(batch, height // window, window, width // window, window, channels)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, channels)


def window_reverse(windows: Tensor, window: int, height: int, width: int) -> Tensor:
    """The inverse of :func:`window_partition`."""
    batch = int(windows.shape[0] / (height * width / window / window))
    x = windows.view(batch, height // window, width // window, window, window, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, height, width, -1)


class _Mlp(nn.Module):
    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, features)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _WindowAttention(nn.Module):
    """Self-attention inside one window, with a learned relative position bias."""

    def __init__(self, dim: int, window: tuple[int, int], num_heads: int) -> None:
        super().__init__()
        self.window_size = window
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window[0] - 1) * (2 * window[1] - 1), num_heads)
        )

        coords = torch.stack(
            torch.meshgrid(torch.arange(window[0]), torch.arange(window[1]), indexing="ij")
        )
        flat = torch.flatten(coords, 1)
        relative = (flat[:, :, None] - flat[:, None, :]).permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window[0] - 1
        relative[:, :, 1] += window[1] - 1
        relative[:, :, 0] *= 2 * window[1] - 1
        self.register_buffer("relative_position_index", relative.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        windows, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(windows, tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn = (query * self.scale) @ key.transpose(-2, -1)

        area = self.window_size[0] * self.window_size[1]
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(area, area, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            count = mask.shape[0]
            attn = attn.view(windows // count, count, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)
        attn = attn.softmax(dim=-1)

        out = (attn @ value).transpose(1, 2).reshape(windows, tokens, channels)
        return self.proj(out)


class _Block(nn.Module):
    """One Swin block: windowed attention then an MLP, both residual and pre-normed."""

    def __init__(self, dim: int, num_heads: int, window: int, shift: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.window_size = window
        self.shift_size = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, (window, window), num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, mask_matrix: Tensor, height: int, width: int) -> Tensor:
        # The resolution arrives as arguments. Upstream smuggles it in by assigning ``blk.H`` and
        # ``blk.W`` before every call, which makes a shared module carry per-call state; two
        # ints down the call chain say the same thing and leave the module stateless.
        batch, _, channels = x.shape

        shortcut = x
        x = self.norm1(x).view(batch, height, width, channels)

        # Pad up to a whole number of windows. Only the right and bottom edges are padded, which
        # is what keeps the unpadded region aligned with the window grid's origin.
        pad_r = (self.window_size - width % self.window_size) % self.window_size
        pad_b = (self.window_size - height % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, padded_h, padded_w, _ = x.shape

        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted = x
            attn_mask = None

        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels)
        attended = self.attn(windows, mask=attn_mask)

        attended = attended.view(-1, self.window_size, self.window_size, channels)
        shifted = window_reverse(attended, self.window_size, padded_h, padded_w)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted

        if pad_r > 0 or pad_b > 0:
            x = x[:, :height, :width, :].contiguous()

        x = x.view(batch, height * width, channels)
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class _PatchMerging(nn.Module):
    """Halve the resolution by folding each 2x2 neighbourhood into the channel axis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        batch, _, channels = x.shape
        x = x.view(batch, height, width, channels)

        if height % 2 or width % 2:
            x = F.pad(x, (0, 0, 0, width % 2, 0, height % 2))

        quadrants = torch.cat(
            [x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]],
            -1,
        )
        return self.reduction(self.norm(quadrants.view(batch, -1, 4 * channels)))


class _Stage(nn.Module):
    """A run of blocks at one resolution, optionally followed by a merge."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window: int,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.window_size = window
        self.shift_size = window // 2
        self.blocks = nn.ModuleList(
            _Block(dim, num_heads, window, 0 if i % 2 == 0 else window // 2)
            for i in range(depth)
        )
        self.downsample = _PatchMerging(dim) if downsample else None

    def _shift_mask(self, height: int, width: int, device: torch.device) -> Tensor:
        """The mask that stops a shifted window attending across the wrap-around seam."""
        padded_h = -(-height // self.window_size) * self.window_size
        padded_w = -(-width // self.window_size) * self.window_size
        regions = torch.zeros((1, padded_h, padded_w, 1), device=device)
        slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        count = 0
        for rows in slices:
            for cols in slices:
                regions[:, rows, cols, :] = count
                count += 1

        windows = window_partition(regions, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, float(-100.0)).masked_fill(mask == 0, float(0.0))

    def forward(
        self, x: Tensor, height: int, width: int
    ) -> tuple[Tensor, int, int, Tensor, int, int]:
        attn_mask = self._shift_mask(height, width, x.device)
        for block in self.blocks:
            x = block(x, attn_mask, height, width)

        if self.downsample is not None:
            merged = self.downsample(x, height, width)
            return x, height, width, merged, (height + 1) // 2, (width + 1) // 2
        return x, height, width, x, height, width


class _PatchEmbed(nn.Module):
    """A strided convolution that turns pixels into 4x4 patch tokens."""

    def __init__(self, patch: int = 4, in_channels: int = 3, embed_dim: int = 96) -> None:
        super().__init__()
        self.patch_size = patch
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        if width % self.patch_size:
            x = F.pad(x, (0, self.patch_size - width % self.patch_size))
        if height % self.patch_size:
            x = F.pad(x, (0, 0, 0, self.patch_size - height % self.patch_size))

        x = self.proj(x)
        rows, cols = x.shape[2], x.shape[3]
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).view(-1, self.embed_dim, rows, cols)


class SwinTransformer(nn.Module):
    """The image backbone: pixels in, a feature pyramid out.

    Args:
        embed_dim: Width after patch embedding. 96 for Swin-T, 128 for Swin-B.
        depths: Blocks per stage.
        num_heads: Attention heads per stage.
        window_size: Attention window side. 7 for the 224-pretrained backbones, 12 for the
            384-pretrained Swin-B the `base` checkpoint is built on.
        out_indices: Which stages to emit.

    Examples:
        >>> backbone = SwinTransformer()                        # doctest: +SKIP
        >>> [f.shape for f in backbone(image)]                  # doctest: +SKIP
    """

    def __init__(
        self,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 7,
        out_indices: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()
        self.num_layers = len(depths)
        self.out_indices = out_indices
        self.num_features = [embed_dim * 2**i for i in range(self.num_layers)]

        self.patch_embed = _PatchEmbed(embed_dim=embed_dim)
        self.layers = nn.ModuleList(
            _Stage(
                dim=self.num_features[i],
                depth=depths[i],
                num_heads=num_heads[i],
                window=window_size,
                downsample=i < self.num_layers - 1,
            )
            for i in range(self.num_layers)
        )
        for index in out_indices:
            self.add_module(f"norm{index}", nn.LayerNorm(self.num_features[index]))

    def forward(self, x: Tensor) -> list[Tensor]:
        """Return one ``(batch, channels, height, width)`` map per requested stage."""
        x = self.patch_embed(x)
        rows, cols = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        outputs = []
        for index in range(self.num_layers):
            out, height, width, x, rows, cols = self.layers[index](x, rows, cols)
            if index in self.out_indices:
                normalised = getattr(self, f"norm{index}")(out)
                outputs.append(
                    normalised.view(-1, height, width, self.num_features[index])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        return outputs
