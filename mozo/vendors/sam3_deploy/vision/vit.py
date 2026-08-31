# SPDX-License-Identifier: Apache-2.0
"""SAM 3's image trunk: a ViT-L/14 with 2-D rotary position embeddings and windowed attention.

Derived from ``transformers/models/sam3/modeling_sam3.py`` (Apache-2.0, Meta AI Authors and the
HuggingFace Team), reduced to inference and detached from the ``PreTrainedModel`` machinery.

Two deliberate departures from that source, both so the numbers match the shipped weights rather
than a re-derivation of them:

**The rotary embedding is loaded and applied in complex arithmetic**, exactly as upstream does.
``transformers`` rebuilds ``cos``/``sin`` tables from the config, discards the ``freqs_cis``
buffers the checkpoint carries, and rotates with real multiplies. Both changes cost accuracy that
compounds through 32 blocks, and both were measured rather than assumed:

- Rebuilding the tables lands one float32 ulp from the shipped values (5.96e-08 on every block),
  because ``torch.polar`` and ``cos()`` round differently.
- Rotating with ``q * cos + rotate_pairwise(q) * sin`` is algebraically identical to a complex
  multiply but not numerically -- 4.77e-07 per element, on a fifth of them -- which grows to
  1.35e-02 by the final block and moves mask boundaries.

So ``freqs_cis`` is kept complex and multiplied directly. Complex operators are a known gap in
ONNX; that is a problem for the export wrapper to solve against a correct reference, not a reason
to make the reference wrong.

**``qkv`` stays fused.** ``transformers`` splits it into three projections; the checkpoint stores
one. Keeping it fused is one GEMM per block instead of three, and removes a transformation
between disk and compute -- fewer places to be silently wrong.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import TrunkSpec
from ..layers import FeedForward

__all__ = ["Trunk"]


def apply_rotary(query: Tensor, key: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Apply the 2-D rotary position embedding to ``query`` and ``key``.

    Each adjacent pair of channels is read as one complex number and multiplied by the
    corresponding entry of ``freqs_cis``, which is a unit-magnitude complex exponential -- so the
    multiply is a rotation.

    Args:
        query: ``(B, heads, N, head_dim)``.
        key: Same shape as ``query``.
        freqs_cis: ``(N, head_dim // 2)`` complex. Broadcasts over batch and heads.

    Returns:
        The rotated pair, back in the input dtype. The rotation runs in float32, the precision
        the table was built at.
    """
    paired_query = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    paired_key = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    rotated_query = torch.view_as_real(paired_query * freqs_cis).flatten(3)
    rotated_key = torch.view_as_real(paired_key * freqs_cis).flatten(3)
    return rotated_query.type_as(query), rotated_key.type_as(key)


def window_partition(x: Tensor, window: int) -> tuple[Tensor, tuple[int, int]]:
    """Split ``(B, H, W, C)`` into ``(B * windows, window, window, C)``, padding if needed.

    Returns the windows and the padded ``(height, width)``, which :func:`window_unpartition`
    needs to undo the split.
    """
    batch, height, width, channels = x.shape
    pad_h = (window - height % window) % window
    pad_w = (window - width % window) % window
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = height + pad_h, width + pad_w
    x = x.view(batch, padded_h // window, window, padded_w // window, window, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window, window, channels)
    return windows, (padded_h, padded_w)


def window_unpartition(
    windows: Tensor, window: int, padded: tuple[int, int], original: tuple[int, int]
) -> Tensor:
    """Reverse :func:`window_partition`, dropping whatever padding it added."""
    padded_h, padded_w = padded
    height, width = original
    batch = windows.shape[0] // (padded_h * padded_w // window // window)
    x = windows.view(batch, padded_h // window, padded_w // window, window, window, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, padded_h, padded_w, -1)
    return x[:, :height, :width, :].contiguous()


class PatchEmbeddings(nn.Module):
    """The strided convolution that turns pixels into a grid of patch tokens."""

    def __init__(self, spec: TrunkSpec):
        super().__init__()
        # No bias: upstream builds this with ``bias_patch_embed=False``.
        self.projection = nn.Conv2d(
            3, spec.hidden, kernel_size=spec.patch, stride=spec.patch, bias=False
        )

    def forward(self, pixels: Tensor) -> Tensor:
        """``(B, 3, H, W)`` in, ``(B, H//patch * W//patch, hidden)`` out."""
        return self.projection(pixels).flatten(2).transpose(1, 2)


class Embeddings(nn.Module):
    """Patch embeddings plus a tiled absolute position embedding.

    The position embedding was trained at 336 pixels -- a 24x24 grid -- and inference runs at
    1008, a 72x72 grid. Upstream *tiles* it rather than interpolating: repeat the 24x24 block four
    times in each direction and crop to 72. Interpolating instead is a plausible-looking change
    that shifts every token slightly, so the tiling is spelled out here rather than left to a
    resize helper.
    """

    def __init__(self, spec: TrunkSpec):
        super().__init__()
        self.spec = spec
        self.patch = PatchEmbeddings(spec)
        # ``pretrain_grid ** 2`` positions, *without* the class token. The checkpoint ships 577
        # (a leading class position plus 576 patches); the loader drops index 0, because this
        # trunk is built with ``retain_cls_token=False`` and never forms a class token.
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, spec.pretrain_grid**2, spec.hidden)
        )

    def _tiled(self, height: int, width: int) -> Tensor:
        """Return the position embedding tiled to ``(1, height * width, hidden)``."""
        pretrain = self.spec.pretrain_grid
        if pretrain == height and pretrain == width:
            return self.position_embeddings
        hidden = self.position_embeddings.shape[-1]
        grid = self.position_embeddings.reshape(1, pretrain, pretrain, hidden).permute(0, 3, 1, 2)
        grid = grid.tile([1, 1, height // pretrain + 1, width // pretrain + 1])
        grid = grid[:, :, :height, :width]
        return grid.permute(0, 2, 3, 1).reshape(1, height * width, hidden)

    def forward(self, pixels: Tensor) -> Tensor:
        height, width = pixels.shape[-2] // self.spec.patch, pixels.shape[-1] // self.spec.patch
        return self.patch(pixels) + self._tiled(height, width)



class RoPEAttention(nn.Module):
    """Multi-head self-attention with a rotary position embedding on q and k.

    ``freqs_cis`` is a buffer rather than a computed value: it comes from the checkpoint. See the
    module docstring.
    """

    def __init__(self, spec: TrunkSpec, positions: int):
        super().__init__()
        self.heads = spec.heads
        self.head_dim = spec.head_dim
        self.qkv = nn.Linear(spec.hidden, spec.hidden * 3)
        self.o_proj = nn.Linear(spec.hidden, spec.hidden)
        # ``persistent`` so it is part of the state dict and must be supplied by the checkpoint.
        # A missing rotary table should fail loudly at load, not silently rotate by zero.
        self.register_buffer(
            "freqs_cis",
            torch.zeros(positions, self.head_dim // 2, dtype=torch.complex64),
            persistent=True,
        )

    def rotate(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the rotation, in the complex arithmetic the checkpoint's table is stored in.

        One line, and its own method, because it is the only part of this block a runtime
        without complex numbers has to write differently -- see ``tools/export/sam3.py``, which
        overrides exactly this and inherits the rest. A graph built by restating the block
        around a real-valued rotation would be a second copy of the arithmetic above, free to
        drift from it.
        """
        return apply_rotary(query, key, self.freqs_cis)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, H, W, C)`` in and out -- spatial, because windowing needs the grid."""
        batch, height, width, _ = x.shape
        length = height * width
        shape = (batch, length, self.heads, self.head_dim)
        query, key, value = self.qkv(x).view(batch, length, 3, -1).unbind(dim=2)
        query = query.view(*shape).transpose(1, 2)
        key = key.view(*shape).transpose(1, 2)
        value = value.view(*shape).transpose(1, 2)

        query, key = self.rotate(query, key)
        # No explicit ``scale``: upstream relies on the 1/sqrt(head_dim) default, and passing it
        # by hand is one more place for the two to drift apart.
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, height, width, -1)
        return self.o_proj(attended)


class Layer(nn.Module):
    """One pre-norm transformer block, attending either within a window or over the whole grid.

    Args:
        spec: The trunk geometry.
        window: Window side in patches, or 0 for a global block.
    """

    def __init__(self, spec: TrunkSpec, window: int):
        super().__init__()
        self.window = window
        side = window if window else spec.grid
        self.layer_norm1 = nn.LayerNorm(spec.hidden, eps=spec.layer_norm_eps)
        self.attention = RoPEAttention(spec, positions=side * side)
        self.layer_norm2 = nn.LayerNorm(spec.hidden, eps=spec.layer_norm_eps)
        self.mlp = FeedForward(spec.hidden, spec.intermediate)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.layer_norm1(x)
        if self.window:
            height, width = x.shape[1], x.shape[2]
            x, padded = window_partition(x, self.window)
            x = self.attention(x)
            x = window_unpartition(x, self.window, padded, (height, width))
        else:
            x = self.attention(x)
        x = residual + x
        return x + self.mlp(self.layer_norm2(x))


class Trunk(nn.Module):
    """The ViT itself: embed, normalise, then 32 blocks.

    Attributes:
        spec: The geometry this was built from.
    """

    def __init__(self, spec: TrunkSpec):
        super().__init__()
        self.spec = spec
        self.embeddings = Embeddings(spec)
        # ``ln_pre``: applied once before the stack, not after it. Upstream builds ``ln_post`` as
        # an identity for this model, so there is no trailing norm to mirror it.
        self.layer_norm = nn.LayerNorm(spec.hidden, eps=spec.layer_norm_eps)
        self.layers = nn.ModuleList(
            Layer(spec, window=0 if i in spec.global_blocks else spec.window)
            for i in range(spec.layers)
        )

    def forward(self, pixels: Tensor) -> Tensor:
        """``(B, 3, size, size)`` in, ``(B, hidden, grid, grid)`` out.

        Returned in NCHW rather than as a token sequence, because every consumer is convolutional.

        The permutation is deliberately *not* made contiguous. ``conv2d`` selects a different
        kernel for a channels-last view than for a contiguous NCHW tensor, and the two accumulate
        differently -- enough to move the neck's output by 2e-05 while the trunk itself stays
        bit-identical. Upstream hands the neck a permuted view, so this does too.
        """
        batch = pixels.shape[0]
        height = pixels.shape[-2] // self.spec.patch
        width = pixels.shape[-1] // self.spec.patch

        x = self.embeddings(pixels).view(batch, height, width, self.spec.hidden)
        x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)
