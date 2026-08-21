# SPDX-License-Identifier: Apache-2.0
"""The transformer block both of OWLv2's towers are built from.

OWLv2 is CLIP with detection heads bolted on, and CLIP's image and text towers are the same block
at different widths. Upstream expresses that by constructing one ``Owlv2EncoderLayer`` class from
either config; this module is that class. Sharing it *within* this package is not the duplication
mozo forbids -- that rule is about one vendor reaching into another, and both callers here are
this vendor.

Two details are load-bearing and neither is visible from the weights:

**The activation is ``quick_gelu``, not ``gelu``.** Both published configs say so, inheriting it
from CLIP. A checkpoint loads strictly under either, and the difference shows up as a small,
plausible shift in every logit -- 0.017 on a mask-free forward, which is the size of a real
detection score changing its mind.

**Attention is a single fused ``scaled_dot_product_attention``.** That is what upstream dispatches
to on any machine where PyTorch offers it, and computing the same algebra as an explicit
``q @ k.T * scale`` then softmax is bit-for-bit different -- the same class of trap that cost the
EdgeTAM extraction a 9.2e-05 divergence. The scaling is left to the kernel rather than applied to
``q`` first, for the same reason.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["Encoder", "quick_gelu"]


def quick_gelu(x: Tensor) -> Tensor:
    """CLIP's sigmoid approximation of GELU. ``ACT2FN["quick_gelu"]`` upstream."""
    return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    """Multi-head self-attention with separate q, k, v projections, as the checkpoint stores them.

    Args:
        width: Block width.
        heads: Attention heads. Must divide ``width``.
    """

    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError(f"width {width} is not divisible by {heads} heads")
        self.heads = heads
        self.head_dim = width // heads
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.q_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Attend over the sequence.

        Args:
            x: ``(B, L, width)``.
            mask: ``(B, 1, L, L)`` additive float mask, or None for unmasked attention. The
                vision tower never passes one; the text tower always does, because its mask is
                causal *and* has to hide padding.
        """
        batch, length, _ = x.shape
        shape = (batch, length, self.heads, self.head_dim)
        query = self.q_proj(x).view(shape).transpose(1, 2)
        key = self.k_proj(x).view(shape).transpose(1, 2)
        value = self.v_proj(x).view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        return self.out_proj(attended.transpose(1, 2).reshape(batch, length, -1))


class Block(nn.Module):
    """One pre-norm residual block: attention, then a widen-activate-narrow MLP.

    Args:
        width: Block width.
        heads: Attention heads.
        intermediate: MLP width.
        eps: Layer-norm epsilon. OWLv2 uses ``1e-5``, not PyTorch's default.
    """

    def __init__(self, width: int, heads: int, intermediate: int, eps: float):
        super().__init__()
        self.self_attn = Attention(width, heads)
        self.layer_norm1 = nn.LayerNorm(width, eps=eps)
        self.mlp = _FeedForward(width, intermediate)
        self.layer_norm2 = nn.LayerNorm(width, eps=eps)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class _FeedForward(nn.Module):
    """Widen, ``quick_gelu``, narrow. Named ``mlp`` to match the checkpoint's keys."""

    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(width, intermediate)
        self.fc2 = nn.Linear(intermediate, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(quick_gelu(self.fc1(x)))


class Encoder(nn.Module):
    """A stack of :class:`Block`. Named ``layers`` to match the checkpoint's keys.

    Args:
        layers: How many blocks.
        width: Block width.
        heads: Attention heads.
        intermediate: MLP width.
        eps: Layer-norm epsilon.
    """

    def __init__(self, layers: int, width: int, heads: int, intermediate: int, eps: float):
        super().__init__()
        self.layers = nn.ModuleList(
            Block(width, heads, intermediate, eps) for _ in range(layers)
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        for block in self.layers:
            x = block(x, mask)
        return x
