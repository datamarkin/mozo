# SPDX-License-Identifier: Apache-2.0
"""The transformer block both CLIP towers are built from.

One block, used twice: the image tower stacks it without a mask, the text tower stacks it with a
causal one. Everything that differs between the towers is in how they embed their input and how
they pool the result, not in here.

Two details in this file are the ones a from-scratch rewrite gets wrong, and neither raises.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["LayerNorm", "QuickGELU", "Transformer"]


class LayerNorm(nn.LayerNorm):
    """``nn.LayerNorm`` that always normalises in fp32 and casts back.

    Upstream subclasses it for this and CLIP's checkpoints are mixed precision, so under fp16 a
    plain ``nn.LayerNorm`` accumulates in half and diverges. mozo publishes fp32 weights, where the
    cast is a no-op -- it is kept because the class is what upstream runs, and because an fp16
    artifact later would need it and nobody would remember why.
    """

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.type(torch.float32)).type(x.dtype)


class QuickGELU(nn.Module):
    """``x * sigmoid(1.702 * x)``.

    **Not** ``nn.GELU``. It is a different function -- a sigmoid approximation predating the erf
    one -- and substituting the standard activation is the commonest CLIP reimplementation error.
    It does not raise, it does not warn, and it moves every embedding.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)


class _ResidualAttentionBlock(nn.Module):
    """Pre-norm attention, then pre-norm MLP, both residual."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential()
        # Named to match the checkpoint: mlp.c_fc, mlp.gelu, mlp.c_proj.
        self.mlp.add_module("c_fc", nn.Linear(width, width * 4))
        self.mlp.add_module("gelu", QuickGELU())
        self.mlp.add_module("c_proj", nn.Linear(width * 4, width))
        self.ln_2 = LayerNorm(width)

    def forward(self, x: Tensor, attn_mask: Tensor | None) -> Tensor:
        # ``need_weights=False`` is upstream's, and it is load-bearing rather than an optimisation:
        # it selects torch's fused attention path, which is the same arithmetic in a different
        # order from the unfused one. Grounding DINO's gate measured that difference at 1.5e-06,
        # small enough to look like noise and large enough to move the answer downstream.
        normed = self.ln_1(x)
        attended = self.attn(normed, normed, normed, need_weights=False, attn_mask=attn_mask)[0]
        x = x + attended
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    """A stack of residual attention blocks, sequence-first.

    Args:
        width: Model width.
        layers: How many blocks.
        heads: Attention heads per block.

    Examples:
        >>> Transformer(512, 12, 8)(x, None)      # doctest: +SKIP
    """

    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList(
            _ResidualAttentionBlock(width, heads) for _ in range(layers)
        )

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Run every block over ``(sequence, batch, width)``."""
        for block in self.resblocks:
            x = block(x, attn_mask)
        return x
