# SPDX-License-Identifier: Apache-2.0
"""The image tower: patches in, one pooled vector out.

Two things separate it from CLIP's Vision Transformer.

**There is no class token.** The position embedding covers the patch grid and nothing else, so
there is no learned slot travelling through the stack to be read off at the end. Which raises the
question of what to read off instead --

**-- and the answer is an attention-pooling head.** A single learned ``probe`` cross-attends the
final patch states, and its one output row is the image vector. Upstream calls it ``MAPHead``.
It is also the image tower's only projection: there is no ``visual_projection`` matrix, so the head
emits the shared space directly.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import LAYER_NORM_EPS, Spec
from ..layers import MLP, Encoder

__all__ = ["VisionTower"]


class AttentionPoolingHead(nn.Module):
    """Upstream's ``SiglipMultiheadAttentionPoolingHead``, reproduced including its two oddities.

    **The residual is taken before the layernorm, not after.** The block reads
    ``h = attn(...); r = h; h = ln(h); h = r + mlp(h)`` -- so the normalisation is inside the
    residual branch rather than in front of it, which is the opposite of every other block in this
    package. Read as pre-norm it runs fine and returns different numbers.

    **The attention is ``nn.MultiheadAttention``**, with a single fused ``in_proj_weight``, while
    every encoder block uses three separate projections. That is how the checkpoint stores it, and
    it is why this is torch's module rather than :class:`~mozo.vendors.siglip2_deploy.layers.Attention`.

    Keeping torch's module is not only about weight layout. ``need_weights`` is left at its default
    ``True``, which sends torch down its unfused branch -- where the query is scaled *before* the
    matmul rather than the product after it. That is CLIP's operator-precedence trap in a second
    costume: the same arithmetic, a different rounding, and 1.9e-06 of difference in CLIP's case.
    An unfused reimplementation would have to reproduce that ordering deliberately. Using the
    module upstream uses gets it for free and cannot drift.
    """

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        width = spec.vision_width
        self.probe = nn.Parameter(torch.zeros(1, 1, width))
        self.attention = nn.MultiheadAttention(width, spec.vision_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(width, eps=LAYER_NORM_EPS)
        self.mlp = MLP(width, spec.vision_mlp)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        probe = self.probe.repeat(hidden.shape[0], 1, 1)
        hidden = self.attention(probe, hidden, hidden)[0]
        residual = hidden
        hidden = residual + self.mlp(self.layernorm(hidden))
        return hidden[:, 0]


class Embeddings(nn.Module):
    """Patch projection plus a learned position per patch."""

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, spec.vision_width, spec.patch, stride=spec.patch, padding="valid")
        self.position_embedding = nn.Embedding(spec.patches, spec.vision_width)
        self.register_buffer(
            "position_ids", torch.arange(spec.patches).expand((1, -1)), persistent=False)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        return patches + self.position_embedding(self.position_ids)


class VisionTower(nn.Module):
    """The whole image tower. Takes preprocessed pixels, returns ``(N, projection)``."""

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.embeddings = Embeddings(spec)
        self.encoder = Encoder(
            spec.vision_width, spec.vision_layers, spec.vision_heads, spec.vision_mlp)
        self.post_layernorm = nn.LayerNorm(spec.vision_width, eps=LAYER_NORM_EPS)
        self.head = AttentionPoolingHead(spec)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.embeddings(pixels))
        return self.head(self.post_layernorm(hidden))
