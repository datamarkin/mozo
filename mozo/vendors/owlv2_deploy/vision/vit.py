# SPDX-License-Identifier: Apache-2.0
"""OWLv2's image tower: CLIP's ViT, run for its patch grid rather than its pooled embedding.

This is the expensive half. A B/16 at 960 is 3,600 patches and an L/14 at 1008 is 5,184, which is
several times what a classifier ViT sees -- open-vocabulary detection needs one prediction per
patch, so the resolution cannot be traded away.

**The pooled output is not built here, and neither is the projection that consumes it.** CLIP ends
with a class token projected into a shared image-text space; the detector never reads it. The
weights for that projection are in the checkpoint (``visual_projection``, 1.5 MB) and upstream
computes it on every forward, along with the full contrastive similarity matrix, before discarding
both. See ``PROVENANCE.md``.

What the detector reads instead is the *whole* sequence after ``post_layernorm``, class token
included -- the class token is folded into every patch multiplicatively in
:mod:`~mozo.vendors.owlv2_deploy.network`, which is where that happens.

``interpolate_pos_encoding`` is not implemented. Upstream offers it to run at a resolution the
position embedding was not trained for; this package always runs at the published one, so there is
nothing to interpolate and no branch that can silently take the wrong path.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import VisionSpec
from ..layers import Encoder

__all__ = ["VisionTower"]


class Embeddings(nn.Module):
    """Patchify, prepend the class token, add the learned position embedding.

    Args:
        spec: Vision geometry.
    """

    def __init__(self, spec: VisionSpec):
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(spec.width))
        # No bias, which is CLIP's choice and the checkpoint's shape.
        self.patch_embedding = nn.Conv2d(
            3, spec.width, kernel_size=spec.patch_size, stride=spec.patch_size, bias=False
        )
        positions = spec.patches**2 + 1
        # An ``nn.Embedding`` rather than a plain parameter, because that is what the checkpoint
        # is keyed for: ``position_embedding.weight``.
        self.position_embedding = nn.Embedding(positions, spec.width)
        self.register_buffer(
            "position_ids", torch.arange(positions).expand((1, -1)), persistent=False
        )

    def forward(self, pixels: Tensor) -> Tensor:
        """``(B, 3, S, S)`` in, ``(B, 1 + patches^2, width)`` out."""
        patches = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(pixels.shape[0], 1, -1)
        return torch.cat([cls, patches], dim=1) + self.position_embedding(self.position_ids)


class VisionTower(nn.Module):
    """An image in, the full token sequence out.

    Args:
        spec: Vision geometry.
    """

    def __init__(self, spec: VisionSpec):
        super().__init__()
        self.embeddings = Embeddings(spec)
        self.pre_layernorm = nn.LayerNorm(spec.width, eps=spec.layer_norm_eps)
        self.encoder = Encoder(
            spec.layers, spec.width, spec.heads, spec.intermediate, spec.layer_norm_eps
        )
        self.post_layernorm = nn.LayerNorm(spec.width, eps=spec.layer_norm_eps)

    def forward(self, pixels: Tensor) -> Tensor:
        """``(B, 3, S, S)`` in, ``(B, 1 + patches^2, width)`` out, normalised.

        Upstream applies ``post_layernorm`` to the pooled class token inside the tower and again
        to the full sequence in the detector. Only the second is on this path, so it is applied
        once, here.
        """
        hidden = self.pre_layernorm(self.embeddings(pixels))
        return self.post_layernorm(self.encoder(hidden))
