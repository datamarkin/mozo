# SPDX-License-Identifier: Apache-2.0
"""The numbers that define a CLIP variant, as frozen data.

Upstream derives these from the checkpoint at load time -- ``build_model`` reads shapes out of the
state dict and infers the geometry. That works, and it means the architecture is only ever
described by the weights. Written out here instead, for the reason every other family in this tree
writes its geometry down: a spec that is inferred cannot be checked, and a variant that loads with
the wrong geometry inferred is a silent wrong answer rather than a failed load.

The two are held in step by the strict load in :mod:`~mozo.vendors.clip_deploy.checkpoint`.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPECS", "VARIANTS", "Spec"]


@dataclass(frozen=True)
class Spec:
    """One published variant, in full.

    Attributes:
        variant: mozo's name for it.
        upstream: OpenAI's name, as ``clip.available_models()`` reports it.
        embed_dim: Width of the shared space both towers project into.
        resolution: Square side the image tower runs at.
        patch: Side of one image patch. The grid is ``(resolution // patch) ** 2``.
        vision_width: Transformer width of the image tower.
        vision_layers: Blocks in the image tower.
        text_width: Transformer width of the text tower.
        text_layers: Blocks in the text tower.
        text_heads: Attention heads in the text tower. The image tower derives its own from
            ``vision_width // 64``, which is upstream's rule and not a separate number.
    """

    variant: str
    upstream: str
    embed_dim: int
    resolution: int
    patch: int
    vision_width: int
    vision_layers: int
    text_width: int
    text_layers: int
    text_heads: int

    @property
    def vision_heads(self) -> int:
        """Heads in the image tower. Upstream fixes the head dimension at 64 and divides."""
        return self.vision_width // 64


#: The four Vision Transformer variants OpenAI publishes. The five ResNet ones use a different
#: image tower -- a modified ResNet with attention pooling -- and are not carried yet.
SPECS: dict[str, Spec] = {
    spec.variant: spec
    for spec in (
        Spec("base", "ViT-B/32", 512, 224, 32, 768, 12, 512, 12, 8),
        Spec("base-16", "ViT-B/16", 512, 224, 16, 768, 12, 512, 12, 8),
        Spec("large", "ViT-L/14", 768, 224, 14, 1024, 24, 768, 12, 12),
        Spec("large-336", "ViT-L/14@336px", 768, 336, 14, 1024, 24, 768, 12, 12),
    )
}

VARIANTS = list(SPECS)
