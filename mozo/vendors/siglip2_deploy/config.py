# SPDX-License-Identifier: Apache-2.0
"""The numbers that define a SigLIP 2 variant, as frozen data.

Upstream derives these from the published ``config.json``, which is itself incomplete -- most of
the fifteen omit most fields and inherit them from ``SiglipTextConfig``/``SiglipVisionConfig``
defaults. That works, and it means a variant's architecture is only ever described by two files
neither of which states it in full. Written out here instead, for the reason every family in this
tree writes its geometry down: a spec that is inferred cannot be checked, and a variant that loads
with the wrong geometry inferred is a silent wrong answer rather than a failed load.

The two are held in step by the strict load in :mod:`~mozo.vendors.siglip2_deploy.checkpoint`.

**Nothing here is derivable.** CLIP fixes its head dimension at 64 and divides; SigLIP 2 does not.
``so400m`` is 1152 wide over 16 heads, so its head dimension is 72, and ``giant-opt``'s is 96. Nor
is the MLP four times the width: ``so400m`` is 1152 -> 4304. Seven of the fifteen variants break
one rule or the other, which is why every number is written rather than computed.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CONTEXT", "LAYER_NORM_EPS", "SPECS", "VARIANTS", "VOCAB", "Spec"]

#: Pieces in the Gemma vocabulary. Every one of the fifteen states it explicitly and every one
#: states the same number -- checked against all fifteen published configs, not assumed from one.
VOCAB = 256000

#: Tokens every prompt is padded to, and the height of the text tower's position embedding.
#:
#: A constant rather than a field because all fifteen published variants agree on it -- checked,
#: not assumed. It cannot be read from the checkpoint's ``tokenizer_config.json`` either: SigLIP 2
#: inherits Gemma's ``model_max_length`` of 1000000000000000019884624838656.
CONTEXT = 64

#: LayerNorm epsilon. **Not** torch's 1e-5 default, and overridden by none of the fifteen configs.
#: A plain ``nn.LayerNorm(width)`` runs, produces plausible numbers, and is wrong.
LAYER_NORM_EPS = 1e-6


@dataclass(frozen=True)
class Spec:
    """One published variant, in full.

    Attributes:
        variant: mozo's name for it.
        upstream: Google's own name for the variant. Where it is *published* is a fact about
            publishing rather than about inference, so it lives in ``tools/`` and not here.
        resolution: Square side the image tower runs at.
        patch: Side of one image patch. The grid is ``(resolution // patch) ** 2``.
        vision_width: Transformer width of the image tower.
        vision_layers: Blocks in the image tower.
        vision_heads: Attention heads in the image tower.
        vision_mlp: Hidden width of the image tower's MLP.
        text_width: Transformer width of the text tower.
        text_layers: Blocks in the text tower.
        text_heads: Attention heads in the text tower.
        text_mlp: Hidden width of the text tower's MLP.
        projection: Width of the shared space. The text tower's head projects into it; the image
            tower's attention-pooling head already emits it. Equal to ``vision_width`` in every
            published variant -- including ``giant-opt``, where that means the *text* head projects
            up from 1152 to 1536.
    """

    variant: str
    upstream: str
    resolution: int
    patch: int
    vision_width: int
    vision_layers: int
    vision_heads: int
    vision_mlp: int
    text_width: int
    text_layers: int
    text_heads: int
    text_mlp: int
    projection: int

    @property
    def grid(self) -> int:
        """Patches along one side.

        **Floor division, and for two variants it truncates.** ``so400m-384`` is 384 pixels over a
        14-pixel patch, which is 27 patches and a remainder of six: the convolution strides off the
        edge and those six pixels never enter the model. Upstream computes the same number the same
        way -- ``(image_size // patch_size)`` in ``SiglipVisionEmbeddings`` and a ``stride=patch``
        convolution, which agree at 27 -- so this is faithful rather than a bug being reproduced.
        It is written down because ``resolution % patch == 0`` looks like an invariant and is not.
        """
        return self.resolution // self.patch

    @property
    def patches(self) -> int:
        """Positions in the image tower's embedding. No class token, so this is the whole grid."""
        return self.grid**2


#: The fifteen fixed-resolution variants Google publishes. The two ``-naflex`` ones run at variable
#: resolution through a different image tower and are not carried; see ``PROVENANCE.md``.
SPECS: dict[str, Spec] = {
    spec.variant: spec
    for spec in (
        Spec("base-224", "base-patch16-224", 224, 16, 768, 12, 12, 3072, 768, 12, 12, 3072, 768),
        Spec("base-256", "base-patch16-256", 256, 16, 768, 12, 12, 3072, 768, 12, 12, 3072, 768),
        Spec("base-384", "base-patch16-384", 384, 16, 768, 12, 12, 3072, 768, 12, 12, 3072, 768),
        Spec("base-512", "base-patch16-512", 512, 16, 768, 12, 12, 3072, 768, 12, 12, 3072, 768),
        Spec("base32-256", "base-patch32-256", 256, 32, 768, 12, 12, 3072, 768, 12, 12, 3072, 768),
        Spec("large-256", "large-patch16-256", 256, 16, 1024, 24, 16, 4096, 1024, 24, 16, 4096, 1024),
        Spec("large-384", "large-patch16-384", 384, 16, 1024, 24, 16, 4096, 1024, 24, 16, 4096, 1024),
        Spec("large-512", "large-patch16-512", 512, 16, 1024, 24, 16, 4096, 1024, 24, 16, 4096, 1024),
        Spec("so400m-224", "so400m-patch14-224", 224, 14, 1152, 27, 16, 4304, 1152, 27, 16, 4304, 1152),
        Spec("so400m-384", "so400m-patch14-384", 384, 14, 1152, 27, 16, 4304, 1152, 27, 16, 4304, 1152),
        Spec("so400m16-256", "so400m-patch16-256", 256, 16, 1152, 27, 16, 4304, 1152, 27, 16, 4304, 1152),
        Spec("so400m16-384", "so400m-patch16-384", 384, 16, 1152, 27, 16, 4304, 1152, 27, 16, 4304, 1152),
        Spec("so400m16-512", "so400m-patch16-512", 512, 16, 1152, 27, 16, 4304, 1152, 27, 16, 4304, 1152),
        Spec("giant-256", "giant-opt-patch16-256", 256, 16, 1536, 40, 16, 6144, 1152, 27, 16, 4304, 1536),
        Spec("giant-384", "giant-opt-patch16-384", 384, 16, 1536, 40, 16, 6144, 1152, 27, 16, 4304, 1536),
    )
}

VARIANTS = list(SPECS)
