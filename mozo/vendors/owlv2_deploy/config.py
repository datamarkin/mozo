# SPDX-License-Identifier: Apache-2.0
"""OWLv2's geometry, as frozen dataclasses.

Upstream keeps these in ``Owlv2Config``, which reads a JSON file beside the weights and fills in
defaults for whatever it omits. Both published geometries omit most of it -- the base config names
five numbers and inherits nineteen. That is fine when the config travels with the checkpoint and
wrong here, because mozo publishes the checkpoint alone: the geometry has to be knowable without
it, so that :mod:`mozo.registry` can answer "what variants exist" without a download.

So every number is written out, including the ones upstream leaves to a default. The values were
read off the published checkpoints' tensor shapes, not transcribed from the configs, and
``tests/families/test_owlv2.py`` holds them against a strict load.

The two geometries differ in more than depth. B/16 runs at 960 and L/14 at 1008, which is not a
detail: it is chosen so both divide into a whole number of patches, 60x60 and 72x72.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPECS", "Spec", "TextSpec", "VisionSpec"]


@dataclass(frozen=True)
class VisionSpec:
    """The CLIP ViT trunk that turns an image into a grid of patch features.

    Args:
        width: Block width.
        layers: Number of blocks.
        heads: Attention heads.
        intermediate: MLP width. A plain 4x.
        image_size: Square side the trunk runs at.
        patch_size: Side of one patch. Divides ``image_size`` exactly.
        layer_norm_eps: Upstream's ``1e-5``, which is not PyTorch's ``nn.LayerNorm`` default.
    """

    width: int
    layers: int
    heads: int
    intermediate: int
    image_size: int
    patch_size: int
    layer_norm_eps: float = 1e-5

    @property
    def patches(self) -> int:
        """Patches along one side. The feature grid is this squared."""
        return self.image_size // self.patch_size


@dataclass(frozen=True)
class TextSpec:
    """The CLIP text tower that turns a prompt into one embedding.

    Args:
        width: Block width.
        layers: Number of blocks.
        heads: Attention heads.
        intermediate: MLP width. A plain 4x.
        vocab_size: CLIP's byte-pair vocabulary.
        context_length: Prompt length in tokens. **16, not CLIP's usual 77** -- the checkpoint's
            position embedding is ``(16, width)``. Longer prompts are truncated.
        projection: What ``text_projection`` maps to. Equal to ``width`` in both published
            geometries, and it has to be: the class head projects patches to ``width`` and dots
            them against the projected prompt, so a mismatch would not multiply.
        layer_norm_eps: As above.
    """

    width: int
    layers: int
    heads: int
    intermediate: int
    projection: int
    vocab_size: int = 49408
    context_length: int = 16
    layer_norm_eps: float = 1e-5


@dataclass(frozen=True)
class Spec:
    """One published model: both towers, and the heads' widths follow from them."""

    vision: VisionSpec
    text: TextSpec


#: The four variants mozo publishes, keyed by the name the registry and manifest use.
#:
#: ``-ensemble`` averages the self-trained and fine-tuned checkpoints; it is the one nearly
#: everyone uses (1.34M downloads against 105k for plain ``base``) and the one whose numbers the
#: paper reports. The two ``-finetuned`` checkpoints Google also publishes are not here: under
#: 1,100 downloads between them, and nothing in this package would differ if they were.
SPECS: dict[str, Spec] = {
    "base": Spec(
        vision=VisionSpec(width=768, layers=12, heads=12, intermediate=3072,
                          image_size=960, patch_size=16),
        text=TextSpec(width=512, layers=12, heads=8, intermediate=2048, projection=512),
    ),
    "large": Spec(
        vision=VisionSpec(width=1024, layers=24, heads=16, intermediate=4096,
                          image_size=1008, patch_size=14),
        text=TextSpec(width=768, layers=12, heads=12, intermediate=3072, projection=768),
    ),
}
SPECS["base-ensemble"] = SPECS["base"]
SPECS["large-ensemble"] = SPECS["large"]
