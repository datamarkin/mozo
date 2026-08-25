# SPDX-License-Identifier: Apache-2.0
"""ViTPose's geometry, as frozen dataclasses.

Upstream keeps these in ``VitPoseConfig`` and ``VitPoseBackboneConfig``, which read a JSON file
beside the weights and fill in defaults for whatever it omits -- and the published configs omit a
lot. ``vitpose-plus-base`` names three numbers and inherits fifteen. That is fine when the config
travels with the checkpoint and wrong here, because mozo publishes the checkpoint alone: the
geometry has to be knowable without it, so :mod:`mozo.registry` can answer "what variants exist"
with no download.

So every number is written out, including the ones upstream leaves to a default. The values were
read off the published checkpoints' tensor shapes, not transcribed from the configs, and
``tests/families/test_vitpose.py`` holds them against a strict load.

**Every published variant is ViTPose++**, which is the mixture-of-experts revision of the original.
The original's checkpoints exist and are not published here: ViTPose++ is better at every size, and
``plus-small`` is smaller than the smallest of them. Nothing in this package is specific to the MoE
beyond :class:`~.layers.MoeMLP`; a single-expert variant would need one more branch, and would come
with the variant that needed it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPECS", "Spec", "get_spec"]


@dataclass(frozen=True)
class Spec:
    """One published geometry.

    Args:
        hidden: Block width.
        layers: Number of transformer blocks.
        heads: Attention heads.
        part_features: Width of the slice each expert produces. The block's MLP writes
            ``hidden - part_features`` columns shared across datasets and ``part_features``
            columns from the selected expert, then concatenates them.
        experts: How many dataset experts the MoE carries. Six in every published checkpoint.
        mlp_ratio: MLP width as a multiple of ``hidden``.
        keypoints: Heatmap channels, one per joint. COCO's 17 in every published checkpoint.
        height: Input height. Not square, and not negotiable -- the position embedding is sized
            for this grid.
        width: Input width.
        patch: Patch side.
        patch_padding: Padding on the patch-embedding convolution. **2, not 0** -- one of the two
            things that separate this trunk from a plain ViT, and it changes the patch count.
        scale_factor: How much the head upsamples. Two stride-2 deconvolutions, so 4.
        layer_norm_eps: Upstream's ``1e-12``, which is not PyTorch's ``nn.LayerNorm`` default.
    """

    hidden: int
    layers: int
    heads: int
    part_features: int
    experts: int = 6
    mlp_ratio: int = 4
    keypoints: int = 17
    height: int = 256
    width: int = 192
    patch: int = 16
    patch_padding: int = 2
    scale_factor: int = 4
    layer_norm_eps: float = 1e-12

    @property
    def grid(self) -> tuple[int, int]:
        """Patch rows and columns.

        Read off the *unpadded* division, which is what upstream's reshape uses. The padded
        convolution happens to produce the same count for every published geometry, and the two
        are not the same statement: this one is the shape the features are folded back into.
        """
        return self.height // self.patch, self.width // self.patch

    @property
    def heatmap(self) -> tuple[int, int]:
        """Heatmap rows and columns."""
        rows, columns = self.grid
        return rows * self.scale_factor, columns * self.scale_factor


#: Every published geometry. ``part_features`` for ``large`` is upstream's default rather than a
#: number its config states; the others state their own.
SPECS: dict[str, Spec] = {
    "small": Spec(hidden=384, layers=12, heads=12, part_features=96),
    "base": Spec(hidden=768, layers=12, heads=12, part_features=192),
    "large": Spec(hidden=1024, layers=24, heads=16, part_features=256),
    "huge": Spec(hidden=1280, layers=32, heads=16, part_features=320),
}


def get_spec(variant: str) -> Spec:
    """The geometry for *variant*.

    Args:
        variant: One of :data:`SPECS`.

    Raises:
        ValueError: If the variant is not published, naming the ones that are.
    """
    try:
        return SPECS[variant]
    except KeyError:
        raise ValueError(
            f"Unknown ViTPose variant {variant!r}. Choose from: {list(SPECS)}"
        ) from None
