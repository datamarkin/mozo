# SPDX-License-Identifier: Apache-2.0
"""The numbers that define a Grounding DINO variant, as frozen data.

Upstream keeps these in ``groundingdino/config/*.py`` -- Python modules read through a bespoke
config loader (``SLConfig``) that also carries training-only settings. Written out here as frozen
dataclasses, because a config that can execute is a config that can differ between two runs, and
because the training settings are not carried at all.

The two published variants differ in exactly one field: which Swin backbone they use. Everything
else -- 900 queries, 6 encoder and 6 decoder layers, 4 feature levels, a 256-token text budget --
is shared, and upstream's own two config files differ on that single line.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPECS", "VARIANTS", "Spec", "SwinSpec"]


@dataclass(frozen=True)
class SwinSpec:
    """Geometry of the Swin Transformer image backbone."""

    embed_dim: int
    depths: tuple[int, ...]
    num_heads: tuple[int, ...]
    window_size: int

    @property
    def num_features(self) -> tuple[int, ...]:
        """Channel width of each stage's output."""
        return tuple(self.embed_dim * 2**i for i in range(len(self.depths)))


#: Upstream's ``build_swin_transformer`` table, restricted to the two backbones the published
#: Grounding DINO checkpoints actually use. ``swin_B_384_22k`` is the 384-pretrained Swin-B, which
#: is why its window is 12 rather than 7 -- the Swin-B checkpoint is `cogcoor`, built on it.
_SWIN = {
    "swin_T_224_1k": SwinSpec(96, (2, 2, 6, 2), (3, 6, 12, 24), 7),
    "swin_B_384_22k": SwinSpec(128, (2, 2, 18, 2), (4, 8, 16, 32), 12),
}


@dataclass(frozen=True)
class Spec:
    """One published variant, in full.

    Attributes:
        backbone: Which Swin geometry to build.
        hidden_dim: Width of the transformer, and of the projected text features.
        num_queries: Object queries the decoder carries. Every one produces a row of logits.
        max_text_len: Hard cap on prompt tokens. Upstream truncates past it; mozo refuses.
        return_levels: Which Swin stages feed the neck. The fourth feature level is made by a
            stride-2 convolution over the last of these, not by the backbone.
    """

    variant: str
    backbone_name: str
    hidden_dim: int = 256
    nheads: int = 8
    num_queries: int = 900
    enc_layers: int = 6
    dec_layers: int = 6
    dim_feedforward: int = 2048
    num_feature_levels: int = 4
    enc_n_points: int = 4
    dec_n_points: int = 4
    max_text_len: int = 256
    text_hidden_dim: int = 768
    #: Not 10000. Upstream sets both to 20, and a sine position encoding copied from any other
    #: DETR brings 10000 with it -- which runs, and moves every box.
    pe_temperature_h: int = 20
    pe_temperature_w: int = 20
    return_levels: tuple[int, ...] = (1, 2, 3)
    #: Shortest side, and the cap on the longest. Aspect ratio is preserved and nothing is
    #: padded, so the tensor a photograph becomes depends on the photograph.
    short_side: int = 800
    max_side: int = 1333
    #: Upstream's demo default, and the only threshold mozo publishes as its own. Upstream also
    #: carries a ``text_threshold``; it selects which tokens get decoded into a phrase, and this
    #: package reports the caller's prompt instead of decoding one. Not carried, so that nobody
    #: wires up a number whose only use would re-introduce the behaviour we diverged from.
    box_threshold: float = 0.35

    @property
    def swin(self) -> SwinSpec:
        return _SWIN[self.backbone_name]

    @property
    def backbone_channels(self) -> tuple[int, ...]:
        """Channel width of each level handed to the neck, backbone stages only."""
        features = self.swin.num_features
        return tuple(features[i] for i in self.return_levels)


#: Published variants, most downloaded first. `tiny` is 82% of upstream's own release downloads;
#: `base` is 8.3 box AP better on COCO zero-shot, which is why both are carried.
SPECS: dict[str, Spec] = {
    "tiny": Spec(variant="tiny", backbone_name="swin_T_224_1k"),
    "base": Spec(variant="base", backbone_name="swin_B_384_22k"),
}

VARIANTS = list(SPECS)
