# SPDX-License-Identifier: Apache-2.0
"""Per-variant model geometry, replacing upstream's Hydra configs.

Upstream builds a model by handing a YAML file to Hydra, which imports and instantiates each
``_target_`` class. That pulls in ``hydra-core`` and ``omegaconf`` to do what amounts to calling
four constructors with recorded numbers, so the numbers are written here directly and the
constructors are called in :mod:`.network`.

Every value below was read out of ``sam2/configs/sam2.1/sam2.1_hiera_*.yaml``. The four variants
differ only in the trunk and in the neck's input widths; everything else is identical across them,
which is why it lives in :data:`SHARED` rather than being repeated four times.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SPECS", "Spec", "SHARED"]


@dataclass(frozen=True)
class Spec:
    """The geometry of one published variant.

    Args:
        embed_dim: Width of the Hiera trunk's first stage.
        num_heads: Attention heads in the trunk's first stage.
        backbone_channel_list: Trunk output widths, coarsest first, as the neck receives them.
        stages: Blocks per trunk stage.
        global_att_blocks: Indices of the blocks that attend globally rather than within a window.
        window_pos_embed_bkg_spatial_size: Spatial size of the background window position embedding.
        window_spec: Window size per stage.
    """

    embed_dim: int
    num_heads: int
    backbone_channel_list: tuple[int, ...]
    stages: tuple[int, ...] = (2, 3, 16, 3)
    global_att_blocks: tuple[int, ...] = (12, 16, 20)
    window_pos_embed_bkg_spatial_size: tuple[int, int] = (14, 14)
    window_spec: tuple[int, ...] = (8, 4, 14, 7)


#: Settings every variant shares. Named rather than inlined so the two places that need the
#: resolution -- the preprocessing and the prompt scaling -- read the same number.
SHARED = {
    "image_size": 1024,
    "backbone_stride": 16,
    "hidden_dim": 256,
    "scalp": 1,
    "fpn_top_down_levels": (2, 3),
    "fpn_interp_model": "nearest",
    "num_feature_levels": 3,
}

#: Variant name -> geometry. ``base_plus`` takes the trunk defaults; the other three override.
SPECS: dict[str, Spec] = {
    "tiny": Spec(
        embed_dim=96,
        num_heads=1,
        backbone_channel_list=(768, 384, 192, 96),
        stages=(1, 2, 7, 2),
        global_att_blocks=(5, 7, 9),
        window_pos_embed_bkg_spatial_size=(7, 7),
    ),
    "small": Spec(
        embed_dim=96,
        num_heads=1,
        backbone_channel_list=(768, 384, 192, 96),
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
    ),
    "base_plus": Spec(
        embed_dim=112,
        num_heads=2,
        backbone_channel_list=(896, 448, 224, 112),
    ),
    "large": Spec(
        embed_dim=144,
        num_heads=2,
        backbone_channel_list=(1152, 576, 288, 144),
        stages=(2, 6, 36, 4),
        global_att_blocks=(23, 33, 43),
        window_pos_embed_bkg_spatial_size=(7, 7),
        window_spec=(8, 4, 16, 8),
    ),
}
