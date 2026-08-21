# SPDX-License-Identifier: Apache-2.0
"""The published model's geometry, replacing upstream's Hydra config.

Upstream builds EdgeTAM by handing ``sam2/configs/edgetam.yaml`` to Hydra, which imports and
instantiates every ``_target_`` in it. That pulls in ``hydra-core`` and ``omegaconf`` to do what
amounts to calling four constructors with recorded numbers, so the numbers are written here and
the constructors are called in :mod:`.network`.

Meta publishes one EdgeTAM checkpoint, so there is one geometry and no variant table. Every value
below was read out of that YAML, except the three at the bottom, which come from ``build_sam2``
rather than from the config -- see :data:`STABILITY`.
"""

from __future__ import annotations

__all__ = ["SETTINGS", "STABILITY"]

#: The image path's geometry, from ``sam2/configs/edgetam.yaml``.
#:
#: ``backbone_channel_list`` is the trunk's output widths coarsest-first, which is the order the
#: neck builds its lateral convolutions in; :class:`~.backbones.repvit.RepViT` exposes the same
#: list and :class:`~.backbones.image_encoder.ImageEncoder` asserts the two agree.
#:
#: ``hidden_dim`` and ``backbone_stride`` are not in the YAML: upstream reads the first off the
#: neck it just built and defaults the second on ``SAM2Base``. They are named here because two
#: places need each of them -- the prompt encoder's grid and the preprocessing size -- and a
#: second lookup is how two numbers that must agree stop agreeing.
SETTINGS = {
    "image_size": 1024,
    "backbone_stride": 16,
    "hidden_dim": 256,
    "scalp": 1,
    "backbone_channel_list": (384, 192, 96, 48),
    "fpn_top_down_levels": (2, 3),
    "fpn_interp_model": "nearest",
    "num_feature_levels": 3,
}

#: The mask decoder's stability fallback, which is on and carries no weights.
#:
#: These are the one group of settings the checkpoint cannot vouch for. Upstream does not put them
#: in the config at all -- ``build_sam2`` appends them as Hydra overrides when
#: ``apply_postprocessing`` is true, which is its default, so every published way of loading
#: EdgeTAM has them on and reading only the YAML would leave them off. They change nothing unless
#: a caller asks for a single mask, so a suite that only ever asked for three would not notice.
STABILITY = {
    "dynamic_multimask_via_stability": True,
    "dynamic_multimask_stability_delta": 0.05,
    "dynamic_multimask_stability_thresh": 0.98,
}
