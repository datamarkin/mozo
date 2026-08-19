# ------------------------------------------------------------------------
# Depth Anything V2
# Copyright (c) 2024 TikTok / The University of Hong Kong. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Architecture specifications for the released Depth Anything V2 variants.

Upstream expresses these as a bare ``model_configs`` dict repeated verbatim in ``run.py``,
``run_video.py``, ``app.py`` and their four counterparts under ``metric_depth/`` -- seven copies
of the same four rows. They are one frozen dataclass here.

Nine checkpoints ship: three relative-depth models, and the same three sizes fine-tuned twice for
metric depth, on Hypersim (indoor) and Virtual KITTI 2 (outdoor). ``max_depth`` is the entire
architectural difference between the two groups, which is why upstream's duplicated
``metric_depth/depth_anything_v2/`` tree collapses into this one.

``vitg`` is absent: upstream lists it as "coming soon" and has published no checkpoint.
"""

from __future__ import annotations

__all__ = ["MODEL_SPECS", "ModelSpec", "get_spec"]

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build one Depth Anything V2 variant and interpret its output.

    Attributes:
        encoder: DINOv2 backbone size -- ``vits``, ``vitb`` or ``vitl``.
        features: Width of the DPT decoder's fusion path.
        out_channels: Per-stage projection widths feeding the four fusion blocks.
        max_depth: Metre value the sigmoid head is scaled by, or ``None`` for a relative-depth
            model whose head ends in a ReLU instead.
        input_size: Side length the shorter axis is resized to, upstream's ``--input-size``
            default. The longer axis follows the image's aspect ratio, so the tensor fed to the
            model is not square.
    """

    encoder: Literal["vits", "vitb", "vitl"]
    features: int
    out_channels: tuple[int, int, int, int]
    max_depth: float | None = None
    input_size: int = 518

    @property
    def relative(self) -> bool:
        """Whether this variant predicts relative (unitless) rather than metric depth."""
        return self.max_depth is None

    @property
    def unit(self) -> Literal["metres"] | None:
        """What one unit of the output means.

        ``"metres"`` for the metric models; ``None`` for the relative ones, whose output is
        inverse depth on an arbitrary scale. Derived from :attr:`max_depth` rather than stored,
        so the two can never disagree. Never guessed -- a caller that needs metres must pick a
        variant that has them.
        """
        return None if self.max_depth is None else "metres"


# The three backbone sizes, shared by all three training regimes.
_ENCODERS: dict[str, tuple[str, int, tuple[int, int, int, int]]] = {
    "small": ("vits", 64, (48, 96, 192, 384)),
    "base": ("vitb", 128, (96, 192, 384, 768)),
    "large": ("vitl", 256, (256, 512, 1024, 1024)),
}

# Regime -> (variant prefix, max_depth). Upstream's own recommendation is 20 m indoors and
# 80 m outdoors; the value is not stored in the checkpoint, so it has to come from here.
_REGIMES: tuple[tuple[str, float | None], ...] = (
    ("", None),          # relative depth
    ("indoor-", 20.0),   # fine-tuned on Hypersim
    ("outdoor-", 80.0),  # fine-tuned on Virtual KITTI 2
)

MODEL_SPECS: dict[str, ModelSpec] = {
    f"{prefix}{size}": ModelSpec(
        encoder=encoder,
        features=features,
        out_channels=out_channels,
        max_depth=max_depth,
    )
    for prefix, max_depth in _REGIMES
    for size, (encoder, features, out_channels) in _ENCODERS.items()
}


def get_spec(name: str) -> ModelSpec:
    """Look up a variant specification by name.

    Args:
        name: A variant name, e.g. ``"large"`` or ``"outdoor-small"``.

    Returns:
        The frozen specification for that variant.

    Raises:
        KeyError: If the name is not a released variant.
    """
    try:
        return MODEL_SPECS[name]
    except KeyError:
        raise KeyError(
            f"unknown variant {name!r}; released variants are {', '.join(MODEL_SPECS)}"
        ) from None
