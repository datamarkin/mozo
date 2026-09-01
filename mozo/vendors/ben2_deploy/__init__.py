# ------------------------------------------------------------------------
# BEN2 -- Background Erase Network
# Copyright (c) 2025 Prama LLC. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deployment-only BEN2: load the published checkpoint, run an image, get an alpha matte.

Depends on ``torch``, ``numpy``, ``cv2`` and ``Pillow`` -- nothing else. Upstream additionally
requires ``timm`` and ``einops``; neither survives to inference. ``timm``'s ``DropPath`` is the
identity at eval, its ``trunc_normal_`` is initialisation the strict load overwrites, and its
``to_2tuple`` is one line. Every ``einops.rearrange`` had a fixed pattern with literal group
sizes, so each is a ``view``/``permute``/``reshape`` here. ``tools/verify/ben2.py`` checks all
nine rewrites against ``einops`` itself.

Pillow is not incidental: upstream resizes with PIL's LANCZOS, whose filter support scales with
the downsampling factor, and neither ``cv2.resize`` nor ``F.interpolate`` reproduces it. Dropping
it would silently change every matte this package produces.

Every import inside this package is relative, so the directory can be renamed or moved into
another project without edits.

Examples:
    >>> from ben2_deploy import Predictor  # doctest: +SKIP
    >>> predictor = Predictor.from_pretrained(path)  # doctest: +SKIP
    >>> alpha = predictor.matte(image)  # doctest: +SKIP
"""

from .config import BACKBONE, DECODER, INPUT, MEAN, STD
from .network import BEN_Base
from .predictor import Predictor

__all__ = ["BACKBONE", "BEN_Base", "DECODER", "INPUT", "MEAN", "STD", "Predictor"]
