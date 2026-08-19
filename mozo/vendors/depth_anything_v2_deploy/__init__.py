# ------------------------------------------------------------------------
# Depth Anything V2
# Copyright (c) 2024 TikTok / The University of Hong Kong. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deployment-only Depth Anything V2: build a released variant, load its weights, run inference.

Depends on ``torch``, ``torchvision``, ``numpy`` and ``cv2`` -- nothing else. ``cv2`` is not
incidental: upstream resizes with ``INTER_CUBIC``, and torchvision's bicubic is a different
resample, so dropping it would silently change every depth map this package produces.

Every import inside this package is relative, so the directory can be renamed or moved into
another project without edits.

Examples:
    >>> from depth_anything_v2_deploy import Predictor  # doctest: +SKIP
    >>> predictor = Predictor.from_pretrained("small", weights=path)  # doctest: +SKIP
    >>> depth = predictor.predict(image)  # doctest: +SKIP
"""

import logging as _logging

# The vendored DINOv2 attention warns at import time when xformers is absent. It is absent by
# design -- this package has no xformers dependency and falls back to plain attention, which is
# what every published number here was measured with. Silencing the logger keeps the file itself
# verbatim, so a future re-extraction has nothing to re-apply. It has to happen before the
# submodule imports below, because the warning fires while attention.py is being executed.
_logging.getLogger("dinov2").setLevel(_logging.ERROR)

from .config import MODEL_SPECS, ModelSpec, get_spec
from .predictor import Predictor

__all__ = ["MODEL_SPECS", "ModelSpec", "Predictor", "get_spec"]
