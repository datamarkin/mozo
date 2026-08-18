# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deployment-only RF-DETR: build a released variant, load its weights, run inference.

Depends on ``torch``, ``torchvision``, ``numpy``, and ``pillow`` — nothing else.  Every import inside this package is
relative, so the directory can be renamed or moved into another project without edits.

Examples:
    >>> from rfdetr_deploy import Predictor  # doctest: +SKIP
    >>> predictor = Predictor.from_pretrained("rfdetr-small", weights=path)  # doctest: +SKIP
    >>> results = predictor.predict("image.jpg", threshold=0.5)  # doctest: +SKIP
"""

from .config import MODEL_SPECS, ModelSpec, get_spec
from .predictor import Predictor

__all__ = ["MODEL_SPECS", "ModelSpec", "Predictor", "get_spec"]
