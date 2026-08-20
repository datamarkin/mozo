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

import logging as _logging


class _SilenceDinov2BackboneWarnings(_logging.Filter):
    """Drop the two warnings ``dinov2.py`` emits on every single model build.

    They say the DINOv2 backbone weights are not being loaded because the positional-encoding
    count and the patch size differ from DINOv2's own. Both are unconditionally true for RF-DETR,
    which is built at a different patch size by design, so the warning fires twice for every
    predictor mozo constructs and never means anything different.

    It is provably nothing to act on here: ``weights.load_state_dict_into`` raises on any missing
    key, with a message saying it "would stay randomly initialized". So every learned parameter
    comes from the RF-DETR checkpoint whether or not DINOv2's would have been loaded first, which
    is exactly the case the warning itself calls "not a problem".

    Filtered by originating module rather than by message text -- rewording upstream should not
    silently un-silence this -- and only for this one module, because the same ``rf-detr`` logger
    carries five other warnings about genuine fallbacks that must stay audible. The sibling
    depth_anything_v2_deploy can silence its whole logger only because DINOv2 has a dedicated one
    there. Doing it here keeps the vendored files verbatim, so a re-extraction has nothing to
    re-apply.
    """

    def filter(self, record: _logging.LogRecord) -> bool:
        return not (record.levelno == _logging.WARNING and record.module == "dinov2")


_logging.getLogger("rf-detr").addFilter(_SilenceDinov2BackboneWarnings())

from .config import MODEL_SPECS, ModelSpec, get_spec  # noqa: E402
from .predictor import Predictor  # noqa: E402

__all__ = ["MODEL_SPECS", "ModelSpec", "Predictor", "get_spec"]
