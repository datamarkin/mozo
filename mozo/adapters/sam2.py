# SPDX-License-Identifier: Apache-2.0
"""mozo's SAM 2 adapter: a click or a box in, PixelFlow detections out.

SAM 2 is the reference promptable segmenter this family is measured against -- four published
sizes, from a 31.4 M-parameter tiny to a large. Only the image path is served here; the video
tracker is not part of the vendored package.

Unlike SAM 3, SAM 2's published checkpoints are Apache-2.0, the same as its code. Nothing in this
family carries a separate weights licence.
"""

from __future__ import annotations

from ..vendors import sam2_deploy
from ._promptable import PromptablePredictor

__all__ = ["Sam2Predictor"]



class Sam2Predictor(PromptablePredictor):
    """Promptable segmentation on SAM 2.

    Examples:
        >>> model = Sam2Predictor("tiny")                                # doctest: +SKIP
        >>> found = model.predict("street.jpg", boxes=[40, 60, 300, 480])  # doctest: +SKIP
        >>> found[0].class_name                                          # doctest: +SKIP
        None
    """

    FAMILY = "sam2"
    DISPLAY = "SAM 2"
    VENDOR = sam2_deploy
    #: Smallest first: it is the default, and it is the one that fits on an edge device.
    VARIANTS = ("tiny", "small", "base_plus", "large")
