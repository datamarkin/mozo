# SPDX-License-Identifier: Apache-2.0
"""mozo's EdgeTAM adapter: a click or a box in, PixelFlow detections out.

EdgeTAM is SAM 2 distilled for phones -- a 9.1 M-parameter image path against SAM 2 tiny's
31.4 M -- and on this machine it encodes a 2 MP photograph in 272 ms against SAM 2 tiny's 439 ms.
Its masks agree with SAM 2 tiny's at 0.94 IoU on box prompts and 0.87 median on single clicks, so
it is the one to reach for when the encode is the cost that matters.

Both its code and its published weights are Apache-2.0, which is not true of every family here.
"""

from __future__ import annotations

from ..vendors import edgetam_deploy
from ._promptable import PromptablePredictor

__all__ = ["EdgeTamPredictor"]



class EdgeTamPredictor(PromptablePredictor):
    """Promptable segmentation on EdgeTAM.

    Examples:
        >>> model = EdgeTamPredictor()                                   # doctest: +SKIP
        >>> found = model.predict("street.jpg", points=[[820, 640]], labels=[1])   # doctest: +SKIP
        >>> found[0].confidence                                          # doctest: +SKIP
        0.899
    """

    FAMILY = "edgetam"
    DISPLAY = "EdgeTAM"
    VENDOR = edgetam_deploy
    #: Meta publishes a single EdgeTAM rather than a size ladder, so there is one name.
    VARIANTS = ("edgetam",)
