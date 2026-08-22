# SPDX-License-Identifier: Apache-2.0
"""Grounding DINO's detection path, extracted from IDEA-Research/GroundingDINO.

Open-vocabulary detection: name a thing in words and get boxes for it, with no class list and no
training. See ``PROVENANCE.md`` for what was taken, what was left, and where this diverges.

    >>> from mozo.vendors.grounding_dino_deploy import Predictor, SPECS
    >>> model = Predictor(weights, SPECS["tiny"])        # doctest: +SKIP
    >>> model(image, ["a cat", "a laptop"])              # doctest: +SKIP
"""

from .boxes import Detection
from .config import SPECS, VARIANTS, Spec
from .predictor import SEPARATORS, Predictor, caption_for

__all__ = [
    "SEPARATORS", "SPECS", "VARIANTS", "Detection", "Predictor", "Spec", "caption_for",
]
