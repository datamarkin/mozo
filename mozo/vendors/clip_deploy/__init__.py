# SPDX-License-Identifier: Apache-2.0
"""CLIP's inference path, extracted from openai/CLIP.

Two encoders trained until their outputs landed in the same space: an image becomes a vector, a
phrase becomes a vector, and the dot product between them says how well they match. That one
property gives both zero-shot classification and text-to-image retrieval.

    >>> from mozo.vendors.clip_deploy import Encoder, SPECS
    >>> encoder = Encoder(weights, SPECS["base"])              # doctest: +SKIP
    >>> encoder.classify(image, ["a forklift", "a person"])    # doctest: +SKIP

See ``PROVENANCE.md`` for what was taken, what was left, and where this diverges.
"""

from .config import SPECS, VARIANTS, Spec
from .predictor import Encoder

__all__ = ["SPECS", "VARIANTS", "Encoder", "Spec"]
