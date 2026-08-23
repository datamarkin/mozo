# SPDX-License-Identifier: Apache-2.0
"""SigLIP 2's inference path, extracted from ``transformers/models/siglip``.

Two encoders trained until their outputs landed in the same space: an image becomes a vector, a
phrase becomes a vector, and how well they match is read off the two. Where CLIP was trained to
rank a batch, SigLIP scores each image-phrase pair on its own -- so the number that comes back
means something without a complete set of classes to normalise it against.

    >>> from mozo.vendors.siglip2_deploy import Encoder, SPECS
    >>> encoder = Encoder(weights, SPECS["base-224"])            # doctest: +SKIP
    >>> encoder.classify(image, ["a forklift", "a person"])      # doctest: +SKIP

See ``PROVENANCE.md`` for what was taken, what was left, and where this diverges.
"""

from .config import CONTEXT, SPECS, VARIANTS, VOCAB, Spec
from .predictor import Encoder

__all__ = ["CONTEXT", "SPECS", "VARIANTS", "VOCAB", "Encoder", "Spec"]
