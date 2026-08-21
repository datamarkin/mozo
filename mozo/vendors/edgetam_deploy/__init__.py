# SPDX-License-Identifier: Apache-2.0
"""Deployment-only EdgeTAM image segmentation, extracted from ``facebookresearch/EdgeTAM``.

    >>> from mozo.vendors.edgetam_deploy import Segmenter   # doctest: +SKIP
    >>> segmenter = Segmenter("edgetam.pt")
    >>> found = segmenter.predict(image, boxes=[40, 60, 300, 480])

See ``README.md`` for the prompt conventions and ``PROVENANCE.md`` for what was extracted.
"""

from .predictor import Segmentation, Segmenter

__all__ = ["Segmentation", "Segmenter"]
