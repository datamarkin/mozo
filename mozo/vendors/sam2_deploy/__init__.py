# SPDX-License-Identifier: Apache-2.0
"""Deployment-only SAM 2 image segmentation, extracted from ``facebookresearch/sam2``.

    >>> from mozo.vendors.sam2_deploy import Segmenter   # doctest: +SKIP
    >>> segmenter = Segmenter("sam2.1_hiera_base_plus.pt")
    >>> found = segmenter.predict(image, boxes=[40, 60, 300, 480])

See ``README.md`` for the prompt conventions and ``PROVENANCE.md`` for what was extracted.
"""

from .predictor import Segmentation, Segmenter

__all__ = ["Segmentation", "Segmenter"]
