# SPDX-License-Identifier: Apache-2.0
"""Deployment-only SAM 3 concept segmentation.

    >>> from mozo.vendors.sam3_deploy import Segmenter   # doctest: +SKIP
    >>> segmenter = Segmenter("torch-fp32.pth")
    >>> found = segmenter.predict(image, "cow")

Derived from ``transformers/models/sam3`` (Apache-2.0), not from ``facebookresearch/sam3``, whose
code ships under the SAM License. The **weights** carry that licence regardless; mozo does not
redistribute them from this package. See ``README.md`` and ``PROVENANCE.md``.
"""

from .predictor import Segmenter, instances

__all__ = ["Segmenter", "instances"]
