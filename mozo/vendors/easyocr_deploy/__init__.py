# SPDX-License-Identifier: Apache-2.0
"""EasyOCR, extracted for deployment.

Text detection and recognition in two graphs: CRAFT finds the lines, a CRNN reads them. See
PROVENANCE.md for what this was taken from and what changed.

    >>> from mozo.vendors.easyocr_deploy import Reader, SPECS       # doctest: +SKIP
    >>> reader = Reader("torch-fp32.pth", SPECS["english"])          # doctest: +SKIP
    >>> [(r.text, round(r.confidence, 2)) for r in reader(image)]    # doctest: +SKIP
    [('Hello World', 1.0)]
"""

from .config import SPECS, VARIANTS, Spec
from .predictor import Reader, Region

__all__ = ["Reader", "Region", "SPECS", "Spec", "VARIANTS"]
