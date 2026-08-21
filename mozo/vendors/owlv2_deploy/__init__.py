# SPDX-License-Identifier: Apache-2.0
"""OWLv2, extracted for deployment: open-vocabulary detection, image path only.

A phrase in, boxes out. See ``PROVENANCE.md`` for what this derives from and what it leaves
behind, and ``README.md`` for how to drive it.

    >>> from mozo.vendors.owlv2_deploy import Detector       # doctest: +SKIP
    >>> model = Detector("torch-fp32.pth", "base-ensemble")  # doctest: +SKIP
    >>> found = model.predict(image, ["cat", "kettle"])      # doctest: +SKIP
"""

from .predictor import Detection, Detector

__all__ = ["Detection", "Detector"]
