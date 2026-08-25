# SPDX-License-Identifier: Apache-2.0
"""ViTPose++, extracted for deployment: person boxes in, joints out.

Top-down pose estimation. This package is told where a person is and answers where their joints
are; it has no detector and does not want one. See ``PROVENANCE.md`` for what this derives from and
what it leaves behind, and ``README.md`` for how to drive it.

    >>> from mozo.vendors.vitpose_deploy import Predictor      # doctest: +SKIP
    >>> model = Predictor("torch-fp32.pth", "base")            # doctest: +SKIP
    >>> joints = model.predict(frame, [[10, 20, 110, 300]])    # doctest: +SKIP
"""

from .config import SPECS, Spec, get_spec
from .predictor import Predictor

__all__ = ["SPECS", "Predictor", "Spec", "get_spec"]
