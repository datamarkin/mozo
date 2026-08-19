# SPDX-License-Identifier: Apache-2.0
"""Deployment-only YOLO11 detection, driven entirely by the checkpoint's own record of itself.

A ``.pt`` file records its whole module tree -- every layer's class name, its wiring and its
leaf hyperparameters -- so this package reads all of that and builds the matching ``torch.nn``
modules. The only hand-written part is the dataflow of the composite blocks, in :mod:`flow`.
"""

from .model import Detections, Detector, detect

__all__ = ["Detections", "Detector", "detect"]
