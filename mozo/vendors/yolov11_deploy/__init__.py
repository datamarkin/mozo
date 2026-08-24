# SPDX-License-Identifier: Apache-2.0
"""Deployment-only YOLO11 inference, driven entirely by the checkpoint's own record of itself.

A ``.pt`` file records its whole module tree -- every layer's class name, its wiring and its
leaf hyperparameters -- so this package reads all of that and builds the matching ``torch.nn``
modules. The only hand-written part is the dataflow of the composite blocks, in :mod:`flow`.

Detection and instance segmentation both, and the checkpoint decides which: a ``Segment`` head
adds mask coefficients and a prototype branch, which :mod:`mask` turns into one boolean mask
per detection. A ``Detect`` head answers with ``masks=None``.
"""

from .model import Detections, Detector, detect

__all__ = ["Detections", "Detector", "detect"]
