# SPDX-License-Identifier: Apache-2.0
"""Deployment-only YOLO26 detection, driven entirely by the checkpoint's own record of itself.

A ``.pt`` file records its whole module tree -- every layer's class name, its wiring and its leaf
hyperparameters -- so this package reads all of that and builds the matching ``torch.nn`` modules.
The hand-written parts are the dataflow of the composite blocks (:mod:`flow`) and the geometry the
head leaves implicit -- the anchor grid, the box decode and the top-k, in :mod:`network`.

This family is NMS-free: the head fires once per object and the network returns a ranked detection
list, so nothing here suppresses anything. :mod:`image` has a letterbox and its inverse and no
third function.
"""

from .model import Detections, Detector, detect

__all__ = ["Detections", "Detector", "detect"]
