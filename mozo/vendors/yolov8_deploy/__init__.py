# SPDX-License-Identifier: Apache-2.0
"""Deployment-only YOLOv8: rebuild a checkpoint's network, run it, get boxes back.

The model is reconstructed from the facts the checkpoint records -- layer classes, wiring and every
leaf module's hyperparameters -- so any width, depth or class count in that format runs unchanged.
Depends on ``torch``, ``numpy`` and ``opencv-python`` and nothing else; every import inside this
package is relative, so the directory can be moved without edits.

Images arrive as RGB ``uint8`` arrays. Decoding belongs to :mod:`mozo.image`, which is the only
place in mozo where bytes become pixels.

Examples:
    >>> from mozo.vendors.yolov8_deploy import Detector   # doctest: +SKIP
    >>> detector = Detector("yolov8n.pt")                 # doctest: +SKIP
    >>> found = detector.predict(rgb_array, conf=0.25)    # doctest: +SKIP
"""

from .model import Detections, Detector, detect

__all__ = ["Detections", "Detector", "detect"]
