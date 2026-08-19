"""YOLO26 detection, on whichever runtime the host can execute.

The architecture lives in :mod:`mozo.vendors.yolov26_deploy`, which rebuilds the network from what
the checkpoint itself records rather than from any model definition. The weights come from
:func:`mozo.weights.resolve`. Which of them runs the forward pass depends on what mozo publishes
for the variant and what the machine can execute -- a torch module, or the same graph as ONNX.

This family is NMS-free. Its head fires once per object, so the network returns a ranked detection
list and the vendor's ``detect`` has no suppression step at all. Nothing here changes for that:
the adapter passes an image, a forward pass, a size and a threshold, which is all this family needs
and a subset of what the others take.

There is no CoreML artifact, as for YOLO11 and unlike YOLOv8 and YOLO12. Two separate things stop
it -- the converter rejects the in-graph top-k's gather indices, and once that is worked around the
Metal compiler aborts on the attention block -- and the configuration that does run is slower than
torch on MPS. ``tools/export/yolov26.py`` records both. ``auto`` needs no special case, because it
only ever chooses among what a variant publishes.

**These weights are AGPL-3.0.** mozo's code is Apache-2.0; the two are separate works travelling
together. Serving predictions from them over a network puts AGPL-3.0 section 13 obligations on
whoever runs the service. The NOTICE published beside every checkpoint says so in full.

    >>> model = YOLOv26Predictor("nano")                        # doctest: +SKIP
    >>> model = YOLOv26Predictor("nano", runtime="onnx-fp32")   # doctest: +SKIP
    >>> detections = model.predict(image, threshold=0.25)       # doctest: +SKIP
"""

from __future__ import annotations

from ..vendors import yolov26_deploy
from ._yolo import YOLOPredictor


class YOLOv26Predictor(YOLOPredictor):
    """One loaded YOLO26 variant. See :class:`~mozo.adapters._yolo.YOLOPredictor` for the arguments."""

    FAMILY = "yolov26"
    DISPLAY = "YOLO26"
    VENDOR = yolov26_deploy
    VARIANTS = ("nano", "small", "medium", "large", "xlarge")
