"""YOLO11 detection, on whichever runtime the host can execute.

The architecture lives in :mod:`mozo.vendors.yolov11_deploy`, which rebuilds the network from what
the checkpoint itself records rather than from any model definition. The weights come from
:func:`mozo.weights.resolve`. Which of them runs the forward pass depends on what mozo publishes
for the variant and what the machine can execute -- a torch module, or the same graph as ONNX.

Only the middle step changes between runtimes. Letterboxing, non-maximum suppression and the
mapping back to source pixels come from the vendor either way, so the two paths cannot drift.

There is no CoreML artifact for this family, unlike YOLOv8 and YOLO12. The ``C2PSA`` attention
block makes Apple's Metal graph compiler abort the process rather than raise, and the CoreML
configuration that does work is slower than torch on MPS. ``tools/export/yolov11.py`` records the
measurements; nothing here has to special-case it, because ``auto`` only ever chooses from what a
variant actually publishes.

Everything else this class does is shared with the other YOLO families and lives in
:class:`~mozo.adapters._yolo.YOLOPredictor`.

**These weights are AGPL-3.0.** mozo's code is Apache-2.0; the two are separate works travelling
together. Serving predictions from them over a network puts AGPL-3.0 section 13 obligations on
whoever runs the service. The NOTICE published beside every checkpoint says so in full.

    >>> model = YOLOv11Predictor("nano")                        # doctest: +SKIP
    >>> model = YOLOv11Predictor("nano", runtime="onnx-fp32")   # doctest: +SKIP
    >>> detections = model.predict(image, threshold=0.25)       # doctest: +SKIP
"""

from __future__ import annotations

from ..vendors import yolov11_deploy
from ._yolo import YOLOPredictor


class YOLOv11Predictor(YOLOPredictor):
    """One loaded YOLO11 variant. See :class:`~mozo.adapters._yolo.YOLOPredictor` for the arguments."""

    FAMILY = "yolov11"
    DISPLAY = "YOLO11"
    VENDOR = yolov11_deploy
    VARIANTS = ("nano", "small", "medium", "large", "xlarge")
