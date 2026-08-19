"""YOLO12 detection, on whichever runtime the host can execute.

The architecture lives in :mod:`mozo.vendors.yolov12_deploy`, which rebuilds the network from what
the checkpoint itself records rather than from any model definition. The weights come from
:func:`mozo.weights.resolve`. Which of them runs the forward pass depends on what mozo publishes
for the variant and what the machine can execute -- a torch module, an ONNX graph, or a CoreML
package.

Only the middle step changes between runtimes. Letterboxing, non-maximum suppression and the
mapping back to source pixels come from the vendor either way, so the paths cannot drift.

Unlike YOLO11, this family does publish CoreML. Its area-attention blocks convert and run cleanly
where YOLO11's ``C2PSA`` aborts Apple's Metal graph compiler, and on Apple silicon CoreML is the
fastest artifact by a wide margin. Nothing here selects it: ``auto`` takes the best artifact each
family actually publishes.

Everything else this class does is shared with the other YOLO families and lives in
:class:`~mozo.adapters._yolo.YOLOPredictor`.

**These weights are AGPL-3.0.** mozo's code is Apache-2.0; the two are separate works travelling
together. Serving predictions from them over a network puts AGPL-3.0 section 13 obligations on
whoever runs the service. The NOTICE published beside every checkpoint says so in full.

    >>> model = YOLOv12Predictor("nano")                        # doctest: +SKIP
    >>> model = YOLOv12Predictor("nano", runtime="onnx-fp32")   # doctest: +SKIP
    >>> detections = model.predict(image, threshold=0.25)       # doctest: +SKIP
"""

from __future__ import annotations

from ..vendors import yolov12_deploy
from ._yolo import YOLOPredictor


class YOLOv12Predictor(YOLOPredictor):
    """One loaded YOLO12 variant. See :class:`~mozo.adapters._yolo.YOLOPredictor` for the arguments."""

    FAMILY = "yolov12"
    DISPLAY = "YOLO12"
    VENDOR = yolov12_deploy
    VARIANTS = ("nano", "small", "medium", "large", "xlarge")
