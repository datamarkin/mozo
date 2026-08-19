"""YOLOv8 detection, on whichever runtime the host can execute.

The architecture lives in :mod:`mozo.vendors.yolov8_deploy`, which rebuilds the network from what
the checkpoint itself records rather than from any model definition. The weights come from
:func:`mozo.weights.resolve`. Which of them runs the forward pass depends on what mozo publishes
for the variant and what the machine can execute -- a torch module, an ONNX graph, or a CoreML
package, the last of which is by a wide margin the fastest way to run these models on Apple
silicon.

Only the middle step changes between runtimes. Letterboxing, non-maximum suppression and the
mapping back to source pixels come from the vendor either way, so the paths cannot drift.

Everything this class does is shared with the other YOLO families and lives in
:class:`~mozo.adapters._yolo.YOLOPredictor`. What is specific to YOLOv8 is the vendored package
named below, and nothing else.

**These weights are AGPL-3.0.** mozo's code is Apache-2.0; the two are separate works travelling
together. Serving predictions from them over a network puts AGPL-3.0 section 13 obligations on
whoever runs the service. The NOTICE published beside every checkpoint says so in full.

    >>> model = YOLOv8Predictor("nano")                        # doctest: +SKIP
    >>> model = YOLOv8Predictor("nano", runtime="onnx-fp32")   # doctest: +SKIP
    >>> detections = model.predict(image, threshold=0.25)      # doctest: +SKIP
"""

from __future__ import annotations

from ..vendors import yolov8_deploy
from ._yolo import YOLOPredictor


class YOLOv8Predictor(YOLOPredictor):
    """One loaded YOLOv8 variant. See :class:`~mozo.adapters._yolo.YOLOPredictor` for the arguments."""

    FAMILY = "yolov8"
    DISPLAY = "YOLOv8"
    VENDOR = yolov8_deploy
    VARIANTS = ("nano", "small", "medium", "large", "xlarge")
