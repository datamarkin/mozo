#!/usr/bin/env python3
"""Export YOLO12 variants to ONNX and CoreML, into the local ``weights/`` tree.

    python tools/export/yolov12.py nano
    python tools/export/yolov12.py nano small --revision 2026-08-19

The machinery is ``tools/export/_detection.py``, shared with the other detection families. What is
specific to YOLO12 is that it publishes CoreML where YOLO11 cannot.

Two artifacts land in ``weights/yolov12/<variant>/<revision>/``, both mapping a letterboxed
``(1, 3, imgsz, imgsz)`` batch to the raw head output ``(1, 4 + classes, anchors)``. That is a
classic head, so whoever runs one still applies non-maximum suppression -- which mozo does with
the vendor's own :func:`~mozo.vendors.yolov12_deploy.detect`, so every runtime shares it:

``onnx-fp32``    the graph.
``coreml-fp32``  the same model as a CoreML package, zipped because an ``.mlpackage`` is a
                 directory and an artifact is a file. Measured on this laptop, nano runs 7.0 ms
                 against 14.3 ms on torch MPS and 113.7 ms on torch CPU, at a worst box error of
                 0.0017 px.

This family converts to CoreML where YOLO11 does not. Its area-attention blocks (``A2C2f``,
``ABlock``, ``AAttn``) place cleanly; it is specifically YOLO11's ``C2PSA`` that makes Apple's
Metal graph compiler abort. Checked rather than assumed, on every compute-unit setting.

**No fp16.** The measurements are the YOLOv8 family's and live in ``tools/export/yolov8.py``
rather than being paraphrased here: fp16 finds every object fp32 finds and puts them in slightly
the wrong place, which is not a trade a detector should make.

**These weights are AGPL-3.0, and so is anything exported from them.** The graphs produced here
contain the weights. They land in the same revision directory as the LICENSE and NOTICE that
``tools/fetch/yolov12.py`` placed, which is what keeps those terms travelling with them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _detection import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("yolov12", coreml=True, description=__doc__))
