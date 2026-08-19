#!/usr/bin/env python3
"""Export YOLOv8 variants to ONNX and CoreML, into the local ``weights/`` tree.

    python tools/export/yolov8.py nano
    python tools/export/yolov8.py nano small --revision 2026-08-19

The machinery is ``tools/export/_detection.py``, shared with the other detection families. What is
specific to YOLOv8 is that it publishes CoreML, and why there is no fp16.

Two artifacts land in ``weights/yolov8/<variant>/<revision>/``, both mapping a letterboxed
``(1, 3, imgsz, imgsz)`` batch to the raw head output ``(1, 4 + classes, anchors)``. That is a
classic head, so whoever runs one still applies non-maximum suppression -- which mozo does with
the vendor's own :func:`~mozo.vendors.yolov8_deploy.detect`, so every runtime shares it:

``onnx-fp32``    the graph.
``coreml-fp32``  the same model as a CoreML package, zipped because an ``.mlpackage`` is a
                 directory and an artifact is a file. It is the fastest way to run these models on
                 Apple silicon by a wide margin -- measured on this laptop, nano runs 4.2 ms
                 against 7.9 ms on torch MPS and 52.2 ms on torch CPU, and xlarge 41.1 ms against
                 48.8 and 330.2 -- at a worst box error of 0.0004 px.

There is deliberately no fp16 path, in either format. torch fp16 on MPS is *slower* than fp32
(8.2 ms against 7.9 on nano) and moves boxes 0.76 px; ONNX fp16 is slower too (43.2 against 34.4).
CoreML fp16 is genuinely faster, about 1.4x, and costs 2.3 px on nano, 1.5 on small, 1.4 on medium
and 7.4 on xlarge -- measured by pairing detections by IoU at a serving threshold, where fp16 finds
every object fp32 finds, same count, all paired.

An earlier version of this note reported 636 px on nano and 341 on medium and called the error
unbounded. Those numbers were wrong, and the way they were produced is worth recording: they came
from ``_compare``, which pairs detections by position. At ``CONF`` the head emits hundreds of
near-tied noise boxes whose order any perturbation reshuffles, so position-pairing subtracts
unrelated boxes. "636 px" was simply the largest box coordinate in the tensor. Moving the decode
out of the fp16 graph and running it in fp32 was tried as a fix and is not one -- 2.6 px, and
2.3 ms slower -- so the error is in the convolutions, not the anchor arithmetic. Should CUDA
tensor cores justify revisiting fp16, start from measurements, and pair by IoU.

The class names a graph cannot carry are published by ``tools/labels/yolov8.py``, which runs over
every variant you fetched rather than only the ones exported here.

**These weights are AGPL-3.0, and so is anything exported from them.** The graph produced here
contains the weights. It lands in the same revision directory as the LICENSE and NOTICE that
``tools/fetch/yolov8.py`` placed, which is what keeps those terms travelling with it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _detection import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("yolov8", coreml=True, description=__doc__))
