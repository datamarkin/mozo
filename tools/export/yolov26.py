#!/usr/bin/env python3
"""Export YOLO26 variants to ONNX, into the local ``weights/`` tree.

    python tools/export/yolov26.py nano
    python tools/export/yolov26.py nano small --revision 2026-08-19

The machinery is ``tools/export/_detection.py``, shared with the other detection families. What is
specific to YOLO26 is that its graph carries more than the others' -- and that it publishes **no
CoreML**, for two independent reasons.

This family is NMS-free, so the exported graph contains the box decode, the anchor grid and a
two-stage top-k, and returns ``(1, max_det, 6)`` rows of ``x1, y1, x2, y2, score, class`` rather
than a raw head. Whoever runs it applies a threshold and undoes the letterbox, which is all the
vendor's ``detect`` does.

**No CoreML.** Converting fails before it starts::

    Op "chosen" (op_type: gather_along_axis) Input indices expects tensor of dtype
    ['int32', 'uint16', 'int16'] but got tensor[1,300,4,fp32]

-- the in-graph top-k's gather indices lose their integer dtype through ``expand``. Casting them
explicitly to int32 is a real fix and it converts; what happens next is the same Metal compiler
abort YOLO11 hits, ``MPSGraphExecutable.mm: failed assertion 'MLIR pass manager failed'``, from the
same ``C2PSA`` attention block. Off the GPU it does run, accurately -- 0.00006 px on CPU and the
Neural Engine -- at 22.6 ms against 13.1 ms for torch on MPS. So the fix is recorded here rather
than applied: it unlocks an artifact slower than the one already published.

**No fp16.** Measured on YOLOv8 and recorded in ``tools/export/yolov8.py``.

One thing to watch if the verification threshold is ever lowered. The graph returns a fixed 300
rows padded with noise, and ``_compare`` pairs detections by position. Above any real threshold
that is sound -- at ``CONF`` the positional and content-based comparisons agree to 0.0005 px -- but
across all 300 rows position-pairing reads 0.54 px where content pairing reads 0.001, because two
executors of the same top-k are free to break ties differently. The vendor ships
``compare_detection_sets`` for that, and it is the right tool if this ever needs to compare
everything the graph emits.

**These weights are AGPL-3.0, and so is anything exported from them.** The graph produced here
contains the weights. It lands in the same revision directory as the LICENSE and NOTICE that
``tools/fetch/yolov26.py`` placed, which is what keeps those terms travelling with it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _detection import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("yolov26", coreml=False, description=__doc__))
