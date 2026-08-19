#!/usr/bin/env python3
"""Export YOLO11 variants to ONNX, into the local ``weights/`` tree.

    python tools/export/yolov11.py nano
    python tools/export/yolov11.py nano small --revision 2026-08-19

The machinery is ``tools/export/_detection.py``, shared with the other detection families. What is
specific to YOLO11 is that it publishes **no CoreML**, and why.

Converting the full network produces a package that aborts the process when it runs::

    MPSGraphExecutable.mm:5070: failed assertion 'Error: MLIR pass manager failed'

That is an abort, not an exception: no ``except`` clause anywhere catches it, and a server that
loaded such an artifact would simply die. It was bisected to layer 10, the ``C2PSA`` attention
block -- a graph cut after layer 9 converts and runs, and the block converts and runs correctly on
its own, so it is a compiler pass failing on the assembled graph rather than an unsupported
operation. Rewriting the attention as a 3-D batched matmul (bit-identical in torch, ``max|d|`` 0.0)
did not help, and neither did ``macOS15`` as the deployment target. The sibling YOLO12, whose
area-attention blocks are a different design, converts cleanly.

Restricting compute units to CPU and the Neural Engine does produce a working, accurate package --
0.0002 px -- but at 23.5 ms against 10.4 ms for torch on MPS, so there is nothing to gain even
where it is safe. Nothing in ``mozo/runtimes.py`` special-cases this: ``auto`` only ever chooses
among what a variant publishes, and this family publishes no CoreML.

**No fp16 either.** Those measurements were taken on the CoreML path this family does not have, so
they belong to the sibling and live in ``tools/export/yolov8.py`` rather than being paraphrased
here where they cannot be reproduced.

**These weights are AGPL-3.0, and so is anything exported from them.** The graph produced here
contains the weights. It lands in the same revision directory as the LICENSE and NOTICE that
``tools/fetch/yolov11.py`` placed, which is what keeps those terms travelling with it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _detection import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("yolov11", coreml=False, description=__doc__))
