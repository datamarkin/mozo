#!/usr/bin/env python3
"""Check that mozo returns exactly what the vendor does, for YOLO12.

    python tools/verify/yolov12.py                        # fixtures, nano, every runtime
    python tools/verify/yolov12.py --variant small
    python tools/verify/yolov12.py photo.jpg other.jpg    # your own images

The comparison itself is in tools/verify/_detection.py, shared across the detection families
because it is a gate: a stale second copy keeps exiting zero while checking the wrong thing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _detection import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("yolov12", __doc__))
