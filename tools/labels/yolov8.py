#!/usr/bin/env python3
"""Write the class vocabulary for each YOLOv8 variant into the local ``weights/`` tree.

    python tools/labels/yolov8.py                 # every fetched variant
    python tools/labels/yolov8.py nano small

Run this for every variant you fetched, not only the ones you export. A variant published without
labels serves unnamed detections on any runtime that does not carry the names itself. The work is
in ``tools/labels/_ultralytics.py``, shared across the YOLO families.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ultralytics import run  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run("yolov8", __doc__))
