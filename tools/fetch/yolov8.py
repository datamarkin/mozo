#!/usr/bin/env python3
"""Fetch Ultralytics' published YOLOv8 checkpoints into the local ``weights/`` tree.

    python tools/fetch/yolov8.py                 # everything
    python tools/fetch/yolov8.py nano small

Two artifacts follow from these: run ``tools/export/yolov8.py`` for the ONNX graph
and the CoreML package, then ``tools/labels/yolov8.py`` and ``tools/generate_manifest.py``.

The download, the digest check, the NOTICE and the licence gate are in
``tools/fetch/_ultralytics.py``, shared with the other Ultralytics families -- the release tag it
pins is what the NOTICE names as the corresponding source, and one family naming a stale one would
be a compliance document going quietly wrong.

The output is ``weights/yolov8/<variant>/<revision>/torch-fp32.pth`` plus a NOTICE.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ultralytics import run  # noqa: E402

#: Variant -> (upstream filename, SHA-256 of the asset). Keys are mozo's variant names, which is
#: why they differ from upstream's: mozo names sizes in words across every family. The digests are
#: the ones GitHub publishes for the release assets, not ones computed from a local download --
#: a digest taken from the bytes you already have verifies nothing about them.
CHECKPOINTS: dict[str, tuple[str, str]] = {
    "nano": ("yolov8n.pt", "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"),
    "small": ("yolov8s.pt", "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"),
    "medium": ("yolov8m.pt", "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"),
    "large": ("yolov8l.pt", "64c9115303f6a25575f82200d1b22ec409fa6bd7d08d0313884fc20d919478cd"),
    "xlarge": ("yolov8x.pt", "3df4ada6b4dad6d657868f2fdf7faecfb34dcfccf3a25c4b82079064718524c8"),
}

if __name__ == "__main__":
    raise SystemExit(run("yolov8", "YOLOv8", CHECKPOINTS, __doc__))
