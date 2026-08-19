#!/usr/bin/env python3
"""Fetch Ultralytics' published YOLO12 checkpoints into the local ``weights/`` tree.

    python tools/fetch/yolov12.py                 # everything
    python tools/fetch/yolov12.py nano small

Two artifacts follow from these: run ``tools/export/yolov12.py`` for the ONNX graph
and the CoreML package, then ``tools/labels/yolov12.py`` and ``tools/generate_manifest.py``.

The download, the digest check, the NOTICE and the licence gate are in
``tools/fetch/_ultralytics.py``, shared with the other Ultralytics families -- the release tag it
pins is what the NOTICE names as the corresponding source, and one family naming a stale one would
be a compliance document going quietly wrong.

The output is ``weights/yolov12/<variant>/<revision>/torch-fp32.pth`` plus a NOTICE.
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
    "nano": ("yolo12n.pt", "419ff3dca37d69bacc93a50fa0c186a1c6f9fe62fae0f108b0872829689e9ca6"),
    "small": ("yolo12s.pt", "e915c2c4286e3f6f8610ef106fa3f94a7b8c19b30eccede5887e22c33ef75f58"),
    "medium": ("yolo12m.pt", "4c6d179786eddf6134ee469ae2f4ce04cbe4e9d1a47d6b669d9cd6b9c6c513d8"),
    "large": ("yolo12l.pt", "0babd8dc8f775bb64bb052debdff3d8b9e9b57efa9d7bfa11c84bb82c3fec336"),
    "xlarge": ("yolo12x.pt", "682ce8dadee004dbe964950f1bf3eda451671815a6ed62db80b398916b9b7c6f"),
}

if __name__ == "__main__":
    raise SystemExit(run("yolov12", "YOLO12", CHECKPOINTS, __doc__))
