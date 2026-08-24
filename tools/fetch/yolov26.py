#!/usr/bin/env python3
"""Fetch Ultralytics' published YOLO26 checkpoints into the local ``weights/`` tree.

    python tools/fetch/yolov26.py                 # everything
    python tools/fetch/yolov26.py nano small

This family publishes no CoreML -- see ``tools/export/yolov26.py``. Run that for the ONNX
graph, then ``tools/labels/yolov26.py`` and ``tools/generate_manifest.py``.

The download, the digest check, the NOTICE and the licence gate are in
``tools/fetch/_ultralytics.py``, shared with the other Ultralytics families -- the release tag it
pins is what the NOTICE names as the corresponding source, and one family naming a stale one would
be a compliance document going quietly wrong.

The output is ``weights/yolov26/<variant>/<revision>/torch-fp32.pth`` plus a NOTICE.
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
    "nano": ("yolo26n.pt", "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"),
    "small": ("yolo26s.pt", "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"),
    "medium": ("yolo26m.pt", "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"),
    "large": ("yolo26l.pt", "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26"),
    "xlarge": ("yolo26x.pt", "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"),
    # Instance segmentation: the same backbone and neck with a Segment26 head, published in the
    # same release under the same terms. Named on the RF-DETR convention, where a segmentation
    # variant sits beside its detection counterpart in one family rather than forming a second.
    "seg-nano": ("yolo26n-seg.pt", "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"),
    "seg-small": ("yolo26s-seg.pt", "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4"),
    "seg-medium": ("yolo26m-seg.pt", "16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5"),
    "seg-large": ("yolo26l-seg.pt", "636024306410afa1732692322fba57d22ea2b1c2f07613fcee131a93d7dd380c"),
    "seg-xlarge": ("yolo26x-seg.pt", "92b3de0065766a17180d6219858717dc9d03cdce8a3ca9576c97fd75aabb64f3"),
}

if __name__ == "__main__":
    raise SystemExit(run("yolov26", "YOLO26", CHECKPOINTS, __doc__))
