#!/usr/bin/env python3
"""Fetch Ultralytics' published YOLO11 checkpoints into the local ``weights/`` tree.

    python tools/fetch/yolov11.py                 # everything
    python tools/fetch/yolov11.py nano small

This family publishes no CoreML -- see ``tools/export/yolov11.py``. Run that for the ONNX
graph, then ``tools/labels/yolov11.py`` and ``tools/generate_manifest.py``.

The download, the digest check, the NOTICE and the licence gate are in
``tools/fetch/_ultralytics.py``, shared with the other Ultralytics families -- the release tag it
pins is what the NOTICE names as the corresponding source, and one family naming a stale one would
be a compliance document going quietly wrong.

The output is ``weights/yolov11/<variant>/<revision>/torch-fp32.pth`` plus a NOTICE.
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
    "nano": ("yolo11n.pt", "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"),
    "small": ("yolo11s.pt", "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"),
    "medium": ("yolo11m.pt", "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"),
    "large": ("yolo11l.pt", "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"),
    "xlarge": ("yolo11x.pt", "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"),
    # Instance segmentation: the same backbone and neck with a Segment head, published in the same
    # release under the same terms. Named on the RF-DETR convention, where a segmentation variant
    # sits beside its detection counterpart in one family rather than forming a second.
    "seg-nano": ("yolo11n-seg.pt", "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"),
    "seg-small": ("yolo11s-seg.pt", "1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3"),
    "seg-medium": ("yolo11m-seg.pt", "eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf"),
    "seg-large": ("yolo11l-seg.pt", "cabe90049795dfc9a370b7934d6dec7f6b9e44a20e573b0ff81b7e205512c872"),
    "seg-xlarge": ("yolo11x-seg.pt", "4e53a5f5fd3ee2ae3361c62169c6bb3ed4ae251dd0e57606e230955aa52d919c"),
}

if __name__ == "__main__":
    raise SystemExit(run("yolov11", "YOLO11", CHECKPOINTS, __doc__))
