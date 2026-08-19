#!/usr/bin/env python3
"""Write the class vocabulary for each YOLOv8 variant into the local ``weights/`` tree.

Bootstrap tooling; never ships. It exists because a graph records no class names: the torch path
reads them out of the checkpoint it loads, but an ONNX artifact carries only numbers, so the names
have to be published beside it or that runtime can name nothing it finds.

    python tools/labels/yolov8.py                 # every fetched variant
    python tools/labels/yolov8.py nano small

The names come from the checkpoint itself -- ``model.names``, as the weights record it -- so mozo
invents nothing and a fine-tuned checkpoint publishes its own vocabulary rather than COCO's.
YOLOv8 numbers its classes contiguously from 0, unlike RF-DETR's original sparse COCO ids; both
are read from their own source rather than derived from each other.

Run this for every variant you fetched, not only the ones you export. A variant published without
labels serves unnamed detections on any runtime that does not carry the names itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mozo.vendors.yolov8_deploy import Detector  # noqa: E402


def write_labels(variant: str, revision: str, weights_dir: Path) -> None:
    """Read one variant's class names out of its checkpoint and publish them beside it."""
    revision_dir = weights_dir / "yolov8" / variant / revision
    checkpoint = revision_dir / "torch-fp32.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"{checkpoint} is missing. Run tools/fetch/yolov8.py {variant} first.")

    # Built without fusing: nothing here runs the network, and folding batch norm into every
    # convolution to read a dictionary off the side of it is pure waste.
    names = Detector(checkpoint, device="cpu", fuse_norm=False).names
    destination = revision_dir / "labels.json"
    destination.write_text(json.dumps([{"id": int(i), "name": name} for i, name in sorted(names.items())]))
    print(f"  {variant:<8} {len(names)} classes -> {destination.relative_to(weights_dir)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", help="variants to write (default: every one fetched)")
    parser.add_argument("--revision", default="2026-08-19", help="revision directory to write into")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    args = parser.parse_args()

    family = args.weights_dir / "yolov8"
    wanted = args.variants or sorted(p.name for p in family.iterdir() if p.is_dir())
    if not wanted:
        raise SystemExit(f"nothing fetched under {family}. Run tools/fetch/yolov8.py first.")

    for variant in wanted:
        write_labels(variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} vocabularies written. Run tools/generate_manifest.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
