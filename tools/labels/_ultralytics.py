#!/usr/bin/env python3
"""Write the class vocabulary for each variant of an Ultralytics-trained family.

Bootstrap tooling; never ships. It exists because a graph records no class names: the torch path
reads them out of the checkpoint it loads, but an ONNX or CoreML artifact carries only numbers, so
the names have to be published beside it or that runtime can name nothing it finds.

The names come from the checkpoint itself, as the weights record them, so mozo invents nothing and
a fine-tuned checkpoint publishes its own vocabulary rather than COCO's. Every YOLO family numbers
its classes contiguously from 0, unlike RF-DETR's original sparse COCO ids -- which is exactly why
each is read from its own source rather than copied from the other.

Shared across the YOLO families, which differ here in nothing but the directory name. The
``labels.json`` shape is a contract with :func:`mozo.labels.resolve`; in three separate copies a
change to it would have to be made three times, and the family that missed it would publish a
vocabulary the resolver cannot read.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import variant_parser  # noqa: E402


def write_labels(family: str, vendor, variant: str, revision: str, weights_dir: Path) -> None:
    """Read one variant's class names out of its checkpoint and publish them beside it."""
    revision_dir = weights_dir / family / variant / revision
    checkpoint = revision_dir / "torch-fp32.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"{checkpoint} is missing. Run tools/fetch/{family}.py {variant} first.")

    # Built without fusing: nothing here runs the network, and folding batch norm into every
    # convolution to read a dictionary off the side of it is pure waste. The network is still
    # built rather than the names read straight from the pickle, because building is what checks
    # the vocabulary against the head's own class count -- 0.7 s across five variants to know the
    # names being published match the model publishing them.
    names = vendor.Detector(checkpoint, device="cpu", fuse_norm=False).names
    destination = revision_dir / "labels.json"
    destination.write_text(json.dumps([{"id": int(i), "name": name} for i, name in sorted(names.items())]))
    print(f"  {variant:<8} {len(names)} classes -> {destination.relative_to(weights_dir)}")


def run(family: str, description: str = "") -> int:
    """Publish the vocabulary for every variant named, or for every one already fetched."""
    args = variant_parser(description or f"labels for {family}", ROOT / "weights").parse_args()
    vendor = importlib.import_module(f"mozo.vendors.{family}_deploy")

    directory = args.weights_dir / family
    wanted = args.variants or sorted(p.name for p in directory.iterdir() if p.is_dir())
    if not wanted:
        raise SystemExit(f"nothing fetched under {directory}. Run tools/fetch/{family}.py first.")

    for variant in wanted:
        write_labels(family, vendor, variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} vocabularies written. Run tools/generate_manifest.py next.")
    return 0
