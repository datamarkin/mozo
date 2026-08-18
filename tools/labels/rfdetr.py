#!/usr/bin/env python3
"""Write the class vocabulary for each RF-DETR variant into the local ``weights/`` tree.

Bootstrap tooling; never ships. It exists because RF-DETR's COCO checkpoints do not record
their own class names, so the names have to be supplied once, from a source, rather than
assumed at inference time.

    python tools/labels/rfdetr.py

The names come from the installed ``rfdetr`` package -- the project that trained these weights,
so it is the authority on what they were trained to find. The **ids** do not come from there.
Upstream stores its names as a contiguous list of 80 while its models emit COCO's original ids,
which run to 90 with ten gaps; indexing one by the other reports a person as a bicycle. Pairing
upstream's names with the original id space is the whole content of this file.

Verified: the name sequence matches ``rfdetr.assets.coco_classes.COCO_CLASS_NAMES`` exactly and
in order; the ids are strictly increasing across 1-90 with exactly the ten ids COCO never
assigned; and ids 1, 47, 67, 73 and 77 were confirmed against a photograph the model reads as
person, cup, dining table, laptop and cell phone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

#: Each contiguous COCO index (0-79) to its id in COCO's original space. The gaps are ids
#: assigned to categories that were never annotated.
CONTIGUOUS_TO_COCO = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

#: Detection and segmentation variants share COCO's vocabulary. The keypoint preview is a
#: person-only model and is handled separately.
COCO_VARIANTS = (
    "nano", "small", "medium", "large",
    "seg-nano", "seg-small", "seg-medium", "seg-large",
)


def coco_labels() -> list[dict]:
    """Return COCO's 80 categories keyed by their id in the original space.

    Raises:
        SystemExit: If the installed ``rfdetr`` no longer lists 80 classes, which would mean the
            id mapping below no longer describes it.
    """
    # This file is called rfdetr.py, so its own directory shadows the package it needs.
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
    from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

    if len(COCO_CLASS_NAMES) != len(CONTIGUOUS_TO_COCO):
        raise SystemExit(
            f"upstream now lists {len(COCO_CLASS_NAMES)} classes, not {len(CONTIGUOUS_TO_COCO)}. "
            f"The id mapping no longer applies and must be revisited."
        )
    return [{"id": i, "name": n} for n, i in zip(COCO_CLASS_NAMES, CONTIGUOUS_TO_COCO)]


def write(variant: str, labels: list[dict], weights_dir: Path) -> None:
    """Write ``labels.json`` into every revision of *variant* that has been fetched.

    The revisions come from the tree rather than a default this script and ``tools/fetch`` would
    both have to carry: disagreeing on one would leave a revision published with weights and no
    names, and nothing would fail.
    """
    variant_dir = weights_dir / "rfdetr" / variant
    revisions = sorted(d for d in variant_dir.iterdir() if d.is_dir()) if variant_dir.is_dir() else []
    if not revisions:
        print(f"  {variant:<17} skipped, not fetched")
        return
    for directory in revisions:
        (directory / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    print(f"  {variant:<17} {len(labels)} classes in {', '.join(d.name for d in revisions)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    args = parser.parse_args()

    labels = coco_labels()
    for variant in COCO_VARIANTS:
        write(variant, labels, args.weights_dir)

    print("\nRun tools/generate_manifest.py to pick these up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
