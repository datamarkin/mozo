#!/usr/bin/env python3
"""Write the joint vocabulary for each ViTPose variant into the local ``weights/`` tree.

Bootstrap tooling; never ships. It exists because a checkpoint's seventeen heatmap channels carry
no names of their own, so the names have to be supplied once, from a source, rather than assumed at
inference time.

    python tools/labels/vitpose.py

The names are **COCO's**, from the ``person_keypoints`` annotations, because that is the vocabulary
the published heads were trained against. They are not taken from the checkpoints' ``config.json``,
which spells the same seventeen joints as ``Nose``, ``L_Eye``, ``R_Eye`` and so on. That spelling
is Hugging Face's, not the dataset's, and mozo already publishes COCO's for
``rfdetr/keypoint-preview``: two families naming the same joint two ways would make
``show_names=True`` render differently depending on which model drew the skeleton.

The **order** is not assumed. :func:`check_against_upstream` reads each published checkpoint's
``id2label`` and holds it against this list position by position, case-folded and with the
abbreviations expanded. A names list one entry out would publish silently and mislabel every joint
after the gap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import variant_parser  # noqa: E402

#: COCO's 17 person keypoints, in the order the dataset defines them -- which is the order the
#: heatmap channels come out in. The same list ``tools/labels/rfdetr.py`` publishes, because it is
#: the same vocabulary and not a coincidence: both models were trained on COCO.
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

#: How the checkpoints' own ``id2label`` abbreviates a side. Expanded before comparing, so the
#: check tests the ordering rather than the spelling -- which is the thing that can be wrong.
ABBREVIATIONS = {"l_": "left_", "r_": "right_"}


def keypoint_labels() -> list[dict]:
    """Return the single category these checkpoints predict, with its joint names.

    One entry, because every published head has one class. The ``keypoints`` key is what
    :func:`pixelflow.detections.from_arrays` reads to name a joint, exactly as ``name`` names a
    class -- so the joint vocabulary travels with the weights on the same mechanism, and nothing
    downstream has to know which family it came from.

    Each joint is a ``{"name": ...}`` dict rather than a bare string, which is the shape
    ``pixelflow.labels.get_label_info`` reads. It carries no ``id``: a keypoint's id *is* its
    position in this list, assigned by the converter from the array index, and writing the number
    down a second time would create a second source for one fact and no check that they agree.
    """
    return [{"id": 1, "name": "person",
             "keypoints": [{"name": name} for name in KEYPOINT_NAMES]}]


def check_against_upstream(repo: str) -> None:
    """Hold :data:`KEYPOINT_NAMES` against one published checkpoint's own ``id2label``.

    Needs the network, and is skipped when it is unavailable -- this is a check on a fact that does
    not change between runs, not a step the publish depends on.

    Raises:
        SystemExit: If the checkpoint names a different number of joints, or names them in a
            different order.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
                f"https://huggingface.co/{repo}/raw/main/config.json", timeout=30) as response:
            declared = json.load(response)["id2label"]
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"  could not reach {repo} ({error}); ordering not re-checked")
        return

    theirs = [declared[str(index)].lower() for index in range(len(declared))]
    for short, full in ABBREVIATIONS.items():
        theirs = [name.replace(short, full, 1) for name in theirs]

    if theirs != KEYPOINT_NAMES:
        differences = [f"{index}: {a!r} vs {b!r}"
                       for index, (a, b) in enumerate(zip(theirs, KEYPOINT_NAMES)) if a != b]
        raise SystemExit(
            f"{repo} names its joints differently from this file: "
            f"{differences or f'{len(theirs)} joints against {len(KEYPOINT_NAMES)}'}"
        )
    print(f"  {repo.split('/')[-1]:24s} 17 joints, order matches")


def write(variant: str, labels: list[dict], weights_dir: Path) -> None:
    """Write ``labels.json`` into every revision of *variant* that has been fetched.

    The revisions come from the tree rather than a default this script and ``tools/fetch`` would
    both have to carry: disagreeing on one would leave a revision published with weights and no
    names, and nothing would fail.
    """
    variant_dir = weights_dir / "vitpose" / variant
    revisions = sorted(d for d in variant_dir.iterdir() if d.is_dir()) if variant_dir.is_dir() else []
    if not revisions:
        print(f"  {variant:<6} skipped, not fetched")
        return
    for directory in revisions:
        (directory / "labels.json").write_text(json.dumps(labels, indent=2) + "\n")
    print(f"  {variant:<6} 17 joints in {', '.join(d.name for d in revisions)}")


def main() -> int:
    from tools.fetch.vitpose import REPOSITORIES

    args = variant_parser(__doc__, ROOT / "weights").parse_args()
    wanted = args.variants or list(REPOSITORIES)

    labels = keypoint_labels()
    for variant in wanted:
        check_against_upstream(REPOSITORIES[variant])
        write(variant, labels, args.weights_dir)

    print("\nRun tools/generate_manifest.py to pick these up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
