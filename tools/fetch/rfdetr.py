#!/usr/bin/env python3
"""Fetch RF-DETR's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package used to carry them and no longer should, because
where a checkpoint came from is a fact about publishing, not about inference.

    python tools/fetch/rfdetr.py                 # everything
    python tools/fetch/rfdetr.py small seg-large

Each file is checked against the MD5 Roboflow publishes for it before it is placed. Those
digests were transcribed from upstream's own ``ModelWeights`` manifest, so a mismatch means the
bytes changed under us -- a re-release, a truncated transfer, or the wrong file entirely. None
of those should pass silently into a tree we then hash and publish.

The output is ``weights/rfdetr/<variant>/<revision>/torch-fp32.pth``. The LICENSE beside it is
not written here: a licence is part of what is published, so it lives in the weights tree like
any other artifact, and this script only checks that one is present. Run
``tools/export/rfdetr.py`` next for the ONNX artifacts, then ``tools/generate_manifest.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from common import download_verified, require_licence  # noqa: E402

_BUCKET = "https://storage.googleapis.com/rfdetr"

#: Variant -> (url, expected md5). Keys are mozo's variant names, which is why they differ from
#: upstream's filenames: the seg checkpoints are published as ``-ft`` fine-tunes, and ``large``
#: carries a year in its name.
CHECKPOINTS: dict[str, tuple[str, str]] = {
    "nano": (f"{_BUCKET}/nano_coco/checkpoint_best_regular.pth", "fb6504cce7fbdc783f7a46991f07639f"),
    "small": (f"{_BUCKET}/small_coco/checkpoint_best_regular.pth", "fb37061c1af7bace359c91b723a8d5c1"),
    "medium": (f"{_BUCKET}/medium_coco/checkpoint_best_regular.pth", "7223f764a87b863f02eb8d52bf0ce2ee"),
    "large": (f"{_BUCKET}/rf-detr-large-2026.pth", "5cb72153541cbcb9aa6efa26222acc75"),
    "seg-nano": (f"{_BUCKET}/rf-detr-seg-n-ft.pth", "9995497791d0ff1664a1d9ddee9cfd20"),
    "seg-small": (f"{_BUCKET}/rf-detr-seg-s-ft.pth", "0a2a3006381d0c42853907e700eadd08"),
    "seg-medium": (f"{_BUCKET}/rf-detr-seg-m-ft.pth", "a49af1562c3719227ad43d0ca53b4c7a"),
    "seg-large": (f"{_BUCKET}/rf-detr-seg-l-ft.pth", "275f7b094909544ed2841c94a677d07e"),
    "keypoint-preview": (
        f"{_BUCKET}/rf-detr-keypoint-preview-xlarge.pth", "6de511943ee85a547d4c5cb527daf0eb"),
}

#: Where the licence's canonical text lives. Not kept in this repository: a licence is part of
#: what gets published, so it belongs in the weights tree beside the checkpoint it covers, which
#: is where ``tools/generate_manifest.py`` looks for it.
LICENCE_SOURCE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"

def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint and verify it, then check its licence is in place."""
    url, expected = CHECKPOINTS[variant]
    target = weights_dir / "rfdetr" / variant / revision / "torch-fp32.pth"
    download_verified(url, target, expected, algorithm="md5", label=variant,
                      width=17, timeout=120, detail=f"md5 {expected}")
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE_URL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=None, help="variants to fetch (default: all)")
    parser.add_argument("--revision", default="2026-08-18", help="revision directory to write into")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    args = parser.parse_args()

    wanted = args.variants or list(CHECKPOINTS)
    unknown = [v for v in wanted if v not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(CHECKPOINTS)}")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
