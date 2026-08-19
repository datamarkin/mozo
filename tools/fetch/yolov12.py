#!/usr/bin/env python3
"""Fetch Ultralytics' published YOLO12 checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/yolov12.py                 # everything
    python tools/fetch/yolov12.py nano small

Each file is checked against the SHA-256 that GitHub records for the release asset, read from its
API rather than transcribed from a README, so a mismatch means the bytes changed under us.

These weights are **AGPL-3.0**, and mozo's own code is Apache-2.0. Those are two separately
licensed works travelling together, which the GPL's aggregation clause allows and which is how
every Linux distribution ships copyleft and permissive packages side by side. What redistribution
does require is that the licence text travel with the bytes and that recipients be told where the
corresponding source is -- so this script writes a NOTICE naming the exact release each file came
from and pointing at Ultralytics' repository, and refuses to place a checkpoint whose revision has
no LICENSE.

An ONNX or CoreML export of these weights contains the weights, so it is AGPL-3.0 too. Exporting is not
laundering, and ``tools/export/yolov12.py`` writes into the same revision directory as this one so
the licence and notice already sitting there cover it.

The output is ``weights/yolov12/<variant>/<revision>/torch-fp32.pth`` plus a NOTICE. Run
``tools/export/yolov12.py`` next for the ONNX and CoreML artifacts, then ``tools/labels/yolov12.py`` and
``tools/generate_manifest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from common import (  # noqa: E402
    download_verified, require_licence, ultralytics_notice, variant_parser,
)

#: Family name, which is the directory the weights land in and the key the manifest uses.
FAMILY = "yolov12"

#: What to call this family to a reader, in the NOTICE.
DISPLAY = "YOLO12"

#: The release these checkpoints were taken from. Pinned rather than tracking "latest": a release
#: tag is what makes the corresponding source identifiable, and AGPL-3.0 section 6 asks for a
#: place recipients can actually get it from.
RELEASE = "v8.4.0"

_ASSETS = f"https://github.com/ultralytics/assets/releases/download/{RELEASE}"

#: Where the corresponding source for these weights lives, for the NOTICE.
SOURCE_URL = "https://github.com/ultralytics/ultralytics"

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

#: Where the licence's canonical text lives. Deliberately not kept in this repository: a licence is
#: part of what gets published, so it belongs in the weights tree beside the checkpoint it covers,
#: which is where ``tools/generate_manifest.py`` looks for it.
LICENCE_SOURCE_URL = "https://www.gnu.org/licenses/agpl-3.0.txt"


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, and place it with its licence and notice."""
    filename, expected = CHECKPOINTS[variant]
    target = weights_dir / FAMILY / variant / revision / "torch-fp32.pth"
    download_verified(f"{_ASSETS}/{filename}", target, expected, label=variant)

    (target.parent / "NOTICE").write_text(
        ultralytics_notice(DISPLAY, variant, filename, RELEASE, _ASSETS, SOURCE_URL))
    require_licence(target.parent, "AGPL-3.0", LICENCE_SOURCE_URL)


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights").parse_args()

    wanted = args.variants or list(CHECKPOINTS)
    unknown = [v for v in wanted if v not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(CHECKPOINTS)}")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}")
    print("all AGPL-3.0: mozo redistributes them under their own terms, not its own")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
