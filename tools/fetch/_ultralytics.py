#!/usr/bin/env python3
"""Fetch one Ultralytics-trained family's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored packages carry none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

Each file is checked against the SHA-256 that GitHub records for the release asset, so a mismatch
means the bytes changed under us.

These weights are **AGPL-3.0**, and mozo's own code is Apache-2.0. Those are two separately
licensed works travelling together, which the GPL's aggregation clause allows and which is how
every Linux distribution ships copyleft and permissive packages side by side. What redistribution
does require is that the licence text travel with the bytes and that recipients be told where the
corresponding source is -- so this writes a NOTICE naming the exact release each file came from,
and refuses to place a checkpoint whose revision has no LICENSE.

Shared across the families because ``RELEASE`` is the load-bearing part. It is what the NOTICE
names as "where the corresponding source is", and Ultralytics publishes every generation from one
assets repository. Written into each family's script separately, a release bump reaches whichever
scripts someone remembered, and the ones that were missed keep naming the old tag in a compliance
document while still passing their digest checks, because the digests move with the tag.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from common import (  # noqa: E402
    download_verified, require_licence, ultralytics_notice, variant_parser,
)

#: The release these checkpoints are taken from. Pinned rather than tracking "latest": a release
#: tag is what makes the corresponding source identifiable, and AGPL-3.0 section 6 asks for a
#: place recipients can actually get it from.
RELEASE = "v8.4.0"

ASSETS = f"https://github.com/ultralytics/assets/releases/download/{RELEASE}"

#: Where the corresponding source for these weights lives, for the NOTICE.
SOURCE_URL = "https://github.com/ultralytics/ultralytics"

#: Where the licence's canonical text lives. Deliberately not kept in this repository: a licence is
#: part of what gets published, so it belongs in the weights tree beside the checkpoint it covers,
#: which is where ``tools/generate_manifest.py`` looks for it.
LICENCE_SOURCE_URL = "https://www.gnu.org/licenses/agpl-3.0.txt"


def fetch(family: str, display: str, checkpoints: dict, variant: str,
          revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, and place it with its licence and notice."""
    filename, expected = checkpoints[variant]
    target = weights_dir / family / variant / revision / "torch-fp32.pth"
    download_verified(f"{ASSETS}/{filename}", target, expected, label=variant)

    (target.parent / "NOTICE").write_text(
        ultralytics_notice(display, variant, filename, RELEASE, ASSETS, SOURCE_URL))
    require_licence(target.parent, "AGPL-3.0", LICENCE_SOURCE_URL)


def run(family: str, display: str, checkpoints: dict, description: str = "") -> int:
    """Fetch the variants named on the command line, or every one this family publishes."""
    args = variant_parser(description or f"fetch {family}", ROOT / "weights").parse_args()

    wanted = args.variants or list(checkpoints)
    unknown = [v for v in wanted if v not in checkpoints]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(checkpoints)}")

    for variant in wanted:
        fetch(family, display, checkpoints, variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}")
    print("all AGPL-3.0: mozo redistributes them under their own terms, not its own")
    return 0
