#!/usr/bin/env python3
"""Fetch Ultralytics' published YOLO11 checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/yolov11.py                 # everything
    python tools/fetch/yolov11.py nano small

Each file is checked against the SHA-256 that GitHub records for the release asset, read from its
API rather than transcribed from a README, so a mismatch means the bytes changed under us.

These weights are **AGPL-3.0**, and mozo's own code is Apache-2.0. Those are two separately
licensed works travelling together, which the GPL's aggregation clause allows and which is how
every Linux distribution ships copyleft and permissive packages side by side. What redistribution
does require is that the licence text travel with the bytes and that recipients be told where the
corresponding source is -- so this script writes a NOTICE naming the exact release each file came
from and pointing at Ultralytics' repository, and refuses to place a checkpoint whose revision has
no LICENSE.

An ONNX export of these weights contains the weights, so it is AGPL-3.0 too. Exporting is not
laundering, and ``tools/export/yolov11.py`` writes into the same revision directory as this one so
the licence and notice already sitting there cover it.

The output is ``weights/yolov11/<variant>/<revision>/torch-fp32.pth`` plus a NOTICE. Run
``tools/export/yolov11.py`` next for the ONNX artifact, then ``tools/labels/yolov11.py`` and
``tools/generate_manifest.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

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
    "nano": ("yolo11n.pt", "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"),
    "small": ("yolo11s.pt", "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"),
    "medium": ("yolo11m.pt", "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"),
    "large": ("yolo11l.pt", "9ebd0e09d59811db4b1d61e2bc6730649608b1ac47f8dd01e2da6bca7c20023f"),
    "xlarge": ("yolo11x.pt", "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"),
}

#: Where the licence's canonical text lives. Deliberately not kept in this repository: a licence is
#: part of what gets published, so it belongs in the weights tree beside the checkpoint it covers,
#: which is where ``tools/generate_manifest.py`` looks for it.
LICENCE_SOURCE_URL = "https://www.gnu.org/licenses/agpl-3.0.txt"

_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _notice(variant: str, filename: str) -> str:
    """Attribution and the source pointer that AGPL-3.0 requires to travel with the copy."""
    return (
        f"YOLO11 -- {variant}\n\n"
        "Copyright (c) Ultralytics.\n\n"
        f"Source:  {_ASSETS}/{filename}\n"
        f"Release: {RELEASE} of https://github.com/ultralytics/assets\n"
        f"Project: {SOURCE_URL}\n"
        "Licence: AGPL-3.0-only (full text in the LICENSE file beside this one)\n\n"
        "These weights are licensed under the GNU Affero General Public License v3.0, or under a\n"
        "commercial licence obtained from Ultralytics. mozo redistributes them unmodified and\n"
        "under those same terms; mozo's own code is Apache-2.0 and is a separate work.\n\n"
        "The corresponding source for these weights is the Ultralytics project at the URL above.\n\n"
        "Any ONNX or other export in this directory contains these weights and is covered by\n"
        "this same licence, not by mozo's.\n\n"
        "If you serve predictions from these weights over a network, AGPL-3.0 section 13 places\n"
        "obligations on you. Complying with them is your responsibility, not mozo's.\n"
    )


def require_licence(revision_dir: Path) -> None:
    """Fail if the revision has no LICENSE, saying exactly how to supply one.

    Deliberately not carried forward from a previous revision. A new revision is where an upstream
    relicence would show up, and silently copying the old terms is how that gets missed.
    """
    if (revision_dir / "LICENSE").is_file():
        return
    raise SystemExit(
        f"{revision_dir} has no LICENSE, and every published revision ships one.\n"
        f"These weights are AGPL-3.0:\n"
        f"    curl -sL {LICENCE_SOURCE_URL} -o {revision_dir / 'LICENSE'}"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, and place it with its licence and notice."""
    filename, expected = CHECKPOINTS[variant]
    target = weights_dir / "yolov11" / variant / revision / "torch-fp32.pth"
    url = f"{_ASSETS}/{filename}"

    if target.is_file() and _sha256(target) == expected:
        print(f"  {variant:<8} already present, sha256 matches", flush=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Staged beside the target, not in $TMPDIR: a cross-filesystem move would copy every byte
        # a second time. Here the finish is a rename. The digest is computed while the bytes are
        # written, so nothing is read back to verify.
        staged = target.with_name(target.name + ".part")
        print(f"  {variant:<8} downloading {url}", flush=True)
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=300) as response, staged.open("wb") as out:
                for chunk in iter(lambda: response.read(_CHUNK), b""):
                    digest.update(chunk)
                    out.write(chunk)

            actual = digest.hexdigest()
            if actual != expected:
                raise SystemExit(
                    f"{variant}: sha256 mismatch. Expected {expected}, got {actual}. "
                    f"Upstream may have re-released this checkpoint; do not publish it until the "
                    f"change is understood."
                )
            staged.replace(target)
        finally:
            staged.unlink(missing_ok=True)
        print(f"  {variant:<8} ok, {target.stat().st_size / 1e6:.1f} MB", flush=True)

    (target.parent / "NOTICE").write_text(_notice(variant, filename))
    require_licence(target.parent)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=None, help="variants to fetch (default: all)")
    parser.add_argument("--revision", default="2026-08-19", help="revision directory to write into")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    args = parser.parse_args()

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
