#!/usr/bin/env python3
"""Generate ``mozo/manifest.json`` from the local ``weights/`` directory.

The directory is the source of truth. Paths, sizes, hashes, artifact keys and the ``latest``
pointer are all read off the disk, so the manifest cannot disagree with what is actually there,
and nothing in it is typed by hand.

Layout::

    weights/<family>/<variant>/<revision>/<key>.<ext>

``<revision>`` is the publish date in ISO form, so revisions sort lexically and the newest is
``latest``. Each file's *stem* is its artifact key: ``torch-fp32.pth`` publishes as ``torch-fp32``.
``LICENSE`` has no extension, so it is simply an artifact named ``LICENSE`` -- required in every
revision, and fetched alongside whatever else a caller asks for. Shipping the licence text with
the weights is the whole of the compliance story; there is no separate licence field to keep in
sync with it.

Run from the repository root::

    python tools/generate_manifest.py

Output is sorted and stably formatted, so re-running it on unchanged inputs produces a
byte-identical file. That makes "did someone forget to regenerate?" a one-line check in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE_URL = "https://dtmfiles.com/mozo/v1"

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
MANIFEST_FILE = ROOT / "mozo" / "manifest.json"

#: Present in every revision. Downloading weights without their terms is not a thing we do.
LICENCE_KEY = "LICENSE"

_HASH_CHUNK_BYTES = 1 << 20


class GenerateError(RuntimeError):
    """Raised when a revision directory is malformed."""


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of *path*, read in chunks so large checkpoints do not land in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(revision_dir: Path) -> list[Path]:
    """Return the publishable files in *revision_dir*: regular files, no dotfiles, no subdirectories.

    Dotfiles are skipped rather than rejected because ``.DS_Store`` appears unbidden on macOS and
    failing the build over it would teach people to ignore the build.
    """
    return sorted(p for p in revision_dir.iterdir() if p.is_file() and not p.name.startswith("."))


def _scan_revision(revision_dir: Path, root: Path) -> dict[str, dict[str, object]]:
    """Hash every artifact in one revision directory and return them keyed by file stem.

    Raises:
        GenerateError: If the directory holds no artifacts, two files share a stem, or ``LICENSE``
            is absent.
    """
    artifacts: dict[str, dict[str, object]] = {}
    for path in _artifact_files(revision_dir):
        key = path.stem
        if key in artifacts:
            raise GenerateError(
                f"{revision_dir}: two files share the artifact key {key!r}. "
                "Each artifact must have a unique stem."
            )
        artifacts[key] = {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }

    if not artifacts:
        raise GenerateError(f"{revision_dir}: no artifacts found.")
    if LICENCE_KEY not in artifacts:
        raise GenerateError(
            f"{revision_dir}: no {LICENCE_KEY} file. Every revision ships upstream's licence "
            "alongside the weights."
        )
    return artifacts


def _scan_variant(variant_dir: Path, root: Path) -> dict[str, object]:
    """Scan every revision of one model variant and return its manifest entry.

    Raises:
        GenerateError: If the variant has no revision directories.
    """
    revisions = {
        d.name: {"artifacts": _scan_revision(d, root)}
        for d in sorted(p for p in variant_dir.iterdir() if p.is_dir())
    }
    if not revisions:
        raise GenerateError(f"{variant_dir}: no revision directories.")

    # ISO dates sort lexically, so the newest name is the newest revision.
    return {"latest": max(revisions), "revisions": revisions}


def build_manifest(weights_dir: Path = WEIGHTS_DIR) -> dict[str, object]:
    """Walk *weights_dir* and return the manifest as a dict."""
    if not weights_dir.is_dir():
        raise GenerateError(f"{weights_dir} does not exist.")

    models: dict[str, object] = {}
    for family_dir in sorted(p for p in weights_dir.iterdir() if p.is_dir()):
        for variant_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            models[f"{family_dir.name}/{variant_dir.name}"] = _scan_variant(variant_dir, weights_dir)

    return {"schema": 1, "base_url": BASE_URL, "models": models}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR, help="tree to scan")
    parser.add_argument("--out", type=Path, default=MANIFEST_FILE, help="manifest to write")
    args = parser.parse_args()

    try:
        manifest = build_manifest(args.weights_dir)
    except GenerateError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    artifacts = sum(
        len(revision["artifacts"])
        for model in manifest["models"].values()
        for revision in model["revisions"].values()
    )
    print(f"wrote {args.out}: {len(manifest['models'])} models, {artifacts} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
