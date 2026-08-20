"""Plumbing the publishing tools share.

Bootstrap tooling; none of this ships in the wheel. What lives here is the part of fetching that
is the same whatever the model is -- hash a file, download it and check it against a digest the
publisher stated, refuse to place weights whose revision has no licence -- plus the licence notice
the YOLO families must publish verbatim.

The notice is the reason this module exists rather than being a nice-to-have. It is a compliance
artifact: it names the release the weights came from and tells the recipient where the
corresponding source is, which is what makes redistributing AGPL-3.0 weights lawful. Written out
once per family, an amendment to it reaches whichever families someone remembered to edit, and the
rest keep publishing the old wording with nothing to detect it.

Every fetching tool uses this, not only the YOLO families -- which is what keeps ``algorithm`` and
``timeout`` honest parameters rather than settings nobody exercises. RF-DETR publishes md5 rather
than sha256 and wants a shorter timeout; Depth Anything V2 names a different licence per variant.
Those are the only three axes, and they were the three already varying between the copies.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

#: Read size for hashing and streaming. One megabyte: large enough that the syscall overhead
#: disappears, small enough that a checkpoint never has to be held in memory to be verified.
CHUNK = 1 << 20

#: The revision directory the publishing tools write into by default. One constant rather than the
#: same date typed into every tool's argparse block, where the fetch, export and labels tools for
#: one family could disagree about which revision they were building.
REVISION = "2026-08-19"

#: The photographs every export gate compares on. Shared for the same reason ``REVISION`` is:
#: each family's gate should widen when a photograph is added here, and three private copies of
#: this path meant adding one widened whichever gates someone remembered.
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "images"


def fixtures() -> list[Path]:
    """Photographs to compare on. Real images, because synthetic noise proves nothing here."""
    images = sorted(p for p in FIXTURES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"no fixture images in {FIXTURES}. Add photographs to verify against.")
    return images


def digest(path: Path, algorithm: str = "sha256") -> str:
    """Return the hex digest of a file, read in chunks rather than loaded."""
    running = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            running.update(chunk)
    return running.hexdigest()


def download_verified(url: str, target: Path, expected: str, *, algorithm: str = "sha256",
                      label: str = "", width: int = 8, timeout: int = 300, detail: str = "") -> str:
    """Download *url* to *target* unless it is already there and already matches *expected*.

    Returns the verified digest. *width* is the column the label is padded to, so a family whose
    variant names are longer than another's keeps its output aligned; *detail* is appended to the
    success line for whatever else a caller wants to show.

    Staged beside the target rather than in ``$TMPDIR``: a cross-filesystem move would copy every
    byte a second time, whereas here the finish is a rename. The digest is computed while the
    bytes are written, so nothing is read back to verify it.

    Raises:
        SystemExit: If what arrived does not match *expected*. A digest mismatch means the bytes
            changed under us, which is not something to publish through.
    """
    if target.is_file() and digest(target, algorithm) == expected:
        print(f"  {label:<{width}} already present, {algorithm} matches", flush=True)
        return expected

    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".part")
    print(f"  {label:<{width}} downloading {url}", flush=True)
    running = hashlib.new(algorithm)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, staged.open("wb") as out:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                running.update(chunk)
                out.write(chunk)

        actual = running.hexdigest()
        if actual != expected:
            raise SystemExit(
                f"{label}: {algorithm} mismatch. Expected {expected}, got {actual}. "
                f"Upstream may have re-released this file; do not publish it until the change "
                f"is understood."
            )
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)
    print(f"  {label:<{width}} ok, {target.stat().st_size / 1e6:.1f} MB"
          f"{', ' + detail if detail else ''}", flush=True)
    return actual


def require_licence(revision_dir: Path, licence: str, source_url: str) -> None:
    """Fail if the revision has no LICENSE, saying exactly how to supply one.

    Deliberately not carried forward from a previous revision. A new revision is where an upstream
    relicence would show up, and silently copying the old terms is how that gets missed.
    """
    if (revision_dir / "LICENSE").is_file():
        return
    raise SystemExit(
        f"{revision_dir} has no LICENSE, and every published revision ships one.\n"
        f"These weights are {licence}:\n"
        f"    curl -sL {source_url} -o {revision_dir / 'LICENSE'}"
    )


def ultralytics_notice(display: str, variant: str, filename: str, release: str,
                       assets_url: str, source_url: str) -> str:
    """The attribution and source pointer AGPL-3.0 requires to travel with a copy of the weights.

    One wording for every Ultralytics-trained family. *display* is what to call the family to a
    reader, e.g. ``"YOLOv8"``; the rest identify the exact upstream artifact, because "where the
    corresponding source is" has to mean a release someone can actually go and get.
    """
    return (
        f"{display} -- {variant}\n\n"
        "Copyright (c) Ultralytics.\n\n"
        f"Source:  {assets_url}/{filename}\n"
        f"Release: {release} of https://github.com/ultralytics/assets\n"
        f"Project: {source_url}\n"
        "Licence: AGPL-3.0-only (full text in the LICENSE file beside this one)\n\n"
        "These weights are licensed under the GNU Affero General Public License v3.0, or under a\n"
        "commercial licence obtained from Ultralytics. mozo redistributes them unmodified and\n"
        "under those same terms; mozo's own code is Apache-2.0 and is a separate work.\n\n"
        "The corresponding source for these weights is the Ultralytics project at the URL above.\n\n"
        "Any export in this directory -- ONNX, CoreML or otherwise -- contains these weights and\n"
        "is covered by this same licence, not by mozo's.\n\n"
        "If you serve predictions from these weights over a network, AGPL-3.0 section 13 places\n"
        "obligations on you. Complying with them is your responsibility, not mozo's.\n"
    )


def variant_parser(description: str, weights_dir: Path, *, required: bool = False,
                   revision: str = REVISION) -> argparse.ArgumentParser:
    """The argument triple every publishing tool takes: variants, revision, weights directory.

    *required* distinguishes the tools that act on everything by default (fetch, labels) from the
    ones that will not guess (export).
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+" if required else "*",
                        help="variant names, e.g. nano small (default: every one known)")
    parser.add_argument("--revision", default=revision, help="revision directory to work in")
    parser.add_argument("--weights-dir", type=Path, default=weights_dir)
    return parser
