#!/usr/bin/env python3
"""Fetch Depth Anything V2's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/depth_anything_v2.py                    # everything
    python tools/fetch/depth_anything_v2.py small indoor-large

Each file is checked against the SHA-256 that Hugging Face records for the blob, read from its
API rather than transcribed from a README, so a mismatch means the bytes changed under us.

Licensing is not uniform across these nine, which is why it is handled per variant here rather
than by copying one file:

    relative small          Apache-2.0
    relative base, large    CC-BY-NC-4.0     <- non-commercial
    all six metric          Apache-2.0 per their model cards

That last row deserves a second look before anyone relies on it. The metric models are
fine-tunes of the relative ones, so metric base and large descend from CC-BY-NC-4.0 weights
while their cards claim Apache-2.0. mozo publishes what the card says and records the ancestry
in the NOTICE beside it; it does not launder the one into the other.

The output is ``weights/depth_anything_v2/<variant>/<revision>/torch-fp32.pth`` and a NOTICE
naming the authors and source -- attribution that CC-BY-NC requires to travel with the copy, and
that Apache-2.0 asks for too. The LICENSE beside them is not written by this script: a licence is
part of what is published, so it lives in the weights tree like any other artifact. If one is
missing this script says so and prints the command to fetch it.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

_HF = "https://huggingface.co/depth-anything"


@dataclass(frozen=True)
class Checkpoint:
    """One published checkpoint: where it lives, what it should hash to, and its terms."""

    variant: str
    repo: str
    filename: str
    sha256: str
    licence: str
    #: Upstream's own description of what the weights are, for the NOTICE.
    trained_on: str
    #: The relative-depth checkpoint this one was fine-tuned from, when that ancestor's terms
    #: differ from the card this one states. ``None`` when there is nothing to disclose.
    descends_from: str | None

    @property
    def url(self) -> str:
        return f"{_HF}/{self.repo}/resolve/main/{self.filename}"


_RELATIVE = "relative depth, trained on 595K synthetic and 62M+ real unlabelled images"
_HYPERSIM = "metric depth (indoor, 0-20 m), fine-tuned on Hypersim"
_VKITTI = "metric depth (outdoor, 0-80 m), fine-tuned on Virtual KITTI 2"

#: Size -> upstream's encoder tag. The repository and filename are built from these, because
#: upstream names all nine checkpoints to the same pattern.
_ENCODERS = {"small": "vits", "base": "vitb", "large": "vitl"}

#: Regime -> (variant prefix, repository infix, filename infix, description).
_REGIMES = (
    ("", "", "", _RELATIVE),
    ("indoor-", "Metric-Hypersim-", "metric_hypersim_", _HYPERSIM),
    ("outdoor-", "Metric-VKITTI-", "metric_vkitti_", _VKITTI),
)

#: The sizes whose *relative* checkpoint is published non-commercially. This one set decides both
#: the relative models' own licence and which metric fine-tunes have an ancestor worth disclosing,
#: so the two facts cannot drift apart the way a per-row transcription can.
_NON_COMMERCIAL_SIZES = frozenset({"base", "large"})

#: Variant -> SHA-256 of the blob, read from Hugging Face's API. The only irreducible column:
#: everything else about a row follows from its regime and size.
_DIGESTS = {
    "small": "715fade13be8f229f8a70cc02066f656f2423a59effd0579197bbf57860e1378",
    "base": "0d2b7002e62d39d655571c371333340bd88f67ab95050c03591555aa05645328",
    "large": "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345",
    "indoor-small": "b782898d8a3e8be1f639de33837ed85e9b4b73e40f8f5e5cd99067588d722545",
    "indoor-base": "9dc9e274c5eff55c6daf27b660c0ced0eca4e8593a6da90cdcb04d2b4d3f3fa2",
    "indoor-large": "6f82ff2bc543ac02ddff4aa31fa363676a8305dd3ccf04e80e2af115a044cb6d",
    "outdoor-small": "9203e538d35255c90dda4b7fedb47ff33fe725497bcca3b1e53b3a65ee63f0cb",
    "outdoor-base": "4dad67a7cc10b462bca48e6b8569c762b8eb3c1adada170a3851a6d3ba37bb3e",
    "outdoor-large": "239b1054a369e66da2576e9a118d6d7c12d90dc8ebe609579a9a09cd8e05fe38",
}


def _checkpoint(prefix: str, repo_infix: str, file_infix: str, trained_on: str, size: str) -> Checkpoint:
    """Build one row from its regime and size, deriving everything but the digest."""
    variant = f"{prefix}{size}"
    metric = trained_on is not _RELATIVE
    non_commercial = size in _NON_COMMERCIAL_SIZES
    return Checkpoint(
        variant=variant,
        repo=f"Depth-Anything-V2-{repo_infix}{size.capitalize()}",
        filename=f"depth_anything_v2_{file_infix}{_ENCODERS[size]}.pth",
        sha256=_DIGESTS[variant],
        # The metric fine-tunes state Apache-2.0 on their own cards whatever they descend from.
        licence="CC-BY-NC-4.0" if non_commercial and not metric else "Apache-2.0",
        trained_on=trained_on,
        descends_from=size if metric and non_commercial else None,
    )


#: Variant -> checkpoint. Keys are mozo's variant names; upstream identifies the same models by
#: encoder (``vits``/``vitb``/``vitl``) and repository suffix.
CHECKPOINTS: dict[str, Checkpoint] = {
    f"{prefix}{size}": _checkpoint(prefix, repo_infix, file_infix, trained_on, size)
    for prefix, repo_infix, file_infix, trained_on in _REGIMES
    for size in _ENCODERS
}


_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _notice(variant: str, entry: Checkpoint) -> str:
    """Attribution that travels with the weights, as both licences require."""
    descent = ""
    if entry.descends_from is not None:
        descent = (
            "\nNote: this checkpoint is a fine-tune of Depth-Anything-V2-"
            f"{entry.descends_from.capitalize()}, whose own model card states CC-BY-NC-4.0. "
            "The Apache-2.0 above is the licence this checkpoint's own card states.\n"
        )
    return (
        f"Depth Anything V2 -- {variant}\n"
        f"{entry.trained_on}\n\n"
        "Copyright (c) 2024 TikTok and The University of Hong Kong.\n"
        "Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao.\n\n"
        f"Source:  {entry.url}\n"
        f"Project: https://github.com/DepthAnything/Depth-Anything-V2\n"
        f"Licence: {entry.licence} (full text in the LICENSE file beside this one)\n"
        f"{descent}"
    )


#: Where each licence's canonical text lives. The text itself is deliberately *not* kept in this
#: repository: a licence is part of what gets published, so it belongs in the weights tree beside
#: the checkpoint it covers -- which is where ``tools/generate_manifest.py`` looks for it, and
#: what ends up in the bucket. Upstream ships no licence file for these weights (the HF repos hold
#: only a README and the ``.pth``), so the text is placed once per revision.
LICENCE_SOURCES = {
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0.txt",
    "CC-BY-NC-4.0": "https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt",
}


def require_licence(revision_dir: Path, licence: str) -> None:
    """Fail if the revision has no LICENSE, saying exactly how to supply one.

    Deliberately not carried forward from a previous revision. A new revision is where an
    upstream relicence would show up, and silently copying the old terms is how that gets
    missed.
    """
    if (revision_dir / "LICENSE").is_file():
        return
    raise SystemExit(
        f"{revision_dir} has no LICENSE, and every published revision ships one.\n"
        f"This checkpoint is {licence}:\n"
        f"    curl -sL {LICENCE_SOURCES[licence]} -o {revision_dir / 'LICENSE'}"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, and place it with its licence and notice."""
    entry = CHECKPOINTS[variant]
    target = weights_dir / "depth_anything_v2" / variant / revision / "torch-fp32.pth"

    if target.is_file() and _sha256(target) == entry.sha256:
        print(f"  {variant:<15} already present, sha256 matches", flush=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Staged beside the target, not in $TMPDIR: these are gigabyte checkpoints, and a
        # cross-filesystem move would copy every byte a second time. Here the finish is a rename.
        # The digest is computed while the bytes are written, so nothing is read back to verify.
        staged = target.with_name(target.name + ".part")
        print(f"  {variant:<15} downloading {entry.url}", flush=True)
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(entry.url, timeout=300) as response, staged.open("wb") as out:
                for chunk in iter(lambda: response.read(_CHUNK), b""):
                    digest.update(chunk)
                    out.write(chunk)

            actual = digest.hexdigest()
            if actual != entry.sha256:
                raise SystemExit(
                    f"{variant}: sha256 mismatch. Expected {entry.sha256}, got {actual}. "
                    f"Upstream may have re-released this checkpoint; do not publish it until the "
                    f"change is understood."
                )
            staged.replace(target)
        finally:
            staged.unlink(missing_ok=True)
        print(f"  {variant:<15} ok, {target.stat().st_size / 1e6:.1f} MB, {entry.licence}", flush=True)

    (target.parent / "NOTICE").write_text(_notice(variant, entry))
    require_licence(target.parent, entry.licence)


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

    non_commercial = [v for v in wanted if CHECKPOINTS[v].licence == "CC-BY-NC-4.0"]
    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}")
    if non_commercial:
        print(f"non-commercial (CC-BY-NC-4.0): {', '.join(non_commercial)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
