#!/usr/bin/env python3
"""Fetch BEN2's published checkpoint into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/ben2.py

**The download hash comes from Hugging Face's API, not from a README.** The blob's sha256 is read
from ``/api/models/<repo>/tree/main`` at fetch time, so a mismatch means the bytes changed under
us rather than that someone transcribed a digest wrongly.

**Three files are published and mozo takes the smallest.** The repository serves
``model.safetensors`` (380.6 MB), ``BEN2_Base.pth`` (1.13 GB) and ``BEN2_Base.onnx`` (223 MB).

* ``BEN2_Base.pth`` is what ``inference.py`` documents, and it is **not a weights file**. It is a
  training checkpoint at epoch 5: alongside its ``model_state_dict`` it carries 753.1 MB of Adam
  moments over 511 parameters, a gradient scaler, the loss, the learning rate and six validation
  metrics. Publishing it would ship three quarters of a gigabyte of an optimiser nobody will
  resume.
* ``model.safetensors`` is the same tensors and nothing else. The two were compared key by key
  before this script was written -- 535 entries each, no key in one and not the other, no shape
  or dtype mismatch, ``torch.equal`` on all 535 -- and ``PROVENANCE.md`` records that.
* ``BEN2_Base.onnx`` is a **float16** graph exported from the CUDA autocast path, and its own
  runner script feeds it unnormalised input. It cannot agree with the fp32 model and mozo does
  not republish it. ``PROVENANCE.md`` has the measurements.

So this reads the safetensors and writes an ordinary ``.pth``, exactly as ``tools/fetch/vitpose.py``
does and for the same reason: the version risk is taken here, once, on a machine we control.
``safetensors`` is a dependency of *this script* and not of mozo; nothing in ``mozo/`` imports it.

**One variant, because upstream publishes one checkpoint.** A commercial model exists behind a
sales address and is not published; BEN1 is superseded by the same authors and reports its licence
inconsistently. Both are named here so that adding one is a decision rather than a discovery.

Licensing is uniform and stated twice by the copyright holder: MIT in the repository's ``LICENSE``
(© 2025 Prama LLC) and ``license:mit`` on the ungated model card. The LICENSE placed beside the
weights is upstream's own file rather than a canonical copy, because here there is a licensor's
copy to prefer.

The output is ``weights/ben2/base/<revision>/torch-fp32.pth`` and a NOTICE naming the authors and
source. The LICENSE beside them is not written by this script: a licence is part of what is
published, so it lives in the weights tree like any other artifact.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import download_verified, require_licence, variant_parser  # noqa: E402

_HF = "https://huggingface.co"

#: The revision this family was published under.
REVISION = "2026-09-01"

#: Upstream's own MIT text, pinned to the commit ``PROVENANCE.md`` records.
LICENCE_SOURCE = (
    "https://raw.githubusercontent.com/PramaLLC/BEN2/"
    "2c99a5da477b5523585bfa5c893888a6e818a8f6/LICENSE")

#: mozo's variant name -> the Hugging Face repository.
REPOSITORIES: dict[str, str] = {"base": "PramaLLC/BEN2"}

#: The weights revision pinned in PROVENANCE.md. Pinned rather than ``main`` so that a re-fetch
#: a year from now produces the bytes this family's parity was measured against.
WEIGHTS_REVISION = "e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f"


def hub_json(url: str, timeout: int = 60) -> dict:
    """Read one JSON document from the Hub."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def blob_sha256(repo: str, filename: str, revision: str) -> str:
    """The sha256 Hugging Face records for *filename* in *repo*.

    Read from the tree endpoint rather than ``?blobs=true``: the latter omits the ``lfs`` block
    for this repository, and a listing that silently carries no hash is how an unverified
    download gets published.
    """
    for entry in hub_json(f"{_HF}/api/models/{repo}/tree/{revision}?recursive=true"):
        if entry.get("path") == filename and entry.get("lfs"):
            return entry["lfs"]["oid"]
    raise SystemExit(f"{repo} records no content hash for {filename}; refusing to publish it")


def _notice(repo: str, filename: str, sha256: str) -> str:
    """Attribution that travels with the weights."""
    return (
        "BEN2 -- Background Erase Network, base\n"
        "background removal: a photograph in, a per-pixel alpha matte out\n\n"
        "Copyright (c) 2025 Prama LLC.\n"
        "Maxwell Meyer, Jack Spruyt.\n\n"
        f"Source:  {_HF}/{repo}\n"
        "Project: https://github.com/PramaLLC/BEN2\n"
        "Paper:   BEN: Using Confidence-Guided Matting for Dichotomous Image Segmentation,\n"
        "         arXiv:2501.06230\n"
        "Licence: MIT, stated by the copyright holder in the repository's LICENSE file and\n"
        "         again as license:mit on the ungated model card. The full text is in the\n"
        "         LICENSE file beside this one.\n\n"
        "BEN2 was trained on DIS5K together with a proprietary dataset. The MIT grant above is\n"
        "the publisher's statement about these weights; the terms of the data behind them were\n"
        "the publisher's obligation, discharged before publication.\n\n"
        "This is not the byte stream that repository serves: tools/fetch/ben2.py verifies the\n"
        "safetensors against the sha256 the Hub records, then writes the same tensors back out\n"
        "as an ordinary checkpoint. No tensor is altered, renamed, cast or dropped. The source\n"
        "file is:\n"
        f"           {filename}  sha256 {sha256}\n\n"
        "Upstream also publishes BEN2_Base.pth, which is a training checkpoint carrying 753 MB\n"
        "of optimiser state around the same 535 tensors, and BEN2_Base.onnx, which is a float16\n"
        "export of the CUDA path. Neither is republished here.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download the checkpoint, verify it, repack it, and place it with its notice."""
    repo = REPOSITORIES[variant]
    target = weights_dir / "ben2" / variant / revision / "torch-fp32.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    name = "model.safetensors"
    part = target.with_name(name)
    sha256 = download_verified(
        f"{_HF}/{repo}/resolve/{WEIGHTS_REVISION}/{name}", part,
        blob_sha256(repo, name, WEIGHTS_REVISION), label=variant, width=6, detail="MIT")

    if not target.is_file():
        from safetensors.torch import load_file

        print(f"  {variant:6s} repacking safetensors -> state dict")
        torch.save(load_file(part), target)
    part.unlink(missing_ok=True)

    (target.parent / "NOTICE").write_text(_notice(repo, name, sha256))
    require_licence(target.parent, "MIT", LICENCE_SOURCE)


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()

    wanted = args.variants or list(REPOSITORIES)
    unknown = [v for v in wanted if v not in REPOSITORIES]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(REPOSITORIES)}")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} checkpoint in {args.weights_dir}, MIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
