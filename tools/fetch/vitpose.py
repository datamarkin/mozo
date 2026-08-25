#!/usr/bin/env python3
"""Fetch ViTPose++'s published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/vitpose.py            # every variant
    python tools/fetch/vitpose.py small base

**The download hash comes from Hugging Face's API, not from a README.** Each blob's sha256 is read
from ``/api/models/<repo>?blobs=true`` at fetch time, so a mismatch means the bytes changed under
us rather than that someone transcribed a digest wrongly.

**Safetensors only, so mozo repacks.** These repositories publish no ``pytorch_model.bin``, so
OWLv2's trick of placing the ``.bin`` unchanged is not available. This reads the safetensors and
writes an ordinary ``.pth``, exactly as ``tools/fetch/siglip2.py`` does and for the same reason:
the version risk is taken here, once, on a machine we control. ``safetensors`` is a dependency of
*this script* and not of mozo; nothing in ``mozo/`` imports it.

**Seven checkpoints are published; mozo carries four.** The other three -- ``vitpose-base``,
``vitpose-base-simple`` and ``vitpose-base-coco-aic-mpii`` -- are the original ViTPose, which
ViTPose++ beats at every size, and the smallest of them is 344 MB against ``plus-small``'s 133.
They would also need the two branches ``vitpose_deploy`` deliberately does not build. They are
named below so that adding one is a decision rather than a discovery.

Licensing is uniform: Apache-2.0 on the code and on every checkpoint. The repositories ship no
LICENSE file of their own, so the text placed beside the weights is Apache's canonical one -- there
is no licensor's copy to prefer over it.

The output is ``weights/vitpose/<variant>/<revision>/torch-fp32.pth`` and a NOTICE naming the
authors and source. The LICENSE beside them is not written by this script: a licence is part of
what is published, so it lives in the weights tree like any other artifact. If one is missing this
script says so and prints the command to fetch it.
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
REVISION = "2026-08-25"

#: Where Apache-2.0's canonical text lives. The University of Sydney ships no LICENSE file in
#: these repositories -- the terms are stated on the model card -- so the text is placed once per
#: revision rather than copied from an upstream file that does not exist.
LICENCE_SOURCE = "https://www.apache.org/licenses/LICENSE-2.0.txt"

#: mozo's variant name -> the Hugging Face repository. Every one is ViTPose++; mozo names them by
#: size, as it does in every other family, and ``PROVENANCE.md`` records which paper they are from.
REPOSITORIES: dict[str, str] = {
    "small": "usyd-community/vitpose-plus-small",
    "base": "usyd-community/vitpose-plus-base",
    "large": "usyd-community/vitpose-plus-large",
    "huge": "usyd-community/vitpose-plus-huge",
}


def hub_json(url: str, timeout: int = 60) -> dict:
    """Read one JSON document from the Hub."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def blob_sha256(repo: str, filename: str) -> str:
    """The sha256 Hugging Face records for *filename* in *repo*.

    Only LFS-tracked files carry a content hash; the small ones carry a git blob id, which is a
    hash of different bytes and useless for verifying a download. The safetensors are LFS, so a
    file that is missing here raises rather than being waved through unverified.
    """
    listing = hub_json(f"{_HF}/api/models/{repo}?blobs=true")
    for entry in listing["siblings"]:
        if entry["rfilename"] == filename and entry.get("lfs"):
            return entry["lfs"]["sha256"]
    raise SystemExit(f"{repo} records no content hash for {filename}; refusing to publish it")


def _notice(variant: str, repo: str, filename: str, sha256: str) -> str:
    """Attribution that travels with the weights, as Apache-2.0 asks."""
    return (
        f"ViTPose++ -- {variant}\n"
        "top-down human pose estimation: a person box in, seventeen joints out\n\n"
        "Copyright 2022 The University of Sydney.\n"
        "Yufei Xu, Jing Zhang, Qiming Zhang, Dacheng Tao.\n\n"
        f"Source:  {_HF}/{repo}\n"
        "Project: https://github.com/ViTAE-Transformer/ViTPose\n"
        "Paper:   ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation\n"
        "         (NeurIPS 2022), arXiv:2204.12484\n"
        "         ViTPose++: Vision Transformer for Generic Body Pose Estimation\n"
        "         (TPAMI 2023), arXiv:2212.04246\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one)\n\n"
        "The authors' own release is built on mmpose. This file derives from the PyTorch\n"
        "conversion published on Hugging Face, under the same terms. It is not the byte stream\n"
        "that repository serves: tools/fetch/vitpose.py verifies the safetensors against the\n"
        "sha256 the Hub records, then writes the same tensors back out as an ordinary\n"
        "checkpoint. No tensor is altered, renamed, cast or dropped. The source file is:\n"
        f"           {filename}  sha256 {sha256}\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, repack it, and place it with its notice."""
    repo = REPOSITORIES[variant]
    target = weights_dir / "vitpose" / variant / revision / "torch-fp32.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    name = "model.safetensors"
    part = target.with_name(name)
    sha256 = download_verified(f"{_HF}/{repo}/resolve/main/{name}", part, blob_sha256(repo, name),
                               label=variant, width=6, detail="Apache-2.0")

    if not target.is_file():
        from safetensors.torch import load_file

        print(f"  {variant:6s} repacking safetensors -> state dict")
        torch.save(load_file(part), target)
    part.unlink(missing_ok=True)

    (target.parent / "NOTICE").write_text(_notice(variant, repo, name, sha256))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()

    wanted = args.variants or list(REPOSITORIES)
    unknown = [v for v in wanted if v not in REPOSITORIES]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(REPOSITORIES)}")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}, all Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
