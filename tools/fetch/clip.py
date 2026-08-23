#!/usr/bin/env python3
"""Fetch CLIP's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/clip.py                  # every ViT variant
    python tools/fetch/clip.py base

**The download hash comes from upstream.** OpenAI serves these at content-addressed URLs -- the
path segment before the filename *is* the file's sha256, which is what ``clip.load`` verifies
against. So the check below is upstream's own, not a digest transcribed from a README.

**They are TorchScript archives, and mozo republishes plain tensors.** ``torch.jit.load`` recovers
the state dict; this writes it back out as an ordinary ``.pth``. That is a repack --
``tools/fetch/easyocr.py`` does one too -- and it is worth doing rather than avoiding: a scripted
archive is a serialised graph that a future torch may refuse, so the version risk is taken here,
once, on a machine we control, instead of on a user's machine years from now. What mozo publishes
loads on any torch that can read a state dict.

**Cast to fp32.** The archives are mixed precision: ``convert_weights`` halves the convolutions,
linears and attention while leaving LayerNorms and embeddings in fp32, and ``clip.load`` restores
everything to fp32 only when it lands on the CPU. mozo publishes one dtype rather than a mixture,
so the artifact key is ``torch-fp32`` and means it. That roughly doubles each file against the
archive, which is the price of an artifact whose name is true.

**Five ResNet variants are not carried.** RN50, RN101, RN50x4, RN50x16 and RN50x64 use a modified
ResNet with attention pooling instead of a Vision Transformer -- a second image tower that
``clip_deploy`` does not build. They are named here so adding them is a one-line change rather
than a research question.

The output is ``weights/clip/<variant>/<revision>/torch-fp32.pth`` and a NOTICE. The LICENSE
beside them is not written by this script: a licence is part of what is published, so it lives in
the weights tree like any other artifact.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from common import download_verified, require_licence, variant_parser  # noqa: E402

_BASE = "https://openaipublic.azureedge.net/clip/models"

#: The revision this family was published under.
REVISION = "2026-08-23"

#: Where MIT's canonical text lives. OpenAI ships a LICENSE in the repository but not beside the
#: checkpoints, so the text is placed once per revision.
LICENCE_SOURCE = "https://raw.githubusercontent.com/openai/CLIP/main/LICENSE"


@dataclass(frozen=True)
class Checkpoint:
    """One published checkpoint: where it lives, what it hashes to, and what it is."""

    variant: str
    upstream: str
    sha256: str
    filename: str
    #: Parameters, in millions, across both towers.
    params: str

    @property
    def url(self) -> str:
        # The hash is the path segment. That is upstream's own integrity check, not ours.
        return f"{_BASE}/{self.sha256}/{self.filename}"


#: Variant -> checkpoint. Keys are mozo's names; OpenAI identifies the same models by backbone.
CHECKPOINTS: dict[str, Checkpoint] = {
    entry.variant: entry
    for entry in (
        Checkpoint("base", "ViT-B/32",
                   "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
                   "ViT-B-32.pt", "151"),
        Checkpoint("base-16", "ViT-B/16",
                   "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
                   "ViT-B-16.pt", "150"),
        Checkpoint("large", "ViT-L/14",
                   "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
                   "ViT-L-14.pt", "428"),
        Checkpoint("large-336", "ViT-L/14@336px",
                   "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
                   "ViT-L-14-336px.pt", "428"),
    )
}


def _notice(entry: Checkpoint) -> str:
    """Attribution that travels with the weights."""
    return (
        f"CLIP -- {entry.variant} ({entry.upstream})\n"
        f"contrastive image-text pretraining, {entry.params}M parameters across two towers\n\n"
        "Copyright (c) 2021 OpenAI\n"
        "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,\n"
        "Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,\n"
        "Gretchen Krueger, Ilya Sutskever.\n\n"
        f"Source:  {entry.url}\n"
        "Project: https://github.com/openai/CLIP\n"
        "Paper:   Learning Transferable Visual Models From Natural Language Supervision\n"
        "         (ICML 2021), arXiv:2103.00020\n"
        "Licence: MIT (full text in the LICENSE file beside this one)\n\n"
        "The repository's LICENSE is MIT and no separate weights licence is published, so the\n"
        "checkpoints are covered by it. That is an inference from silence, recorded as one.\n\n"
        "This file is not the byte stream OpenAI serves. Upstream publishes a TorchScript\n"
        "archive; tools/fetch/clip.py verifies it against the sha256 in OpenAI's own URL, then\n"
        "recovers the state dict with torch.jit.load and writes the tensors back out as an\n"
        "ordinary checkpoint, cast to fp32. No tensor is otherwise altered. The archive's sha256\n"
        f"is {entry.sha256}.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one checkpoint, verify it, repack it, and place it with its licence and notice."""
    entry = CHECKPOINTS[variant]
    target = weights_dir / "clip" / variant / revision / "torch-fp32.pth"
    archive = target.with_name("archive.pt")

    download_verified(entry.url, archive, entry.sha256, label=variant, width=10, detail="MIT")

    if not target.is_file():
        print(f"  {variant:10s} repacking TorchScript -> state dict, fp32")
        state = torch.jit.load(archive, map_location="cpu").eval().state_dict()
        torch.save({key: value.float() for key, value in state.items()}, target)
    archive.unlink(missing_ok=True)

    (target.parent / "NOTICE").write_text(_notice(entry))
    require_licence(target.parent, "MIT", LICENCE_SOURCE)


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()

    wanted = args.variants or list(CHECKPOINTS)
    unknown = [v for v in wanted if v not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(CHECKPOINTS)}")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}, all MIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
