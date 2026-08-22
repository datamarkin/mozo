#!/usr/bin/env python3
"""Fetch Grounding DINO's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/grounding_dino.py              # both
    python tools/fetch/grounding_dino.py tiny

**Upstream publishes exactly two checkpoints**, and both are carried. `tiny` is 82% of the
project's own release downloads; `base` is the other 18% and is 8.3 box AP better on COCO
zero-shot, which is too wide a gap to call it optional.

The files are placed unchanged and renamed to mozo's artifact key. No repacking, no pruning, no
mozo-format checkpoint -- what is published is exactly what IDEA Research published, including the
fine-tuned BERT tower it carries under ``bert.*``. That tower is why these files are large, and
why nothing needs downloading from Hugging Face to run them.

**On the licence.** IDEA Research's repository is Apache-2.0 and its README says nothing at all
about the checkpoints. The claim that the weights are Apache-2.0 is made in the two Hugging Face
repositories the README's own checkpoint table links to -- the first author's mirror of these
exact files, and the organisation's account. That is enough to publish, and it is why the NOTICE
written here names *where* the claim is made: a reader who checks only the GitHub LICENSE finds a
code licence and no mention of a checkpoint.

The output is ``weights/grounding_dino/<variant>/<revision>/torch-fp32.pth`` and a NOTICE. The
LICENSE beside them is not written by this script: a licence is part of what is published, so it
lives in the weights tree like any other artifact. If one is missing this script says so and
prints the command to fetch it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from common import download_verified, require_licence, variant_parser  # noqa: E402

_RELEASES = "https://github.com/IDEA-Research/GroundingDINO/releases/download"

#: The revision this family was published under.
REVISION = "2026-08-23"

#: Where Apache-2.0's canonical text lives. IDEA Research ships a LICENSE in the repository but
#: not beside the release assets, so the text is placed once per revision.
LICENCE_SOURCE = "https://www.apache.org/licenses/LICENSE-2.0.txt"

#: Where the authors state the weights' terms. Not the GitHub repository, which covers the code
#: and is silent on the checkpoints.
LICENCE_CLAIM = (
    "https://huggingface.co/ShilongLiu/GroundingDINO",
    "https://huggingface.co/IDEA-Research/grounding-dino-tiny",
)


@dataclass(frozen=True)
class Checkpoint:
    """One published checkpoint: where it lives, what it should hash to, and what it is."""

    variant: str
    tag: str
    filename: str
    sha256: str
    backbone: str
    #: Upstream's own description of the training data, for the NOTICE.
    trained_on: str
    #: Box AP on COCO, zero-shot, as the project's README reports it.
    coco_ap: str

    @property
    def url(self) -> str:
        return f"{_RELEASES}/{self.tag}/{self.filename}"


#: Variant -> checkpoint. Keys are mozo's variant names; upstream identifies the same models by
#: backbone and training set (`GroundingDINO-T`, `GroundingDINO-B`).
CHECKPOINTS: dict[str, Checkpoint] = {
    entry.variant: entry
    for entry in (
        Checkpoint(
            "tiny", "v0.1.0-alpha", "groundingdino_swint_ogc.pth",
            "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
            "Swin-T", "Objects365, GoldG and Cap4M (OGC)", "48.4",
        ),
        Checkpoint(
            "base", "v0.1.0-alpha2", "groundingdino_swinb_cogcoor.pth",
            "46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7",
            "Swin-B", "COCO, Objects365, GoldG, Cap4M, OpenImage, ODinW-35 and RefCOCO", "56.7",
        ),
    )
}


def _notice(entry: Checkpoint) -> str:
    """Attribution that travels with the weights, as Apache-2.0 asks."""
    claims = "\n".join(f"         {url}" for url in LICENCE_CLAIM)
    return (
        f"Grounding DINO -- {entry.variant} ({entry.backbone})\n"
        f"open-vocabulary object detection, trained on {entry.trained_on}\n"
        f"{entry.coco_ap} box AP on COCO, zero-shot, as reported by the authors\n\n"
        "Copyright 2023 - present, IDEA Research.\n"
        "Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang,\n"
        "Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, Lei Zhang.\n\n"
        f"Source:  {entry.url}\n"
        "Project: https://github.com/IDEA-Research/GroundingDINO\n"
        "Paper:   Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set\n"
        "         Object Detection (ECCV 2024), arXiv:2303.05499\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one)\n\n"
        "Where that licence is claimed. The GitHub repository's LICENSE covers the code and its\n"
        "README states no terms for the checkpoints. The authors publish these same files under\n"
        "an explicit apache-2.0 tag here:\n"
        f"{claims}\n\n"
        "The checkpoint carries a fine-tuned BERT-base text encoder under its `bert.*` keys. The\n"
        "vocabulary that tokenizes prompts for it is Google's `bert-base-uncased` vocab.txt,\n"
        "Apache-2.0, and ships inside mozo rather than being downloaded.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, and place it with its licence and notice."""
    entry = CHECKPOINTS[variant]
    target = weights_dir / "grounding_dino" / variant / revision / "torch-fp32.pth"
    download_verified(entry.url, target, entry.sha256, label=variant, width=6,
                      detail="Apache-2.0")

    (target.parent / "NOTICE").write_text(_notice(entry))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()

    wanted = args.variants or list(CHECKPOINTS)
    unknown = [v for v in wanted if v not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(CHECKPOINTS)}")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}, all Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
