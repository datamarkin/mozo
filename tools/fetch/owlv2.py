#!/usr/bin/env python3
"""Fetch OWLv2's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/owlv2.py                       # everything
    python tools/fetch/owlv2.py base-ensemble large-ensemble

Each file is checked against the SHA-256 that Hugging Face records for the blob, read from its API
rather than transcribed from a README, so a mismatch means the bytes changed under us.

**``pytorch_model.bin``, not ``model.safetensors``.** Google publishes both, and they hold the same
tensors. mozo consumes the ``.bin`` because ``torch.load`` reads it with nothing installed that is
not already required -- taking the safetensors would put a new dependency on the *runtime* to save
nothing at publish time. The bytes are placed unchanged and renamed to mozo's artifact key; there
is no repacking step and no mozo-format checkpoint.

**Google publishes six; mozo publishes four.** The two ``-finetuned`` checkpoints have 1,017 and
684 downloads against 1.34M for ``base-patch16-ensemble``, and nothing in ``owlv2_deploy`` would
differ if they were here. They are named below so that adding one is a one-line change rather than
a research question.

Licensing is uniform for once: Apache-2.0 on the code and on every checkpoint, which is the reason
this family is in mozo at all. Its only text-prompted sibling, SAM 3, is not.

The output is ``weights/owlv2/<variant>/<revision>/torch-fp32.pth`` and a NOTICE naming the
authors and source. The LICENSE beside them is not written by this script: a licence is part of
what is published, so it lives in the weights tree like any other artifact. If one is missing this
script says so and prints the command to fetch it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from common import download_verified, require_licence, variant_parser  # noqa: E402

_HF = "https://huggingface.co/google"

#: The revision this family was published under.
REVISION = "2026-08-21"

#: Where Apache-2.0's canonical text lives. Google ships no LICENSE file in these repositories,
#: so the text is placed once per revision rather than copied from upstream.
LICENCE_SOURCE = "https://www.apache.org/licenses/LICENSE-2.0.txt"


@dataclass(frozen=True)
class Checkpoint:
    """One published checkpoint: where it lives, what it should hash to, and what it is."""

    variant: str
    repo: str
    sha256: str
    #: Upstream's own description of the training recipe, for the NOTICE.
    trained_by: str

    @property
    def url(self) -> str:
        return f"{_HF}/{self.repo}/resolve/main/pytorch_model.bin"


_ST = "self-training on Web image-text pairs (ST)"
_ENSEMBLE = "an ensemble of the self-trained and human-annotation fine-tuned checkpoints (ST/FT)"

#: Variant -> checkpoint. Keys are mozo's variant names; Google identifies the same models by
#: backbone and suffix. ``-finetuned`` (ST+FT) is published upstream and not carried here.
CHECKPOINTS: dict[str, Checkpoint] = {
    entry.variant: entry
    for entry in (
        Checkpoint("base-ensemble", "owlv2-base-patch16-ensemble",
                   "69feda8b53b1c9e2a85ae756bf58c120c3c1b4b4a4d97d4876578c1809a63d76", _ENSEMBLE),
        Checkpoint("base", "owlv2-base-patch16",
                   "56c7c1adff4a422d9ceba3fc744d8595a29a05ac4714497a9472d1ffc2e7332f", _ST),
        Checkpoint("large-ensemble", "owlv2-large-patch14-ensemble",
                   "2934e1f32c68b49f62e9b7a415c22080a8bf197c50c6f4408f4a60e21e0be252", _ENSEMBLE),
        Checkpoint("large", "owlv2-large-patch14",
                   "3f442c5e2875819ae6173b4f30f399e9c9ae51ed731184dfbed0f09eb85ce23f", _ST),
    )
}


def _notice(entry: Checkpoint) -> str:
    """Attribution that travels with the weights, as Apache-2.0 asks."""
    return (
        f"OWLv2 -- {entry.variant}\n"
        f"open-vocabulary object detection, trained by {entry.trained_by}\n\n"
        "Copyright 2023 The Google Research Authors.\n"
        "Matthias Minderer, Alexey Gritsenko, Neil Houlsby.\n\n"
        f"Source:  {entry.url}\n"
        "Project: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit\n"
        "Paper:   Scaling Open-Vocabulary Object Detection (NeurIPS 2023), arXiv:2306.09683\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one)\n\n"
        "Google's own release is JAX/Flax. This file is the PyTorch conversion Google publishes\n"
        "beside it on Hugging Face, unmodified, under the same terms.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, and place it with its licence and notice."""
    entry = CHECKPOINTS[variant]
    target = weights_dir / "owlv2" / variant / revision / "torch-fp32.pth"
    download_verified(entry.url, target, entry.sha256, label=variant, width=15,
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
