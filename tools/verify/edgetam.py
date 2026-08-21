#!/usr/bin/env python3
"""Check that mozo's EdgeTAM returns exactly what the published model does.

Two paths reach the same weights. One is ``mozo/vendors/edgetam_deploy`` -- build a ``Segmenter``,
hand it an image and a prompt, read the masks off it. The other is ``facebookresearch/EdgeTAM``
itself, driven through its own ``SAM2ImagePredictor``. Between them sit a trunk written out from
timm, a preprocessing rewrite that removes torchvision, a config that replaces Hydra, and a prompt
encoder rewritten to be traceable -- and any of those could quietly change a number.

The comparison is exact. Not "close": a tolerance would hide precisely the drift this exists to
catch, and every divergence found while building this package was smaller than any tolerance
anyone would have picked.

Unlike SAM 3's gate this needs no recorded fingerprints, because EdgeTAM is Apache-2.0 in both its
code and its weights -- so the reference can simply be run. It does need a checkout, which is why
this lives in ``tools`` and not in the test suite.

**One thing about the reference is modified, and it is the reason this docstring is long.**
Upstream pins the attention backend: ``sam2.modeling.sam.transformer.sdp_kernel_context`` wraps
every ``scaled_dot_product_attention`` in ``torch.backends.cuda.sdp_kernel(...)``, with flags that
``get_sdpa_settings()`` derives from the local CUDA capability at import time. On a machine
without a modern CUDA GPU that pins the math kernel; on an A100 it selects flash. Upstream's own
logits therefore differ between two machines running the same weights on the same image.

mozo does not carry that pin -- see the note in ``edgetam_deploy/sam/transformer.py`` for why --
so this gate neutralises it on the reference side, and both paths use whichever backend torch
picks by default. What is being compared is then the arithmetic of the two implementations rather
than torch's kernel dispatch, which is the thing this can actually be responsible for.

That substitution is worth stating plainly because patching the reference is how a gate quietly
stops testing anything. It is bounded: it replaces one context manager with a no-op and touches
nothing else. ``--pin-attention`` runs without the substitution, which is how the cost of the
choice was measured -- 2e-07 per attention layer, 9.2e-05 by the decoder's output, every other
stage still bit-identical.

Run from the repository root::

    python tools/verify/edgetam.py --upstream ~/Projects/EdgeTAM
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mozo.image import load_image  # noqa: E402
from mozo.vendors.edgetam_deploy import Segmenter  # noqa: E402
from mozo.vendors.edgetam_deploy.predictor import LOGIT_LIMIT  # noqa: E402
from mozo.vendors.edgetam_deploy.image import preprocess  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from conftest import FIXTURE  # noqa: E402

PROVENANCE = ROOT / "mozo" / "vendors" / "edgetam_deploy" / "PROVENANCE.md"

#: Prompts, as fractions of the image so they follow the fixture rather than its size.
#:
#: This is EdgeTAM's own copy, and ``tools/verify/sam3.py`` has its own. Sharing them looks like
#: the obvious cleanup and is a trap: ``sam3_digests.json`` records 26 click stages keyed by these
#: prompt names, so adding a case here for one family would invalidate the other's recorded
#: fingerprints -- and re-recording those needs a checkout of Meta's SAM-licensed ``sam3``, which
#: CI does not have. A gate for one family must not be breakable by an edit made for another.
#: ``_detection.py`` is shared for the opposite reason: four families, one upstream, no
#: recorded state.
#:
#: The last three exist for reasons a simpler set would miss. A lone negative point is the only
#: prompt with nothing to include. A box with a negative point is the only one that exercises the
#: ordering rule -- corners first, then clicks. And ``multimask_output=False`` is the only prompt
#: that reaches the decoder's stability fallback, a setting that carries no weights, is not in
#: upstream's config file, and which a strict load therefore cannot check.
CLICKS: tuple[tuple[str, list | None, list | None, list | None, bool], ...] = (
    ("one positive", [[0.62, 0.55]], [1], None, True),
    ("positive and negative", [[0.62, 0.55], [0.58, 0.72]], [1, 0], None, True),
    ("two positives", [[0.62, 0.55], [0.65, 0.60]], [1, 1], None, True),
    ("box", None, None, [0.55, 0.42, 0.72, 0.78], True),
    ("box and negative", [[0.58, 0.72]], [0], [0.55, 0.42, 0.72, 0.78], True),
    ("lone negative", [[0.62, 0.55]], [0], None, True),
    ("single mask output", [[0.62, 0.55]], [1], None, False),
)


def click_prompt(spec, shape: tuple[int, int]) -> dict:
    """Turn one :data:`CLICKS` entry into pixel coordinates for ``shape``."""
    _, points, labels, box, multimask = spec
    height, width = shape
    scale = np.array([width, height], dtype=np.float32)
    return {
        "points": None if points is None else np.asarray(points, dtype=np.float32) * scale,
        "labels": None if labels is None else np.asarray(labels),
        "boxes": None if box is None
        else (np.asarray(box, dtype=np.float32).reshape(2, 2) * scale).reshape(4),
        "multimask_output": multimask,
    }



def pinned_commit() -> str:
    """The upstream commit ``PROVENANCE.md`` records this package as extracted from."""
    match = re.search(r"EdgeTAM.*?\|\s*`([0-9a-f]{40})`", PROVENANCE.read_text(), re.S)
    if match is None:
        raise SystemExit(f"no upstream commit recorded in {PROVENANCE}")
    return match.group(1)


def published() -> Path:
    """Locate the EdgeTAM checkpoint the way mozo itself would."""
    from mozo.weights import manifest

    model = manifest()["models"].get("edgetam/edgetam")
    if model is None:
        raise SystemExit("the manifest publishes no edgetam/edgetam.")
    revision = model["latest"]
    artifact = model["revisions"][revision]["artifacts"]["torch-fp32"]
    path = ROOT / "weights" / artifact["path"]
    if not path.is_file():
        raise SystemExit(
            f"no EdgeTAM weights at {path}; see mozo/vendors/edgetam_deploy/PROVENANCE.md."
        )
    return path


def reference(upstream: Path, checkpoint: Path, pin_attention: bool):
    """Build upstream's own image predictor, with the attention pin removed by default."""
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)  # its Hydra config is resolved relative to the package

    if not pin_attention:
        # See the module docstring. Replaced before ``build_sam2`` imports the transformer, so
        # the module-level ``get_sdpa_settings()`` probe is the only thing that still runs.
        import sam2.modeling.sam.transformer as transformer

        transformer.sdp_kernel_context = lambda dropout_p: contextlib.nullcontext()

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return SAM2ImagePredictor(
        build_sam2("configs/edgetam.yaml", str(checkpoint), device="cpu", mode="eval")
    )


class Comparison:
    """Accumulates stage comparisons and reports how many disagreed."""

    def __init__(self) -> None:
        self.failed: list[str] = []
        self.checked = 0

    def __call__(self, name: str, ours: torch.Tensor, theirs: torch.Tensor) -> None:
        ours = torch.as_tensor(np.asarray(ours)).float()
        theirs = torch.as_tensor(np.asarray(theirs)).float()
        self.checked += 1
        if ours.shape != theirs.shape:
            print(f"  FAIL  {name:<40} shape {tuple(ours.shape)} != {tuple(theirs.shape)}")
            self.failed.append(name)
            return
        if torch.equal(ours, theirs):
            print(f"  ok    {name:<40} identical")
            return
        print(f"  FAIL  {name:<40} max|d| {(ours - theirs).abs().max().item():.3e}")
        self.failed.append(name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--upstream", type=Path, required=True,
                        help="checkout of facebookresearch/EdgeTAM")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a checkpoint to check instead of the published one")
    parser.add_argument("--pin-attention", action="store_true",
                        help="leave upstream's SDPA backend pin in place; see the module docstring")
    parser.add_argument("--allow-commit-drift", action="store_true",
                        help="run against an upstream other than the extracted commit")
    arguments = parser.parse_args()

    if not (arguments.upstream / "sam2" / "build_sam.py").is_file():
        raise SystemExit(
            f"no EdgeTAM checkout at {arguments.upstream}. Clone it:\n"
            f"  git clone https://github.com/facebookresearch/EdgeTAM {arguments.upstream}"
        )
    head = subprocess.run(["git", "-C", str(arguments.upstream), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    pinned = pinned_commit()
    if head != pinned and not arguments.allow_commit_drift:
        raise SystemExit(
            f"upstream is at {head[:12]}, PROVENANCE.md records {pinned[:12]}.\n"
            "A baseline other than the extraction will disagree with it and sound right doing "
            "so.\nCheck it out, or pass --allow-commit-drift."
        )

    checkpoint = (arguments.checkpoint or published()).resolve()
    image = load_image(str(FIXTURE))
    shape = image.shape[:2]
    print(f"upstream   {arguments.upstream} @ {head[:12]}")
    print(f"weights    {checkpoint}")
    print(f"image      {FIXTURE}  {shape[1]}x{shape[0]}")
    print(f"attention  {'upstream pin left in place' if arguments.pin_attention else 'torch default on both sides'}\n")

    torch.set_grad_enabled(False)
    ours = Segmenter(checkpoint)
    theirs = reference(arguments.upstream, checkpoint, arguments.pin_attention)

    same = Comparison()

    print("preprocessing:")
    theirs._orig_hw = [shape]
    same("preprocess", preprocess(image, ours.image_size), theirs._transforms(image)[None])

    print("\nimage encoder:")
    theirs.set_image(image)
    features = ours.encode(image)
    same("image_embed", features["image_embed"], theirs._features["image_embed"])
    for level, tensor in enumerate(features["high_res_feats"]):
        same(f"high_res_feats[{level}]", tensor, theirs._features["high_res_feats"][level])

    print("\nprompted decoder:")
    for spec in CLICKS:
        prompt = click_prompt(spec, shape)
        got = ours.predict(image, **prompt)
        _, iou, low_res = theirs.predict(
            point_coords=prompt["points"], point_labels=prompt["labels"], box=prompt["boxes"],
            multimask_output=prompt["multimask_output"], return_logits=True, normalize_coords=True,
        )
        # Upstream returns one prompt's results unbatched; mozo keeps the batch dimension, so the
        # axis is put back rather than compared away -- a shape is part of what must agree.
        # Its logits are unclamped, and mozo's are bounded before being handed back for
        # refinement, so the same bound is applied here rather than the bound being skipped.
        same(f"{spec[0]}: logits", got.logits, torch.from_numpy(low_res)[None].clamp(-LOGIT_LIMIT, LOGIT_LIMIT))
        same(f"{spec[0]}: iou", got.scores, torch.from_numpy(iou)[None])

    # A held image must answer a repeated prompt identically. A cache that quietly returns
    # something else is worse than no cache, and this is the only place it would show.
    print("\nencoder cache:")
    first = ours.predict(image, **click_prompt(CLICKS[0], shape))
    again = ours.predict(image, **click_prompt(CLICKS[0], shape))
    same("repeated prompt: logits", again.logits, first.logits)

    print()
    if same.failed:
        print(f"{len(same.failed)} of {same.checked} stages differ from the published model.")
        return 1
    print(f"  {same.checked} stages, every one identical to the published model.")
    print("\nEdgeTAM: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
