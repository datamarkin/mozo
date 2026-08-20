#!/usr/bin/env python3
"""Check that mozo's SAM 3 still produces exactly the numbers it was built to produce.

Every other family's gate compares two live paths. This one cannot: the only thing that can say
what SAM 3 *should* return is Meta's own implementation, which ships under the SAM License and is
not in this repository. So the comparison is split in two.

``sam3_reference.py`` runs both implementations side by side, refuses to write unless the stages
they *both* produce agree bit for bit, and then records a fingerprint of every stage mozo has.
That step needs a checkout of the published model and is run only when the model changes.

This module checks those fingerprints. It needs the weights and nothing else -- no reference, no
745 MB of stored tensors, no licensed code -- so anyone holding the checkpoint can run it, on any
machine, forever. A fingerprint is a hash of the numbers, not the numbers, so nothing here is SAM
Materials.

Two kinds of fingerprint are recorded, and they carry different weight. The ``concept.*`` and
``exemplars.*`` stages were checked against the published model when they were written, so a
failure there means mozo has diverged from Meta. The rest -- ``preprocess``, ``vision.*``,
``text.*``, ``tokenizer.ids``, ``concept.*.semantic`` and ``segmenter.*`` -- have no counterpart
in the reference's public surface and were recorded from mozo alone; they catch drift from what
mozo did on the day the reference agreed, which is not the same claim. ``sam3_reference.py``
prints which is which when it writes them.

The comparison is exact. Not "close": a tolerance would hide precisely the drift this exists to
catch, and eight real divergences from the reference implementation were each found at a
magnitude a tolerance would have swallowed -- one of them at 5.96e-08.

Run from the repository root::

    python tools/verify/sam3.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mozo.image import load_image  # noqa: E402
from mozo.vendors.sam3_deploy import Segmenter  # noqa: E402
from mozo.vendors.sam3_deploy.image import preprocess, preprocess_click  # noqa: E402
from mozo.weights import manifest  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
from conftest import FIXTURE  # noqa: E402

DIGESTS = Path(__file__).resolve().parent / "sam3_digests.json"
MODEL_ID = "sam3/sam3"

#: Chosen to exercise what goes wrong quietly: a common concept, a multi-word phrase, and one that
#: is absent -- the last is the only prompt that reaches the presence head's unclamped branch.
PROMPTS = ["person", "coffee mug", "a person holding a coffee mug", "cow"]

#: Exemplar boxes, normalised (cx, cy, w, h), with 1 meaning "an example of what I want".
#: Two boxes rather than one: with a single box a batch/sequence transpose is invisible.
EXEMPLARS = ([[0.45, 0.55, 0.20, 0.30], [0.15, 0.30, 0.12, 0.14]], [1, 0])

#: Click prompts, as fractions of the image so they follow the fixture rather than its size.
#: The last two exist for reasons a simpler set would miss: a lone negative point is the only
#: prompt with nothing to include, and ``multimask_output=False`` is the only one that reaches
#: the decoder's stability fallback -- a flag carrying no weights, which a strict load cannot
#: check and which 23 of 24 parity prompts failed to notice being wrong.
CLICKS: tuple[tuple[str, list, list, list | None, bool], ...] = (
    ("one positive", [[0.62, 0.55]], [1], None, True),
    ("positive and negative", [[0.62, 0.55], [0.58, 0.72]], [1, 0], None, True),
    ("two positives", [[0.62, 0.55], [0.65, 0.60]], [1, 1], None, True),
    ("box", None, None, [0.55, 0.42, 0.72, 0.78], True),
    ("box and negative", [[0.58, 0.72]], [0], [0.55, 0.42, 0.72, 0.78], True),
    ("lone negative", [[0.62, 0.55]], [0], None, True),
    ("single mask output", [[0.62, 0.55]], [1], None, False),
)


def click_prompt(spec, shape):
    """Turn one :data:`CLICKS` entry into pixel coordinates for ``shape``."""
    name, points, labels, box, multimask = spec
    height, width = shape
    scale = np.array([width, height], dtype=np.float32)
    return {
        "points": None if points is None else np.asarray(points, dtype=np.float32) * scale,
        "labels": None if labels is None else np.asarray(labels),
        "boxes": None if box is None
        else (np.asarray(box, dtype=np.float32).reshape(2, 2) * scale).reshape(4),
        "multimask_output": multimask,
    }


def digest(tensor: torch.Tensor) -> dict:
    """Fingerprint one tensor: its shape, its dtype and a hash of its bytes."""
    array = tensor.detach().cpu().contiguous()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype).replace("torch.", ""),
        # Hashed straight from the array's buffer; ``tobytes`` would copy every byte of it first,
        # and the largest of these is 630 MB.
        "sha256": hashlib.sha256(array.numpy()).hexdigest(),
    }


def published() -> tuple[Path, str, str]:
    """Locate the newest published checkpoint the way mozo itself would.

    Returns:
        Its path, the revision it belongs to, and the sha256 the manifest publishes for it. The
        last two are recorded alongside the fingerprints so a failure can say whether the code
        drifted or the weights simply are not the ones this was recorded against.
    """
    model = manifest()["models"].get(MODEL_ID)
    if model is None:
        raise SystemExit(f"the manifest publishes no {MODEL_ID}.")
    revision = model["latest"]
    artifact = model["revisions"][revision]["artifacts"]["torch-fp32"]
    path = ROOT / "weights" / artifact["path"]
    if not path.is_file():
        raise SystemExit(
            f"no SAM 3 weights at {path}. This gate needs the published checkpoint; "
            "see mozo/vendors/sam3_deploy/PROVENANCE.md."
        )
    return path, revision, artifact["sha256"]


def observe(image: Path, checkpoint: Path) -> dict[str, dict]:
    """Run every stage of mozo's SAM 3 and fingerprint what each produces.

    The towers are the Segmenter's own rather than a second set built beside it: loading the
    checkpoint twice and holding two copies of a 3.45 GB model buys nothing, and running the
    stages through the same modules the Segmenter uses is the stronger check anyway.

    Args:
        image: The photograph to run on.
        checkpoint: Meta's published ``sam3.pt``.

    Returns:
        Stage name -> fingerprint, in a stable order.
    """
    torch.set_grad_enabled(False)
    pixels = load_image(str(image))
    segmenter = Segmenter(checkpoint)

    seen: dict[str, dict] = {}
    seen["preprocess"] = digest(preprocess(pixels))

    # Through the cache, so every later stage reuses this one encode rather than paying 4.8 s
    # again -- and so a cache that quietly returns something else fails here.
    features = segmenter.encode_image(pixels)
    for level, tensor in enumerate(features["concept"]):
        seen[f"vision.concept.{level}"] = digest(tensor)
    seen["vision.positions"] = digest(features["positions"])

    # Batched deliberately: padding and the attention mask only have a shape to get wrong when
    # prompts of different lengths are encoded together.
    ids = segmenter.tokenizer(PROMPTS)
    seen["tokenizer.ids"] = digest(ids)
    encoded = segmenter.text(ids)
    for name in ("mask", "features", "embeddings"):
        seen[f"text.{name}"] = digest(encoded[name])

    for prompt in PROMPTS:
        one = segmenter.encode_text(prompt)
        result = segmenter.concept(
            features["concept"], features["positions"], one["features"], one["mask"]
        )
        for name in ("masks", "boxes", "logits", "presence", "semantic"):
            seen[f"concept.{prompt}.{name}"] = digest(result[name])

    boxes, labels = EXEMPLARS
    one = segmenter.encode_text("visual")
    result = segmenter.concept(
        features["concept"], features["positions"], one["features"], one["mask"],
        torch.tensor(boxes, dtype=torch.float32)[None],
        torch.tensor(labels, dtype=torch.long)[None],
    )
    for name in ("masks", "boxes", "logits", "presence"):
        seen[f"exemplars.{name}"] = digest(result[name])

    for prompt in PROMPTS:
        found = segmenter.predict(pixels, prompt)
        for name in ("masks", "boxes", "scores"):
            seen[f"segmenter.{prompt}.{name}"] = digest(found[name])
    # Re-running a held prompt on a held image must return the identical answer.
    repeat = segmenter.predict(pixels, PROMPTS[0])
    seen["segmenter.cached.masks"] = digest(repeat["masks"])

    # The click path. Its encode is deliberately not the concept path's -- the two preprocess
    # differently -- so this also pins that the right one is being used.
    seen["preprocess.click"] = digest(preprocess_click(pixels))
    for level, tensor in enumerate(segmenter.encode_click(pixels)):
        seen[f"click.features.{level}"] = digest(tensor)
    for spec in CLICKS:
        out = segmenter.segment(pixels, **click_prompt(spec, pixels.shape[:2]))
        for name in ("masks", "scores", "logits"):
            seen[f"click.{spec[0]}.{name}"] = digest(out[name])

    # A held image must answer a repeated click identically. A cache that quietly returns
    # something else is worse than no cache, and this is the only place it would show.
    repeat = segmenter.segment(pixels, **click_prompt(CLICKS[0], pixels.shape[:2]))
    seen["click.cached.masks"] = digest(repeat["masks"])
    seen["click.cached.logits"] = digest(repeat["logits"])

    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a checkpoint to check instead of the published one")
    arguments = parser.parse_args()

    if not DIGESTS.exists():
        raise SystemExit(f"no fingerprints at {DIGESTS}. Regenerate with sam3_reference.py.")
    recorded = json.loads(DIGESTS.read_text())

    if arguments.checkpoint:
        checkpoint, sha256 = arguments.checkpoint, None
    else:
        checkpoint, revision, sha256 = published()
        print(f"revision {revision}")
    print(f"weights  {checkpoint}")
    print(f"image    {FIXTURE}\n")

    # Answered before running anything, because it decides what a failure below would mean.
    if sha256 is not None and recorded.get("sha256") not in (None, sha256):
        print("note: these fingerprints were recorded against a different checkpoint\n"
              f"      recorded {recorded.get('revision')} {str(recorded.get('sha256'))[:16]}\n"
              f"      present  {revision} {sha256[:16]}\n"
              "      any difference below is the weights, not the code.\n")

    observed = observe(FIXTURE, checkpoint)
    expected = recorded["stages"]

    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    failed = sorted(n for n in expected if n in observed and observed[n] != expected[n])

    for name in failed:
        got, want = observed[name], expected[name]
        detail = (f"shape {got['shape']} != {want['shape']}"
                  if got["shape"] != want["shape"] else "bytes differ")
        print(f"  FAIL  {name:<44} {detail}")
    for name in missing:
        print(f"  GONE  {name:<44} stage no longer produced")
    for name in extra:
        print(f"  NEW   {name:<44} stage not in the fingerprints")

    checked = len(expected) - len(missing)
    if failed or missing:
        print(f"\n{len(failed)} of {checked} stages differ from the recorded reference.")
        return 1
    print(f"  {checked} stages, every one identical to the recorded reference.")
    if extra:
        print(f"  ({len(extra)} new stage(s) not yet recorded -- rerun sam3_reference.py)")
    print("\nSAM 3: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
