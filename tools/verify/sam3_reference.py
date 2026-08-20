#!/usr/bin/env python3
"""Record what SAM 3 should return, by checking mozo against the published model.

This is the half of the SAM 3 gate that needs Meta's own implementation, and therefore the half
that cannot run in CI or on a machine without a checkout of it. It runs both paths on the
same inputs, refuses to write anything unless every stage they *both* produce agrees bit for bit,
and then stores a fingerprint of every stage mozo produces for :mod:`sam3` to check afterwards.

Those two sets are not the same size, and the difference matters. The reference exposes
``pred_masks``, ``pred_boxes``, ``pred_logits`` and ``presence_logit_dec`` from
``forward_grounding``, so the ``concept.*`` and ``exemplars.*`` fingerprints are genuinely
validated against Meta. ``preprocess``, ``vision.*``, ``text.*``, ``tokenizer.ids``,
``concept.*.semantic`` and ``segmenter.*`` have no counterpart there; they are written from mozo
alone and are regression fingerprints, not reference-validated ones. The closing summary says how
many of each were written.

Run it only when the model changes -- a new revision, a new stage, a deliberate change to the
numerics. Everyday verification is ``tools/verify/sam3.py``, which needs none of this.

The reference is Meta's ``facebookresearch/sam3``, which ships under the SAM License. It is used
here as a black box: run, and its tensors read. None of its source is copied into this repository,
and a fingerprint is a hash of numbers rather than the numbers, so what gets written is not SAM
Materials.

Two accommodations are needed to run it at all, both recorded in ``PROVENANCE.md``: it imports
triton and decord for the video path, which the image path never touches, and it allocates two
tensors on a hardcoded CUDA device. Neither affects a value.

Run from the repository root, with the reference importable::

    python tools/verify/sam3_reference.py --reference ~/Projects/smart-segment
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.verify.sam3 import (  # noqa: E402
    DIGESTS,
    EXEMPLARS,
    FIXTURE,
    PROMPTS,
    digest,
    observe,
    published,
)


class _Stub(types.ModuleType):
    """A module whose every attribute is a fresh throwaway class."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {})


def install(reference: Path) -> None:
    """Make the published model importable without its video dependencies.

    Its package ``__init__`` reaches the video tracker, which imports triton for a CUDA distance
    transform and decord for video decoding. Neither runs on the image path, and triton does not
    exist on macOS. ``torch._dynamo`` is imported first so it initialises against the real absence
    of triton rather than against the stub.
    """
    import torch._dynamo  # noqa: F401

    sys.path.insert(0, str(reference))
    package = types.ModuleType("sam3")
    package.__path__ = [str(reference / "sam3")]
    sys.modules["sam3"] = package

    language = _Stub("triton.language")
    language.constexpr = int
    language.dtype = type("dtype", (), {})
    triton = _Stub("triton")
    triton.jit = lambda fn: fn
    triton.language = language
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = language
    sys.modules["decord"] = _Stub("decord")

    # Both of these allocate on a hardcoded device="cuda". The position-encoding one only
    # pre-fills a shape-keyed cache that forward fills on demand anyway; the decoder's is live but
    # is arange(0, n) / n, so the device is the only thing wrong with it.
    from sam3.model.decoder import TransformerDecoder
    from sam3.model.position_encoding import PositionEmbeddingSine

    original = PositionEmbeddingSine.__init__
    PositionEmbeddingSine.__init__ = lambda self, *a, **k: original(
        self, *a, **{**k, "precompute_resolution": None}
    )
    coords = TransformerDecoder._get_coords
    TransformerDecoder._get_coords = staticmethod(lambda h, w, device: coords(h, w, "cpu"))

    # The geometry encoder pins a tensor before moving it to the input's device; with an
    # accelerator present torch pins into *its* memory and the move to CPU raises. Pinning affects
    # transfer speed, never values.
    torch.Tensor.pin_memory = lambda self, *a, **k: self


def reference_stages(image: Path, checkpoint: Path, bpe: Path) -> dict[str, dict]:
    """Fingerprint what the published model produces, stage for stage."""
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from PIL import Image

    model = build_sam3_image_model(
        bpe_path=str(bpe), device="cpu", checkpoint_path=str(checkpoint),
        load_from_HF=False, enable_segmentation=True, enable_inst_interactivity=False,
    ).eval()
    processor = Sam3Processor(model, device="cpu")
    pil = Image.open(image).convert("RGB")

    seen: dict[str, dict] = {}

    def grounding(prompt: str, boxes=None, labels=None) -> dict:
        state = processor.set_image(pil)
        state["backbone_out"].update(model.backbone.forward_text([prompt], device="cpu"))
        geometric = model._get_dummy_prompt()
        if boxes is not None:
            geometric.append_boxes(
                torch.tensor(boxes, dtype=torch.float32).view(len(boxes), 1, 4),
                torch.tensor(labels, dtype=torch.bool).view(len(labels), 1),
            )
        return model.forward_grounding(
            backbone_out=state["backbone_out"], find_input=processor.find_stage,
            geometric_prompt=geometric, find_target=None,
        )

    for prompt in PROMPTS:
        raw = grounding(prompt)
        seen[f"concept.{prompt}.masks"] = digest(raw["pred_masks"])
        seen[f"concept.{prompt}.boxes"] = digest(raw["pred_boxes"])
        seen[f"concept.{prompt}.logits"] = digest(raw["pred_logits"].squeeze(-1))
        seen[f"concept.{prompt}.presence"] = digest(raw["presence_logit_dec"])

    boxes, labels = EXEMPLARS
    raw = grounding("visual", boxes, labels)
    seen["exemplars.masks"] = digest(raw["pred_masks"])
    seen["exemplars.boxes"] = digest(raw["pred_boxes"])
    seen["exemplars.logits"] = digest(raw["pred_logits"].squeeze(-1))
    seen["exemplars.presence"] = digest(raw["presence_logit_dec"])
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True,
                        help="checkout containing the published sam3 package")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="a checkpoint to record against instead of the published one")
    arguments = parser.parse_args()

    # The image is the fixture and not a choice: the gate checks these fingerprints against that
    # one photograph, so recording them on another would leave it nothing it could reproduce.
    published_path, revision, sha256 = published()
    checkpoint = arguments.checkpoint or published_path
    if arguments.checkpoint:
        revision, sha256 = None, None
    bpe = arguments.reference / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe.exists():
        raise SystemExit(f"no reference package under {arguments.reference}")

    print(f"reference  {arguments.reference}")
    print(f"weights    {checkpoint}")
    print(f"image      {FIXTURE}\n")

    print("running mozo...")
    observed = observe(FIXTURE, checkpoint)

    print("running the published model...")
    install(arguments.reference)
    expected = reference_stages(FIXTURE, checkpoint, bpe)

    print("\ncomparing the stages both produce:")
    disagreed = []
    for name in sorted(expected):
        same = observed.get(name) == expected[name]
        print(f"  {'ok  ' if same else 'FAIL'}  {name}")
        if not same:
            disagreed.append(name)

    if disagreed:
        print(f"\n{len(disagreed)} stage(s) disagree with the published model. Nothing written --"
              " fingerprints are only worth having if they were true when recorded.")
        return 1

    DIGESTS.write_text(json.dumps({
        "image": FIXTURE.name,
        "revision": revision,
        "sha256": sha256,
        "prompts": PROMPTS,
        "exemplars": {"boxes": EXEMPLARS[0], "labels": EXEMPLARS[1]},
        "stages": observed,
    }, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {len(observed)} fingerprints to {DIGESTS.relative_to(ROOT)}: "
          f"{len(expected)} validated against the published model, "
          f"{len(observed) - len(expected)} recorded from mozo alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
