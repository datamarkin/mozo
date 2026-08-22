#!/usr/bin/env python3
"""Check that mozo's Grounding DINO returns exactly what the published model does.

Three paths reach the same weights and this compares all three.

``mozo/vendors/grounding_dino_deploy`` is one: build a ``Predictor``, hand it an image and some
phrases, read the boxes off it. ``IDEA-Research/GroundingDINO`` is the second -- the authors' own
PyTorch implementation, which unlike OWLv2's case *is* the code the published checkpoints
reproduce, so it is the reference rather than a port of one. And mozo itself is the third:
registry lookup, weights resolution, adapter, PixelFlow result.

Between the first two sit a from-scratch WordPiece tokenizer, a from-scratch BERT tower, a
rewritten Swin backbone, a rewritten encoder and decoder, and a rewritten preprocessing step. Any
of them could quietly change a number.

**The comparison is exact.** Not "close": a tolerance would hide precisely the drift this exists
to catch. One real divergence found while building this package would have been swallowed by any
sane tolerance in the wrong direction and was catastrophic in the right one -- aliasing the
encoder's box head to the decoder's, which a *strict* state-dict load reports nothing about
because both keys still match, and which moved the initial reference boxes by 12.9.

**Two things about the reference are pinned.**

``attn_implementation="eager"``. ``transformers`` changed BertModel's default to SDPA, which is
the same arithmetic in a different order and moves ``last_hidden_state`` by 1.5e-06. The
checkpoint predates that default. Eager is what it was trained and released against, so eager is
what mozo matches; running this gate without the pin fails, correctly.

The CUDA extension is not used. Upstream falls back to ``grid_sample`` when ``groundingdino._C``
is absent, and that fallback is what this package carries -- so the reference must take it too, or
the comparison measures a kernel rather than an implementation.

This gate needs a checkout of upstream and ``transformers<5``::

    git clone https://github.com/IDEA-Research/GroundingDINO
    python -m venv refenv && ./refenv/bin/pip install 'transformers<5' timm addict yapf
    ./refenv/bin/python tools/verify/grounding_dino.py --upstream ./GroundingDINO

Run without ``--upstream`` and it checks the mozo-vs-vendor half only, which needs nothing extra.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

# This checks a tree you just built, and reaching for the published bucket would verify the wrong
# bytes. Same line, same reason, as tools/verify/owlv2.py and tools/verify/_detection.py.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import fixtures  # noqa: E402
from conftest import as_pixelflow_reports  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.grounding_dino_deploy import SEPARATORS, SPECS, Predictor  # noqa: E402
from mozo.vendors.grounding_dino_deploy.checkpoint import build  # noqa: E402
from mozo.vendors.grounding_dino_deploy.network import phrase_masks  # noqa: E402
from mozo.vendors.grounding_dino_deploy.predictor import caption_for  # noqa: E402
from mozo.vendors.grounding_dino_deploy.text.tokenizer import Tokenizer  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

#: Prompt sets the gate runs. Chosen to exercise the parts that can silently differ: a single
#: phrase, several, a multi-word phrase whose tokens must stay grouped, and one that matches
#: nothing in the photograph.
PROMPTS = [
    ["person"],
    ["person", "laptop", "cup"],
    ["a yellow school bus", "a person holding a mug"],
    ["dinosaur"],
]

#: Upstream's config file per variant, relative to the checkout.
CONFIGS = {
    "tiny": "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "base": "groundingdino/config/GroundingDINO_SwinB_cfg.py",
}


def images() -> dict:
    """Every fixture photograph, decoded to mozo's contract.

    Through ``tools.common.fixtures`` so that adding a photograph widens this gate too -- three
    private copies of that glob is how adding one used to widen only whichever gates someone
    remembered, and this one matched ``*.jpg`` alone.
    """
    return {path.name: load_image(str(path)) for path in fixtures()}


def use_upstream(upstream: Path) -> None:
    """Put *upstream* ahead of everything on the import path, before anything imports from it.

    Called once, first. ``groundingdino`` may also be pip-installed -- it is a real package on
    PyPI -- and if any module imports it before this runs, that copy wins and the gate silently
    measures a version nobody chose. Which reference is being compared against is the one thing
    this script may not be vague about.
    """
    if not (upstream / "groundingdino").is_dir():
        raise SystemExit(f"{upstream} does not look like a GroundingDINO checkout")
    if "groundingdino" in sys.modules:
        raise SystemExit("groundingdino was imported before --upstream was applied")
    sys.path.insert(0, str(upstream))


def reference(upstream: Path, variant: str, checkpoint: Path):
    """Build the upstream model, with the two pins the module docstring explains."""
    import groundingdino.util.get_tokenlizer as get_tokenlizer
    from transformers import BertModel

    # Pin eager attention. Patched here rather than in the checkout so the reference stays
    # pristine and this gate carries its own reason for the pin.
    original = get_tokenlizer.get_pretrained_language_model

    def eager(text_encoder_type):
        if text_encoder_type == "bert-base-uncased":
            return BertModel.from_pretrained(text_encoder_type, attn_implementation="eager")
        return original(text_encoder_type)

    get_tokenlizer.get_pretrained_language_model = eager

    from groundingdino.models import build_model
    from groundingdino.util.misc import clean_state_dict
    from groundingdino.util.slconfig import SLConfig

    args = SLConfig.fromfile(str(upstream / CONFIGS[variant]))
    args.device = "cpu"
    model = build_model(args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(clean_state_dict(state), strict=False)
    return model.eval()


def tap(module, store: dict, name: str, read=lambda output: output):
    """Record one module's output under *name* every time it runs.

    *read* pulls the tensor out of whatever the module returns -- upstream's BERT wrapper hands
    back a dict, its backbone a pair of lists, its encoder a tuple.
    """
    module.register_forward_hook(lambda _m, _i, output: store.__setitem__(name, read(output)))


def taps(model, store: dict, *, upstream: bool) -> None:
    """Hook every intermediate the extraction could silently get wrong.

    The end-to-end logits are not enough on their own. Two implementations can agree on the last
    tensor and disagree in the middle -- and when they do disagree, a stage name is the whole
    difference between "something moved" and knowing which of eleven rewritten pieces moved it.
    Hooked on both sides from one function so the two can never tap different things.
    """
    if upstream:
        tap(model.bert, store, "bert", lambda out: out["last_hidden_state"])
        tap(model.backbone, store, "backbone", lambda out: [f.tensors for f in out[0]])
    else:
        tap(model.bert, store, "bert")
        tap(model.backbone, store, "backbone")

    tap(model.feat_map, store, "encoded_text")
    tap(model.transformer.encoder, store, "memory", lambda out: out[0])
    tap(model.transformer.encoder, store, "memory_text", lambda out: out[1])
    for index, layer in enumerate(model.transformer.decoder.layers):
        tap(layer, store, f"decoder_{index}")


def against_upstream(upstream: Path, variant: str, checkpoint: Path) -> tuple[int, list[str]]:
    """Compare every stage of the vendored model against the authors' own implementation."""
    import groundingdino.datasets.transforms as T
    from PIL import Image

    from mozo.vendors.grounding_dino_deploy.image import preprocess

    spec = SPECS[variant]
    ours = build(spec, checkpoint)
    theirs = reference(upstream, variant, checkpoint)
    tokenizer = Tokenizer()
    separators = torch.tensor(tokenizer.convert_tokens_to_ids(list(SEPARATORS)))

    ours_stages: dict = {}
    theirs_stages: dict = {}
    taps(ours, ours_stages, upstream=False)
    taps(theirs, theirs_stages, upstream=True)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    checked, failures = 0, []
    for name, image in images().items():
        # Preprocessing, against upstream's own PIL transform rather than a restatement of it.
        theirs_tensor, _ = transform(Image.fromarray(image), None)
        ours_tensor = preprocess(image, spec.short_side, spec.max_side)
        checked += 1
        if not torch.equal(ours_tensor, theirs_tensor):
            failures.append(
                f"{name}: preprocess differs by "
                f"{(ours_tensor - theirs_tensor).abs().max().item():.3e}"
            )
            continue

        for prompts in PROMPTS:
            caption = caption_for(prompts)

            # Tokenization, against the tokenizer upstream downloads.
            ids, types, mask = tokenizer.encode(caption)
            reference_ids = theirs.tokenizer(caption, return_tensors="pt")["input_ids"]
            checked += 1
            if ids != reference_ids[0].tolist():
                failures.append(f"{name} {prompts}: token ids differ")
                continue

            batch_ids = torch.tensor([ids])
            batch_types = torch.tensor([types])
            batch_mask = torch.tensor([mask], dtype=torch.bool)
            attention, positions, _ = phrase_masks(batch_ids, separators)

            with torch.no_grad():
                logits, boxes = ours(
                    ours_tensor[None], batch_ids, batch_types, batch_mask, attention, positions
                )
                want = theirs(theirs_tensor[None], captions=[caption])

            # Every intermediate, in the order the model computes them, so a divergence is
            # reported at the stage that caused it rather than at the end.
            if set(ours_stages) != set(theirs_stages):
                failures.append(
                    f"{name} {prompts}: hooked {sorted(ours_stages)} against "
                    f"{sorted(theirs_stages)} -- the two sides tapped different stages"
                )
                continue

            for stage in sorted(ours_stages):
                mine, reference_value = ours_stages[stage], theirs_stages[stage]
                pairs = (
                    list(zip(mine, reference_value))
                    if isinstance(mine, list)
                    else [(mine, reference_value)]
                )
                for level, (got, want_tensor) in enumerate(pairs):
                    label = f"{stage}[{level}]" if isinstance(mine, list) else stage
                    checked += 1
                    if not torch.equal(got, want_tensor):
                        failures.append(
                            f"{name} {prompts}: {label} differs by "
                            f"{(got - want_tensor).abs().max().item():.3e}"
                        )

            finite = want["pred_logits"].isfinite()
            checked += 2
            if not torch.equal(logits[finite], want["pred_logits"][finite]):
                failures.append(
                    f"{name} {prompts}: logits differ by "
                    f"{(logits[finite] - want['pred_logits'][finite]).abs().max().item():.3e}"
                )
            if not torch.equal(boxes, want["pred_boxes"]):
                failures.append(
                    f"{name} {prompts}: boxes differ by "
                    f"{(boxes - want['pred_boxes']).abs().max().item():.3e}"
                )

    return checked, failures


def against_mozo(variant: str, checkpoint: Path) -> tuple[int, list[str]]:
    """Compare mozo's adapter against the vendored predictor it wraps.

    The adapter sorts, converts to PixelFlow and rounds. This checks that it changes nothing else
    -- same count, same names, same boxes to PixelFlow's own precision.
    """
    from mozo.adapters.grounding_dino import GroundingDinoPredictor

    vendor = Predictor(checkpoint, SPECS[variant], device="cpu")
    adapter = GroundingDinoPredictor(variant, device="cpu", checkpoint_path=checkpoint)

    checked, failures = 0, []
    for name, image in images().items():
        for prompts in PROMPTS:
            raw = vendor(image, prompts)
            got = adapter.predict(image, prompts)
            checked += 1

            if len(raw) != len(got):
                failures.append(f"{name} {prompts}: {len(raw)} from vendor, {len(got)} from mozo")
                continue

            # Both sides through the one helper that owns PixelFlow's rounding, so what is
            # compared is the adapter's own steps rather than a private copy of that policy.
            ordered = sorted(raw, key=lambda d: -d.score)
            want_boxes, want_scores = as_pixelflow_reports(
                [d.box for d in ordered],
                [d.score for d in ordered],
                [d.prompt_index for d in ordered],
            )
            have = got.to_dict()
            have_boxes = [row["bbox"] for row in have]
            if [list(b) for b in want_boxes] != have_boxes:
                failures.append(f"{name} {prompts}: boxes differ after rounding")
                continue
            if [prompts[d.prompt_index] for d in ordered] != [r["class_name"] for r in have]:
                failures.append(f"{name} {prompts}: names differ")

    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=["tiny"],
                        help="variants to check (default: tiny)")
    parser.add_argument("--upstream", type=Path,
                        help="path to an IDEA-Research/GroundingDINO checkout")
    args = parser.parse_args()

    # Before anything can import it. See use_upstream().
    if args.upstream:
        use_upstream(args.upstream)

    total, problems = 0, []
    for variant in args.variants:
        try:
            checkpoint = resolve("grounding_dino", variant, "torch-fp32")
        except WeightsError as error:
            print(f"{variant}: {error}")
            return 1

        checked, failures = against_mozo(variant, checkpoint)
        total += checked
        problems.extend(f"[mozo/{variant}] {f}" for f in failures)
        print(f"{variant}: {checked} mozo-vs-vendor comparisons, {len(failures)} failed")

        if args.upstream:
            checked, failures = against_upstream(args.upstream, variant, checkpoint)
            total += checked
            problems.extend(f"[upstream/{variant}] {f}" for f in failures)
            print(f"{variant}: {checked} upstream comparisons, {len(failures)} failed")

    if problems:
        print(f"\n{len(problems)} FAILURES:")
        for problem in problems[:40]:
            print(f"  {problem}")
        return 1

    print(f"\n{total} comparisons, all identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
