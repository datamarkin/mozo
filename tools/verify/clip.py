#!/usr/bin/env python3
"""Check that mozo's CLIP returns exactly what the published model does.

Three paths reach the same weights and this compares all three.

``mozo/vendors/clip_deploy`` is one: build an ``Encoder``, hand it an image and some phrases, read
the vectors off it. ``openai/CLIP`` is the second -- the authors' own implementation, which is what
the published checkpoints reproduce, so it is the reference rather than a port of one. And mozo
itself is the third: registry lookup, weights resolution, adapter, PixelFlow result.

Between the first two sit a from-scratch byte-pair tokenizer, a rewritten Vision Transformer, a
rewritten text transformer and a rewritten preprocessing step. Any of them could quietly change a
number.

**The comparison is exact.** Not "close": a tolerance would hide precisely the drift this exists to
catch. Two real divergences found while building this package would have been swallowed by any sane
tolerance -- the activation being ``QuickGELU`` rather than ``nn.GELU``, and Python's operator
precedence scaling the logits *before* the matmul rather than after, which moves them by 1.9e-06.

**Three things about the comparison are pinned.**

*The device is the CPU.* The published archives are mixed precision and ``clip.load`` restores fp32
only when the model lands on the CPU. "Bit-exact" is undefined until that is decided, and the CPU
is where mozo's published fp32 artifact matches upstream tensor for tensor.

*The reference is a checkout, not a package.* ``openai/CLIP`` is not on PyPI. It installs from
``git+https://github.com/openai/CLIP.git`` and pulls ``ftfy``, ``regex`` and ``tqdm``.

*The logits are scaled in upstream's order.* See ``compare_logits``.

Run it with a checkout and an isolated environment::

    git clone https://github.com/openai/CLIP
    python -m venv refenv && ./refenv/bin/pip install ftfy regex tqdm
    ./refenv/bin/python tools/verify/clip.py base --upstream ./CLIP

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
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

# This checks a tree you just built, and reaching for the published bucket would verify the wrong
# bytes. Same line, same reason, as tools/verify/owlv2.py and tools/verify/_detection.py.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import fixtures  # noqa: E402
from conftest import as_pixelflow_classifications  # noqa: E402

from mozo.image import load_image  # noqa: E402
from mozo.vendors.clip_deploy import SPECS, Encoder  # noqa: E402
from mozo.vendors.clip_deploy.checkpoint import read_logit_scale  # noqa: E402
from mozo.vendors.clip_deploy.image import preprocess  # noqa: E402
from mozo.vendors.clip_deploy.network import normalise  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

#: Prompt sets the gate runs. A single phrase, several, one with punctuation and case that the
#: tokenizer's cleaning has to survive, and one long enough to exercise the context but not exceed
#: it.
PROMPTS = [
    ["a forklift"],
    ["a forklift", "a person", "an empty aisle"],
    ["A PHOTO of a Café — with 2024 items!", "a person's hand"],
    ["a photograph of a warehouse aisle with pallets stacked high on both sides"],
]


def use_upstream(upstream: Path) -> None:
    """Put *upstream* ahead of everything on the import path, before anything imports from it."""
    if not (upstream / "clip").is_dir():
        raise SystemExit(f"{upstream} does not look like an openai/CLIP checkout")
    if "clip" in sys.modules:
        raise SystemExit("clip was imported before --upstream was applied")
    sys.path.insert(0, str(upstream))


def compare_logits(image: torch.Tensor, text: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale the way upstream does, which is not the way it reads.

    Upstream writes ``logit_scale * image_features @ text_features.t()``. In Python ``*`` and ``@``
    share precedence and associate left to right, so the scale multiplies the *features* and the
    matmul happens afterwards. Scaling the product instead is the same arithmetic in a different
    order and differs by 1.9e-06 on ViT-B/32 -- small enough to look like noise, and the whole
    difference between this gate passing and failing.
    """
    return (scale * image) @ text.T


def against_upstream(upstream: Path, variant: str, checkpoint: Path) -> tuple[int, list[str]]:
    """Compare every stage of the vendored model against the authors' own implementation."""
    import clip
    from PIL import Image

    spec = SPECS[variant]
    ours = Encoder(checkpoint, spec, device="cpu")
    # ``Spec.upstream`` is OpenAI's own name for the variant, so the checkout is asked for
    # exactly what the vendored config says it is -- not a second table to keep in step.
    theirs, their_preprocess = clip.load(spec.upstream, device="cpu", jit=False)
    theirs.eval()
    scale = read_logit_scale(checkpoint)

    checked, failures = 0, []
    for path in fixtures():
        name = path.name
        image = load_image(str(path))

        # Preprocessing, against upstream's own PIL transform rather than a restatement of it.
        want_pixels = their_preprocess(Image.open(path).convert("RGB"))
        got_pixels = preprocess(image, spec.resolution)
        checked += 1
        if not torch.equal(got_pixels, want_pixels):
            failures.append(
                f"{name}: preprocess differs by {(got_pixels - want_pixels).abs().max():.3e}")
            continue

        with torch.no_grad():
            want_image = theirs.encode_image(want_pixels[None])
            got_image = ours.vision()(got_pixels[None])
        checked += 1
        if not torch.equal(got_image, want_image):
            failures.append(
                f"{name}: image features differ by {(got_image - want_image).abs().max():.3e}")

        for prompts in PROMPTS:
            # Tokenization, against upstream's own tokenizer.
            want_tokens = clip.tokenize(prompts)
            got_tokens = ours.tokenizer(prompts)
            checked += 1
            if not torch.equal(got_tokens, want_tokens):
                failures.append(f"{name} {prompts[0]!r}: token ids differ")
                continue

            with torch.no_grad():
                want_text = theirs.encode_text(want_tokens)
                got_text = ours.text()(got_tokens)
            checked += 1
            if not torch.equal(got_text, want_text):
                failures.append(
                    f"{name} {prompts[0]!r}: text features differ by "
                    f"{(got_text - want_text).abs().max():.3e}")
                continue

            # The normalised similarity, which is what mozo actually returns, and the scaled
            # logits, which is what upstream's forward returns.
            want_cos = normalise(want_image) @ normalise(want_text).T
            got_cos = normalise(got_image) @ normalise(got_text).T
            checked += 1
            if not torch.equal(got_cos, want_cos):
                failures.append(
                    f"{name} {prompts[0]!r}: similarity differs by "
                    f"{(got_cos - want_cos).abs().max():.3e}")

            with torch.no_grad():
                want_logits, _ = theirs(want_pixels[None], want_tokens)
            got_logits = compare_logits(
                normalise(got_image), normalise(got_text), scale)
            checked += 1
            if not torch.equal(got_logits, want_logits):
                failures.append(
                    f"{name} {prompts[0]!r}: logits differ by "
                    f"{(got_logits - want_logits).abs().max():.3e}")

    return checked, failures


def against_mozo(variant: str, checkpoint: Path) -> tuple[int, list[str]]:
    """Compare mozo's adapter against the vendored encoder it wraps.

    The adapter sorts, converts to PixelFlow and rounds. This checks it changes nothing else.
    """
    from mozo.adapters.clip import ClipPredictor

    vendor = Encoder(checkpoint, SPECS[variant], device="cpu")
    adapter = ClipPredictor(variant, device="cpu", checkpoint_path=checkpoint)

    checked, failures = 0, []
    for path in fixtures():
        image = load_image(str(path))
        for prompts in PROMPTS:
            raw = vendor.classify(image, prompts)
            got = adapter.predict(image, prompts).to_dict()
            checked += 1

            # Both sides through the one helper that owns PixelFlow's rounding and ordering, so a
            # change to either policy cannot leave a private copy here disagreeing with it. Whole
            # rows, so the rank and the prompt each row names are pinned along with the score.
            want = as_pixelflow_classifications(raw.numpy(), prompts)
            if got != want:
                failures.append(f"{path.name} {prompts[0]!r}: {got} != {want}")

            # The vectors the adapter hands out must be the vendor's, unchanged.
            checked += 1
            if not torch.equal(
                torch.from_numpy(adapter.encode_text(prompts)), vendor.encode_text(prompts)
            ):
                failures.append(f"{prompts[0]!r}: adapter's text vectors differ from the vendor's")

    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=["base"],
                        help="variants to check (default: base)")
    parser.add_argument("--upstream", type=Path, help="path to an openai/CLIP checkout")
    args = parser.parse_args()

    if args.upstream:
        use_upstream(args.upstream)

    total, problems = 0, []
    for variant in args.variants:
        try:
            checkpoint = resolve("clip", variant, "torch-fp32")
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
