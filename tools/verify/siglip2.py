#!/usr/bin/env python3
"""Check that mozo's SigLIP 2 returns exactly what the published model does.

Three paths reach the same weights and this compares all three.

``mozo/vendors/siglip2_deploy`` is one: build an ``Encoder``, hand it an image and some phrases,
read the vectors off it. ``transformers/models/siglip`` is the second. And mozo itself is the
third: registry lookup, weights resolution, adapter, PixelFlow result.

**The reference is ``transformers``, and that is a decision worth stating.** SigLIP 2's authors
publish in JAX -- ``google-research/big_vision`` -- and there is no authors' PyTorch. So the
reference here is the same one ``owlv2_deploy`` chose for the same reason: the PyTorch
implementation is the one whose numbers the published PyTorch checkpoints reproduce, and the one
mozo can be checked against on every run. What relationship those checkpoints bear to the JAX
originals is not something upstream states -- ``convert_siglip_to_hf.py`` carries expected outputs
for eight SigLIP *1* models and none for any SigLIP 2 -- so mozo claims only the link it can
measure, and claims it exactly.

**The comparison is exact.** Not "close": a tolerance would hide precisely the drift this exists to
catch. Three real divergences found while building this package would have been swallowed by any
sane tolerance -- rescaling by 1/255 before normalising instead of folding it into the statistics
(5.9e-08), a ``LayerNorm`` left at torch's 1e-5 epsilon, and resizing the float tensor rather than
the uint8 one.

**Four things about the comparison are pinned.**

*The device is the CPU*, where mozo's published fp32 artifact matches upstream tensor for tensor.

*The attention is eager.* ``transformers`` dispatches through ``ALL_ATTENTION_FUNCTIONS`` and
defaults to ``sdpa``, which is the same arithmetic in a different order. ``siglip2_deploy``
implements the eager path, so the reference is asked for it too. The image tower's pooling head is
*not* governed by this: it calls ``nn.MultiheadAttention`` with ``need_weights`` left at ``True``,
which takes torch's own unfused branch either way.

*The reference's parameters are re-allocated before comparing.* ``from_pretrained`` places tensors
at whatever offset the checkpoint file had, and BLAS picks different vectorised paths for a matrix
whose storage happens to be page-aligned; OWLv2's gate measured 1.2e-07 on that alone. Cloning
first means what is compared is arithmetic rather than addresses.

*The tokenizer reference is ``Siglip2Tokenizer``, not ``AutoTokenizer``.* This is the one place
mozo deliberately diverges from a published config, and it is not cosmetic -- see
``PROVENANCE.md``. Gating against ``AutoTokenizer`` would pin the wrong behaviour bit-exactly.

Run it with the checkpoints in place::

    python tools/verify/siglip2.py base-224

Versions matter for a zero-tolerance gate, and the reference has been refactored under these
checkpoints before -- ``SiglipImageProcessor`` moved from PIL to torchvision, which changed the
pixels. The versions parity was established against are printed on every run.
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
# bytes. Same line, same reason, as tools/verify/clip.py and tools/verify/owlv2.py.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import fixtures  # noqa: E402
from conftest import as_pixelflow_classifications  # noqa: E402

from mozo.image import load_image  # noqa: E402
from mozo.vendors.siglip2_deploy import CONTEXT, SPECS, Encoder  # noqa: E402
from mozo.vendors.siglip2_deploy.checkpoint import load_scoring  # noqa: E402
from mozo.vendors.siglip2_deploy.image import preprocess  # noqa: E402
from mozo.vendors.siglip2_deploy.network import normalise  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

#: The versions this family's parity was established against. Printed, not enforced: a newer
#: reference is worth checking against, and a silent pass under one is worth doubting.
PINNED = {"torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.8.0"}

#: Prompt sets the gate runs. A single phrase, several, one whose case and punctuation the
#: normalisation has to survive, one that exercises byte fallback and non-Latin scripts, and one
#: long enough to use the context without exceeding it.
PROMPTS = [
    ["a forklift"],
    ["a forklift", "a person", "an empty aisle"],
    ["A PHOTO of a Café — with 2024 items!", "a person's hand"],
    ["日本語のテキスト", "emoji 🚀🔥", "Русский текст"],
    ["a photograph of a warehouse aisle with pallets stacked high on both sides"],
]


def versions() -> str:
    """What is actually installed, next to what parity was established against."""
    import torchvision
    import transformers

    actual = {"torch": torch.__version__, "torchvision": torchvision.__version__,
              "transformers": transformers.__version__}
    return " ".join(
        f"{name} {actual[name]}" + ("" if actual[name] == want else f" (pinned {want})")
        for name, want in PINNED.items()
    )


def against_upstream(variant: str, checkpoint: Path) -> tuple[int, list[str]]:
    """Compare every stage of the vendored model against ``transformers``."""
    from transformers import AutoImageProcessor, Siglip2Tokenizer, SiglipModel
    from PIL import Image

    spec = SPECS[variant]
    ours = Encoder(checkpoint, spec, device="cpu")
    scale, bias = load_scoring(checkpoint)

    # Composed here rather than read off the spec: where a variant is published is a fact about
    # publishing, and the vendored package deliberately carries no addresses.
    repo = f"google/siglip2-{spec.upstream}"
    theirs = SiglipModel.from_pretrained(
        repo, attn_implementation="eager", dtype=torch.float32).eval()
    # Compare arithmetic, not addresses. See the module docstring.
    for parameter in theirs.parameters():
        parameter.data = parameter.data.clone()

    their_preprocess = AutoImageProcessor.from_pretrained(repo)
    their_tokenizer = Siglip2Tokenizer.from_pretrained(repo)

    checked, failures = 0, []
    for path in fixtures():
        name = path.name
        image = load_image(str(path))

        # Preprocessing, against upstream's own processor rather than a restatement of it.
        want_pixels = their_preprocess(
            Image.open(path).convert("RGB"), return_tensors="pt")["pixel_values"]
        got_pixels = preprocess(image, spec.resolution)[None]
        checked += 1
        if not torch.equal(got_pixels, want_pixels):
            failures.append(
                f"{name}: preprocess differs by {(got_pixels - want_pixels).abs().max():.3e}")
            continue

        with torch.no_grad():
            want_image = theirs.get_image_features(pixel_values=want_pixels).pooler_output
            got_image = ours.vision()(got_pixels)
        checked += 1
        if not torch.equal(got_image, want_image):
            failures.append(
                f"{name}: image features differ by {(got_image - want_image).abs().max():.3e}")

        for prompts in PROMPTS:
            # Tokenization, against the class whose normalisation matches training.
            want_tokens = torch.tensor(
                their_tokenizer(prompts, padding="max_length", max_length=CONTEXT)["input_ids"])
            got_tokens = ours.tokenizer(prompts)
            checked += 1
            if not torch.equal(got_tokens, want_tokens):
                failures.append(f"{name} {prompts[0]!r}: token ids differ")
                continue

            with torch.no_grad():
                want_text = theirs.get_text_features(input_ids=want_tokens).pooler_output
                got_text = ours.text()(got_tokens)
            checked += 1
            if not torch.equal(got_text, want_text):
                failures.append(
                    f"{name} {prompts[0]!r}: text features differ by "
                    f"{(got_text - want_text).abs().max():.3e}")
                continue

            with torch.no_grad():
                theirs_out = theirs(pixel_values=want_pixels, input_ids=want_tokens)

            # The scaled, biased logits, which is what upstream's forward returns.
            got_logits = ((normalise(got_text) @ normalise(got_image).T) * scale.exp() + bias).T
            checked += 1
            if not torch.equal(got_logits, theirs_out.logits_per_image):
                failures.append(
                    f"{name} {prompts[0]!r}: logits differ by "
                    f"{(got_logits - theirs_out.logits_per_image).abs().max():.3e}")
                continue

            # And the probabilities, which is what mozo actually returns.
            checked += 1
            if not torch.equal(
                ours.classify(image, prompts), torch.sigmoid(theirs_out.logits_per_image)[0]
            ):
                failures.append(f"{name} {prompts[0]!r}: probabilities differ")

    return checked, failures


def against_mozo(variant: str, checkpoint: Path) -> tuple[int, list[str]]:
    """Compare mozo's adapter against the vendored encoder it wraps.

    The adapter sorts, converts to PixelFlow and rounds. This checks it changes nothing else.
    """
    from mozo.adapters.siglip2 import Siglip2Predictor

    vendor = Encoder(checkpoint, SPECS[variant], device="cpu")
    adapter = Siglip2Predictor(variant, device="cpu", checkpoint_path=checkpoint)

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
    parser.add_argument("variants", nargs="*", default=["base-224"],
                        help="variants to check (default: base-224)")
    parser.add_argument("--mozo-only", action="store_true",
                        help="skip the transformers comparison")
    args = parser.parse_args()

    print(f"reference: {versions()}\n")

    total, problems = 0, []
    for variant in args.variants:
        try:
            checkpoint = resolve("siglip2", variant, "torch-fp32")
        except WeightsError as error:
            print(f"{variant}: {error}")
            return 1

        checked, failures = against_mozo(variant, checkpoint)
        total += checked
        problems.extend(f"[mozo/{variant}] {f}" for f in failures)
        print(f"{variant}: {checked} mozo-vs-vendor comparisons, {len(failures)} failed")

        if not args.mozo_only:
            checked, failures = against_upstream(variant, checkpoint)
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
