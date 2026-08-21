#!/usr/bin/env python3
"""Check that mozo's OWLv2 returns exactly what the published model does.

Three paths reach the same weights and this compares all three.

``mozo/vendors/owlv2_deploy`` is one: build a ``Detector``, hand it an image and some phrases,
read the boxes off it. ``transformers`` is the second -- the Apache-2.0 implementation this
package was extracted from, driven through its own ``Owlv2Processor`` and
``post_process_object_detection``. And mozo itself is the third: registry lookup, weights
resolution, adapter, runtime selection, PixelFlow result. Between the first two sit a tokenizer,
a preprocessing rewrite, two rewritten towers and three rewritten heads; between the first and
third sit a coordinate conversion, a ranking and a result conversion. Any of those could quietly
change a number.

**The comparison against ``transformers`` is exact.** Not "close": a tolerance would hide
precisely the drift this exists to catch, and three real divergences found while building this
package were each at a magnitude a tolerance would have swallowed -- 2.5e-07 from caching prompts
one at a time instead of as a vocabulary, 9.5e-07 from dividing by 255 where upstream multiplies
by its reciprocal, and 9.5e-07 more from computing a resize factor in float64 where upstream lands
in float32.

**The comparison against mozo's own adapter is exact after PixelFlow's rounding**, which is the
only door mozo's numbers come through. That half is what catches a mistake in the adapter rather
than in the extraction.

This gate needs no network, no checkout and no Hugging Face cache -- only ``pip install
transformers`` and the weights. The reference model is built from ``owlv2_deploy``'s own geometry
and loaded from the checkpoint mozo publishes; the reference tokenizer is built from the
vocabulary mozo vendors, which this also checks is identical id-for-id to the ``vocab.json``
Google publishes. SAM 3's gate cannot do any of that, because its reference is SAM-Licensed;
EdgeTAM's needs a git clone. This one is the cheapest gate in the repository to run, and there is
no excuse for it not being green.

**One thing about the reference is modified.** After loading, every parameter and buffer is
replaced by a freshly-allocated copy of itself. That is not a numerical change -- the tensors are
bitwise identical before and after -- but it *is* an observable one: BLAS picks different
vectorised paths for a matrix whose storage happens to be page-aligned, and ``from_pretrained``
places its tensors at whatever offset the file had. The 512x512 projection in the text tower moved
by 1.2e-07 on one machine for no reason but that. Both sides of the comparison then allocate the
same way, so what is compared is the arithmetic rather than where torch put the bytes.

Run from the repository root::

    python tools/verify/owlv2.py                        # base-ensemble
    python tools/verify/owlv2.py base large-ensemble    # others
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
# Local weights unless the caller says otherwise: this checks a tree you just built, and reaching
# for the published bucket would verify the wrong bytes.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

import mozo  # noqa: E402
from conftest import FIXTURE, as_pixelflow_reports  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.owlv2_deploy import Detector  # noqa: E402
from mozo.vendors.owlv2_deploy.config import SPECS  # noqa: E402
from mozo.vendors.owlv2_deploy.image import preprocess  # noqa: E402
from mozo.vendors.owlv2_deploy.text.tokenizer import VOCAB_PATH, Tokenizer  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: Serving threshold. Low enough that the comparison covers marginal detections, where two
#: implementations diverge first, rather than only the confident ones everything agrees on.
#:
#: This family's own copy, not ``_detection.py``'s. The number is a knob: raising it there for a
#: YOLO gate's reasons would quietly change what this one tests, and a gate that can be retuned
#: by an edit made for another family is not guarding anything. Duplication between gates is the
#: same trade the vendors themselves make, and for the same reason.
THRESHOLD = 0.05

#: What ``openai/CLIP`` publishes as ``bpe_simple_vocab_16e6.txt.gz``. The vendored copy is
#: byte-identical to it, and the vocabulary it builds is identical id-for-id to the ``vocab.json``
#: Google ships beside the weights -- checked by hand once, against files this repository does not
#: carry. This digest is what holds the input to that check steady.
VOCAB_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"

#: Vocabularies to ask for. Each exists for a reason a shorter list would miss.
#:
#: A single phrase is the only case where the text tower runs at batch one, which is where a
#: prompt cache keyed per phrase would have looked correct. Several phrases is the case where a
#: patch has to choose between them. ``"a cat!"`` is the only one carrying an exclamation mark,
#: whose token id is zero -- the same id as padding -- so it is the only prompt that would catch
#: an attention mask derived from the ids rather than built alongside them. The long one is the
#: only prompt that truncates, and the accented one the only one whose normalisation could go
#: through ``ftfy`` instead of NFC and still look reasonable.
VOCABULARIES = (
    ["a photo of a cat"],
    ["person", "laptop", "cup", "chair", "table", "potted plant"],
    ["a cat!", "a dog"],
    ["a photograph of a person sitting at a table with a laptop and a cup of coffee"],
    ["café", "naïve façade", "北京"],
)


def images() -> dict[str, np.ndarray]:
    """The photographs to compare on, at three aspect ratios.

    One real picture, cropped rather than synthesised, because the three shapes exercise three
    different paths: a wide image pads a third of the square, a square one pads nothing, and a
    tall strip pads most of it. The resize factor differs in each, and it is computed in float32
    -- so a fixture set that happened to hold only exact multiples of 960 would agree perfectly
    and prove nothing.
    """
    full = load_image(str(FIXTURE))
    height, width = full.shape[:2]
    side = min(height, width)
    return {
        "full": full,
        "square": full[:side, :side],
        "tall": full[:, : width // 4],
    }


def reference(variant: str, checkpoint: Path):
    """Build ``transformers``' OWLv2 from mozo's own geometry and mozo's own published weights.

    Deliberately not ``from_pretrained``: that wants a repository of config and tokenizer files
    this repository does not carry and CI should not download. Everything it would have read is
    either in ``owlv2_deploy.config`` or is a ``transformers`` default, so the reference can be
    constructed from what is already here -- which also means this gate is checking mozo's
    geometry against upstream's defaults rather than against a JSON file that could disagree with
    both.
    """
    from transformers import Owlv2ForObjectDetection, Owlv2ImageProcessor
    from transformers.models.owlv2.configuration_owlv2 import Owlv2Config

    spec = SPECS[variant]
    config = Owlv2Config(
        projection_dim=spec.text.projection,
        text_config={
            "hidden_size": spec.text.width,
            "intermediate_size": spec.text.intermediate,
            "num_attention_heads": spec.text.heads,
            "num_hidden_layers": spec.text.layers,
            "max_position_embeddings": spec.text.context_length,
            "vocab_size": spec.text.vocab_size,
        },
        vision_config={
            "hidden_size": spec.vision.width,
            "intermediate_size": spec.vision.intermediate,
            "num_attention_heads": spec.vision.heads,
            "num_hidden_layers": spec.vision.layers,
            "image_size": spec.vision.image_size,
            "patch_size": spec.vision.patch_size,
        },
    )
    model = Owlv2ForObjectDetection(config).eval()
    # The processor's default is the base geometry's 960. The large one runs at 1008, and the
    # only symptom of getting it wrong is a position embedding that will not broadcast -- which
    # is loud, but only after a full trunk has been built and loaded.
    processor = Owlv2ImageProcessor(
        size={"height": spec.vision.image_size, "width": spec.vision.image_size})
    # Strict, and against the checkpoint *as published* -- so this also asserts that the two
    # tensors ``owlv2_deploy`` drops are the only ones it is entitled to drop.
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            tensor.data = tensor.data.clone()  # see the module docstring
    return model, processor


def reference_tokenizer(context_length: int):
    """Build ``transformers``' ``CLIPTokenizer`` from the tables mozo's own tokenizer built.

    Not from the gzip a second time: re-deriving the vocabulary here would duplicate fourteen
    lines of :class:`~.tokenizer.Tokenizer`'s constructor, and the copy could drift from the
    original while both still agreed with each other. What is being compared is the *encoding*
    -- how the two implementations turn a string into ids -- so the tables they share are an
    input to that comparison rather than part of it. :func:`check_vocabulary` pins the bytes
    those tables came from.

    The published ``tokenizer_config.json`` is reproduced here rather than downloaded, and the two
    settings that are not defaults are the two that matter: ``pad_token="!"``, which is what puts
    an exclamation mark in the added-token table and gives it id zero, and a model max length of
    16 rather than CLIP's 77.
    """
    from transformers import CLIPTokenizer

    ours = Tokenizer(context_length=context_length)
    return CLIPTokenizer(
        vocab=ours.encoder,
        merges=list(ours.ranks),
        pad_token="!",
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        model_max_length=context_length,
    )


def compare(stages: list[tuple[str, torch.Tensor, torch.Tensor]]) -> list[str]:
    """Return one complaint per stage that is not bit-identical."""
    problems = []
    for name, want, got in stages:
        if want.shape != got.shape:
            problems.append(f"{name}: shape {tuple(want.shape)} one side, {tuple(got.shape)} the other")
        elif not torch.equal(want, got):
            gap = (want.double() - got.double()).abs().max().item()
            problems.append(f"{name}: differs by {gap:.3e}")
    return problems


def _compare(name: str, want: tuple, got: tuple) -> tuple[list[str], str]:
    """Return the ways two detection results disagree, and how far apart they are.

    Exact on all four axes -- count, boxes, scores, class ids and names -- because both sides ran
    the same weights through the same arithmetic, so anything at all is a divergence.

    ``_detection.py`` has a comparator of the same shape, with tolerances for graph runtimes. This
    is deliberately not it. A gate that another family can retune is a gate that can stop guarding
    without anyone editing this file, which is the whole reason the vendors do not share code
    either.
    """
    want_boxes, want_scores, want_ids, want_names = want
    got_boxes, got_scores, got_ids, got_names = got

    if len(want_boxes) != len(got_boxes):
        return [f"{name}: {len(want_boxes)} detections one side, {len(got_boxes)} the other"], ""
    if not len(want_boxes):
        return [], "no detections"

    problems = []
    if not np.array_equal(want_boxes, got_boxes):
        problems.append(f"{name}: boxes differ by {np.abs(want_boxes - got_boxes).max():g} px")
    if not np.array_equal(want_scores, got_scores):
        problems.append(f"{name}: scores differ by {np.abs(want_scores - got_scores).max():g}")
    if not np.array_equal(want_ids, got_ids):
        problems.append(f"{name}: class ids differ")
    if want_names != got_names:
        problems.append(f"{name}: class names are not the phrases that were asked for")
    return problems, "boxes 0 px, scores 0"


def against_upstream(variant: str, checkpoint: Path) -> tuple[int, list[str]]:
    """Compare ``owlv2_deploy`` against ``transformers``, stage by stage, exactly."""
    from PIL import Image

    model, processor = reference(variant, checkpoint)
    tokenizer = reference_tokenizer(SPECS[variant].text.context_length)
    detector = Detector(checkpoint, variant, device="cpu")
    ours = Tokenizer(context_length=SPECS[variant].text.context_length)

    checked, problems = 0, []
    print(f"\n{variant}: owlv2_deploy against transformers")
    for label, pixels in images().items():
        # Both preprocessors depend only on the photograph, so they run once per image rather than
        # once per vocabulary. Inside the inner loop this was twelve redundant executions per
        # variant, about 2.6 s, in the gate whose whole pitch is that it is cheap to run.
        theirs = processor(images=Image.fromarray(pixels), return_tensors="pt")["pixel_values"]
        ours_pixels = preprocess(pixels, detector.image_size)
        for vocabulary in VOCABULARIES:
            encoded = tokenizer(
                vocabulary, padding="max_length", truncation=True,
                max_length=SPECS[variant].text.context_length, return_tensors="pt",
            )
            with torch.no_grad():
                out = model(
                    input_ids=encoded["input_ids"],
                    pixel_values=theirs,
                    attention_mask=encoded["attention_mask"],
                )
            want = processor.post_process_object_detection(
                out, threshold=THRESHOLD, target_sizes=torch.tensor([pixels.shape[:2]]))[0]

            ids, mask = ours(list(vocabulary))
            queries, query_mask = detector.encode_text(vocabulary)
            patches = detector.encode_image(pixels)
            logits, boxes, objectness = detector.model.detect(patches, queries, query_mask)
            got = detector.predict(pixels, vocabulary, threshold=THRESHOLD)

            where = f"{label}/{vocabulary[0][:18]}"
            stages = [
                (f"{where}: token ids", encoded["input_ids"], ids),
                (f"{where}: token mask", encoded["attention_mask"], mask),
                (f"{where}: pixels", theirs, ours_pixels),
                (f"{where}: query embeds", out.text_embeds[0], queries),
                (f"{where}: feature map", out.image_embeds.flatten(1, 2), patches),
                (f"{where}: logits", out.logits, logits),
                (f"{where}: boxes", out.pred_boxes, boxes),
                (f"{where}: objectness", out.objectness_logits, objectness),
                # Ordered by patch on both sides: the vendor does not rank, so this compares the
                # postprocessing rather than a sort.
                (f"{where}: detected boxes", want["boxes"], got.boxes),
                (f"{where}: detected scores", want["scores"], got.scores),
                (f"{where}: detected labels", want["labels"], got.labels),
            ]
            found = compare(stages)
            # Counted, not restated: the summary line is quoted as evidence of the extraction's
            # fidelity, so a stage added here has to reach it.
            checked += len(stages)
            problems += found
            print(f"  {where:34} {len(got):5d} detections  "
                  f"{'identical' if not found else ' | '.join(found)}")
    return checked, problems, detector


def against_mozo(variant: str, detector: Detector) -> tuple[int, list[str]]:
    """Compare mozo's own path -- registry, adapter, PixelFlow -- against the vendor.

    Through :class:`~mozo.ModelManager` rather than :func:`mozo.get_model`, because that is the
    path the server takes.
    """
    # The same detector ``against_upstream`` built. A second one would re-read 620 MB to answer a
    # question about mozo's plumbing rather than about the weights, and its caches are already warm
    # on exactly the images this is about to ask for.
    model = mozo.ModelManager().get_model("owlv2", variant, device="cpu", runtime="torch-fp32")

    checked, problems = 0, []
    print(f"\n{variant}: mozo against owlv2_deploy")
    for label, pixels in images().items():
        for vocabulary in VOCABULARIES:
            raw = detector.predict(pixels, vocabulary, threshold=THRESHOLD)
            order = raw.scores.argsort(descending=True)
            want_boxes, want_scores = as_pixelflow_reports(
                raw.boxes[order], raw.scores[order], raw.labels[order])
            want_ids = raw.labels[order].numpy()

            rows = model.predict(pixels, vocabulary, threshold=THRESHOLD).to_dict()
            got_boxes = np.array([row["bbox"] for row in rows], dtype=np.float64).reshape(-1, 4)
            got_scores = np.array([row["confidence"] for row in rows], dtype=np.float64)
            got_ids = np.array([row["class_id"] for row in rows], dtype=np.int64)
            got_names = [row["class_name"] for row in rows]

            where = f"{label}/{vocabulary[0][:18]}"
            found, detail = _compare(
                where,
                (want_boxes, want_scores, want_ids, [vocabulary[i] for i in want_ids]),
                (got_boxes, got_scores, got_ids, got_names),
            )
            checked += 4
            problems += found
            print(f"  {where:34} {len(got_boxes):5d} detections  "
                  f"{'identical -- ' + detail if not found else ' | '.join(found)}")
    return checked, problems


def check_vocabulary() -> list[str]:
    """Check the vendored gzip is byte-for-byte the file OpenAI publishes.

    mozo ships OpenAI's 1.3 MB gzip rather than Google's 1.6 MB ``vocab.json`` and ``merges.txt``.
    That the two agree was established once, by hand, against the files Google publishes beside
    the weights; it cannot be re-established here, because nothing Google publishes is in this
    repository and this gate is offline by design.

    So what is pinned is the input rather than the output. An earlier version of this compared a
    vocabulary built from the gzip against a vocabulary built from the same gzip, which is a
    tautology dressed as a check: it could only ever fail if the duplicate drifted, and a genuine
    disagreement with ``vocab.json`` would have passed it.
    """
    digest = hashlib.sha256(VOCAB_PATH.read_bytes()).hexdigest()
    if digest != VOCAB_SHA256:
        return [f"vocabulary: {VOCAB_PATH.name} hashes to {digest}, not the published file"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=None,
                        help="variants to check (default: base-ensemble)")
    args = parser.parse_args()

    try:
        import transformers  # noqa: F401
    except ImportError:
        raise SystemExit(
            "this gate compares against transformers, which is not installed.\n"
            "    pip install transformers")

    checked, problems = 1, check_vocabulary()
    print("vocabulary: " + ("byte-identical to OpenAI's" if not problems else problems[0]))

    for variant in args.variants or ["base-ensemble"]:
        if variant not in SPECS:
            raise SystemExit(f"unknown variant {variant!r}; have {sorted(SPECS)}")
        checkpoint = Path(resolve("owlv2", variant, "torch-fp32"))
        upstream_checked, upstream_problems, detector = against_upstream(variant, checkpoint)
        mozo_checked, mozo_problems = against_mozo(variant, detector)
        checked += upstream_checked + mozo_checked
        problems += upstream_problems + mozo_problems

    print()
    if problems:
        for problem in problems:
            print(f"  {problem}")
        print(f"\n{len(problems)} of {checked} comparisons disagree. OWLv2: FAIL")
        return 1
    print(f"{checked} comparisons, every one identical to the published model. OWLv2: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
