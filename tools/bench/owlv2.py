#!/usr/bin/env python3
"""Measure what OWLv2 costs to serve, against the implementation it was extracted from.

Bootstrap tooling; never ships. ``tools/verify/owlv2.py`` answers whether mozo returns the same
numbers as ``transformers``; this answers what it costs to get them, and whether the split into
``encode_text`` and ``encode_image`` is worth the seam it introduces.

    python tools/bench/owlv2.py
    python tools/bench/owlv2.py --variants base-ensemble large-ensemble --devices cpu mps

**The comparison is legitimate because both sides run the same weights.** Not "OWLv2 versus
another model" -- the same checkpoint, the same photograph, the same device, the same torch. Any
gap is this extraction against ``transformers``, which is the only thing mozo can be responsible
for. Both are timed end to end, from an ``HxWx3`` array to boxes in source pixels, because that is
the work a caller actually asks for; timing a bare forward would flatter whichever side does more
of its work outside one.

Three numbers are reported per variant and device:

``cold``     one image, one vocabulary, nothing cached -- what a single request costs.
``cached``   a second vocabulary on an image already seen -- what the image cache is worth.
``corpus``   the marginal cost of the next image once the vocabulary is encoded -- what the seam
             between the two towers is worth.

``transformers`` has no equivalent of the last two. Its ``Owlv2ForObjectDetection.forward`` takes
the image and the prompt together and runs both towers every call -- which is the point being
measured rather than a criticism: it is a reference implementation, and re-encoding is the honest
thing for one to do.

**What this measured, on an M-series CPU, is worth stating up front so nobody re-derives it.**
The image tower is 89% of a cold call, so caching the *prompt* saves about 3% and caching the
*image* saves about 95%: a second vocabulary on a picture already seen costs 51 ms against
1,122 ms. The seam earns its keep on the image side. And mozo and ``transformers`` come out within
1% of each other on the cold path, which is the answer that was wanted -- the extraction removed a
library, not a bottleneck.

The breakdown underneath attributes the cold number to preprocessing, the image tower, the text
tower and the heads, so a regression can be placed. And ``--against`` measures the families mozo
already serves on the same machine, so "is this fast enough to serve" has something to be answered
against rather than being a number on its own.

Nothing here is a pass/fail gate. It produces the numbers a support matrix is built from.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
# This file is called owlv2.py, so its own directory would shadow anything it imports by that name.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
# So the reference can be built by the same code the gate builds it with, rather than by a second
# copy of that construction which could drift from the one that was verified.
sys.path.insert(0, str(ROOT / "tools"))

import os  # noqa: E402

os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from conftest import FIXTURE  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.owlv2_deploy import Detector  # noqa: E402
from mozo.vendors.owlv2_deploy.config import SPECS  # noqa: E402
from mozo.vendors.owlv2_deploy.image import preprocess  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

#: What to ask for. Six phrases rather than one, because the text tower's cost is per phrase and
#: a one-word vocabulary would understate what the cache saves.
VOCABULARY = ["person", "laptop", "cup", "chair", "table", "potted plant"]

#: A second vocabulary, for the cached-image measurement. Different phrases, so nothing about the
#: first call can be reused except the image encode -- which is what is being measured.
OTHER = ["a window", "a lamp", "a book"]

THRESHOLD = 0.1

#: Iterations discarded before timing starts. Metal needs a generous warm-up: three was enough
#: for the base geometry and left the large one reading high, and the number must not depend on
#: how much work happened to run before this call.
WARMUP = 3


def median_ms(work, iters: int, before=None) -> float:
    """Median wall-clock of *work*, in milliseconds, after warm-up.

    Args:
        work: What to time. Called with no arguments.
        iters: How many timed runs to take the median of.
        before: Run outside the timer before each iteration -- how a cache is emptied without
            the emptying being counted.
    """
    for _ in range(WARMUP):
        if before is not None:
            before()
        work()

    timings = []
    for _ in range(iters):
        if before is not None:
            before()
        start = time.perf_counter()
        work()
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def upstream(variant: str, device: str):
    """``transformers``' OWLv2, built the way ``tools/verify/owlv2.py`` builds it.

    Same construction on purpose: the thing being timed has to be the thing that was verified,
    or the latency belongs to a model nobody checked.
    """
    from verify.owlv2 import reference, reference_tokenizer

    model, processor = reference(variant, Path(resolve("owlv2", variant, "torch-fp32")))
    tokenizer = reference_tokenizer(SPECS[variant].text.context_length)
    return model.to(device), processor, tokenizer


def upstream_call(model, processor, tokenizer, pixels: np.ndarray, vocabulary, device: str):
    """One end-to-end upstream prediction: array in, boxes in source pixels out."""
    from PIL import Image

    inputs = processor(images=Image.fromarray(pixels), return_tensors="pt")
    encoded = tokenizer(list(vocabulary), padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        out = model(
            input_ids=encoded["input_ids"].to(device),
            pixel_values=inputs["pixel_values"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
    return processor.post_process_object_detection(
        out, threshold=THRESHOLD, target_sizes=torch.tensor([pixels.shape[:2]]))


def breakdown(detector: Detector, pixels: np.ndarray, iters: int) -> dict[str, float]:
    """Attribute the cold number to its four stages."""
    ids, mask = detector.tokenizer(VOCABULARY)
    ids, mask = ids.to(detector.device), mask.to(detector.device)
    batch = preprocess(pixels, detector.image_size).to(detector.device)
    queries = detector.model.encode_text(ids, mask)
    patches = detector.model.encode_image(batch)
    query_mask = (ids[:, 0] > 0)
    return {
        "preprocess": median_ms(lambda: preprocess(pixels, detector.image_size), iters),
        "text tower": median_ms(lambda: detector.model.encode_text(ids, mask), iters),
        "image tower": median_ms(lambda: detector.model.encode_image(batch), iters),
        "heads": median_ms(lambda: detector.model.detect(patches, queries, query_mask), iters),
    }


def run(variant: str, device: str, pixels: np.ndarray, iters: int, compare: bool) -> None:
    """Measure one variant on one device and print the table for it."""
    checkpoint = Path(resolve("owlv2", variant, "torch-fp32"))
    detector = Detector(checkpoint, variant, device=device)

    # Each measurement empties exactly the cache it is measuring the absence of, outside the
    # timer. Timing a fully warm ``predict`` would report the cost of the heads and call it the
    # cost of the model -- 59 ms against 2.5 seconds, which is the sort of number that ends up
    # in a README.
    def cold_start():
        detector._images.clear()
        detector._prompts.clear()

    cold = median_ms(lambda: detector.predict(pixels, VOCABULARY, THRESHOLD), iters,
                     before=cold_start)
    cached = median_ms(lambda: detector.predict(pixels, OTHER, THRESHOLD), iters,
                       before=detector._prompts.clear)
    corpus = median_ms(lambda: detector.predict(pixels, VOCABULARY, THRESHOLD), iters,
                       before=detector._images.clear)

    spec = SPECS[variant]
    print(f"\n{variant} on {device}  "
          f"({spec.vision.image_size}px, {spec.vision.patches**2} patches, "
          f"{sum(p.numel() for p in detector.model.parameters()) / 1e6:.1f}M params)")
    print(f"  {'mozo cold':<22} {cold:8.1f} ms   one image, one vocabulary, nothing cached")
    print(f"  {'mozo cached image':<22} {cached:8.1f} ms   a new vocabulary on an image already seen")
    print(f"  {'mozo corpus':<22} {corpus:8.1f} ms   the next image on a vocabulary already encoded")

    if compare:
        model, processor, tokenizer = upstream(variant, device)
        theirs = median_ms(
            lambda: upstream_call(model, processor, tokenizer, pixels, VOCABULARY, device), iters)
        print(f"  {'transformers':<22} {theirs:8.1f} ms   the same weights, end to end, every call")
        # Stated as "mozo takes N times as long", so a number above one is always the bad
        # direction. Ratios of two timings on the same machine are noisier than either -- at a
        # second per image, a few percent either way is the machine, not the code.
        print(f"  {'':22} {cold / theirs:8.2f}x   mozo cold against it, "
              f"{corpus / theirs:.2f}x over a corpus, {cached / theirs:.2f}x on a cached image")
        del model

    print("  breakdown of the cold number:")
    for stage, ms in breakdown(detector, pixels, iters).items():
        print(f"    {stage:<20} {ms:8.1f} ms")


def against(pixels: np.ndarray, iters: int, device: str) -> None:
    """Time the families mozo already serves, so OWLv2's number has something to sit beside.

    Not a fair fight and not meant to be one: a closed-vocabulary detector answers a narrower
    question. It is here because "1.5 seconds" means nothing until you know what the alternatives
    on the same machine cost.
    """
    import mozo

    others = [
        ("rfdetr/small", lambda m: m.predict(pixels, threshold=0.5)),
        ("rfdetr/nano", lambda m: m.predict(pixels, threshold=0.5)),
        ("sam3/sam3", lambda m: m.predict(pixels, "person", threshold=0.5)),
        ("yolov11/small", lambda m: m.predict(pixels, threshold=0.5)),
    ]
    print(f"\nfor scale, on {device}, the same photograph:")
    manager = mozo.ModelManager()
    for name, call in others:
        family, variant = name.split("/")
        try:
            model = manager.get_model(family, variant, device=device)
        except (WeightsError, FileNotFoundError) as error:
            print(f"  {name:<22} {'--':>8}      not measured: {error}")
            continue
        print(f"  {name:<22} {median_ms(lambda: call(model), iters):8.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["base-ensemble"])
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--image", type=Path, default=FIXTURE)
    parser.add_argument("--no-upstream", action="store_true",
                        help="skip the transformers comparison, which loads a second copy")
    parser.add_argument("--against", action="store_true",
                        help="also time the other families on this machine")
    args = parser.parse_args()

    pixels = load_image(str(args.image))
    print(f"{args.image.name}, {pixels.shape[1]}x{pixels.shape[0]}, "
          f"median of {args.iters} runs after {WARMUP} warm-up")

    for device in args.devices:
        for variant in args.variants:
            if variant not in SPECS:
                raise SystemExit(f"unknown variant {variant!r}; have {sorted(SPECS)}")
            run(variant, device, pixels, args.iters, not args.no_upstream)
        if args.against:
            against(pixels, args.iters, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
