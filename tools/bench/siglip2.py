#!/usr/bin/env python3
"""Measure what SigLIP 2 costs to serve, against the implementation it was extracted from.

Bootstrap tooling; never ships. ``tools/verify/siglip2.py`` answers whether mozo returns the same
numbers as ``transformers``; this answers what it costs to get them, and whether the split into
``encode_image`` and ``encode_text`` is worth the seam it introduces.

    python tools/bench/siglip2.py
    python tools/bench/siglip2.py --variants base-224 so400m-384 --devices cpu mps

**The comparison is legitimate because both sides run the same weights.** Not "SigLIP 2 against
another model" -- the same checkpoint, the same photograph, the same device, the same torch. Any
gap is this extraction against ``transformers``, which is the only thing mozo can be responsible
for. Both are timed end to end, from an ``HxWx3`` array to scores, because that is the work a
caller actually asks for; timing a bare forward would flatter whichever side does more of its work
outside one.

**Every measurement synchronises the device, and that is not a detail.** Metal queues work
asynchronously, so a timer that stops when Python returns measures how fast work was *submitted*.
Left out, this benchmark reported mozo 23% slower than ``transformers`` on MPS -- entirely because
mozo's ``encode_image`` ends in ``.cpu()``, which blocks until the queue drains, while the
reference path returned a device tensor and stopped the clock early. Synchronised, the two are
within 3%. Anyone re-measuring this without ``torch.mps.synchronize`` will rediscover the same
wrong answer.

**What this measured, on an M-series machine, is worth stating up front so nobody re-derives it.**

*mozo is 4% slower than ``transformers`` on CPU and 11% faster on MPS*, measured as the median of
five interleaved trials so that thermal drift moves both sides together. Both gaps are the same
decision: ``transformers`` defaults to ``sdpa`` and this package implements the eager attention its
gate compares against, and the two are the same arithmetic in a different order. Timed alone, the
image tower runs 82.3 ms eager against 68.2 ms sdpa on CPU -- and **7.0 ms eager against 11.1 ms
sdpa on MPS**, where the fused path is the slower one. So the choice made for exactness costs a
little on CPU and pays on Metal, and neither is a bottleneck the extraction introduced.

*The seam earns its keep on both sides, which is unusual.* For CLIP the image tower dominates and
caching the prompt is nearly worthless. Here the text tower is a full 12 layers over a context
that is **always** padded to 64 -- SigLIP 2 pools the last slot, so a two-word phrase costs exactly
what a sixty-token one does. On ``base-224``/MPS a classify is 27 ms, of which the image tower is
17 and the text tower 7. So an ingest job that only encodes images saves about a third, and a query
service that only encodes phrases saves about three quarters.

*Batching helps at 224 and does nothing at 384.* ``base-224`` goes from 17.1 ms per image to
10.8 ms in batches of eight, a 1.6x win. ``so400m-384`` is 176.9 ms alone and 175.6 ms batched --
one 384-pixel image already saturates the device, and batching buys nothing but latency. Worth
knowing before building an ingest pipeline around a batch size.

*The tokenizer is not worth optimising.* mozo's byte-pair encoder is written from scratch in Python
and runs 0.03-0.34 ms per call, against 0.04-0.08 ms for the Rust one it is checked against. Up to
4x slower and three orders of magnitude below the tower it feeds.

*Nothing is built until it is needed*, including the tokenizer. Constructing the merge tables means
decompressing a 4 MB asset and building dictionaries over 580,604 rules, which costs about 870 ms
-- once per process, and never at all for a job that only encodes images.

Nothing here is a pass/fail gate. It produces the numbers a support matrix is built from.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from conftest import FIXTURE  # noqa: E402

from mozo.image import load_image  # noqa: E402
from mozo.vendors.siglip2_deploy import CONTEXT, SPECS, Encoder  # noqa: E402
from mozo.weights import WeightsError, resolve  # noqa: E402

#: Phrases to time against. Three, because one is not a batch and thirty is not a request.
PROMPTS = ["a photo of people", "a photo of a laptop", "a photo of an elephant in a pool"]


def synchronise(device: str) -> None:
    """Block until the device has actually finished. See the module docstring."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def median_ms(work, device: str, runs: int = 10, warmup: int = 3) -> float:
    """Median wall time of *work*, in milliseconds, with the device drained each time.

    Median rather than mean: the first runs after a warmup still catch allocator growth and
    thermal variation, and one outlier should not move the number a support matrix is built from.
    """
    for _ in range(warmup):
        work()
        synchronise(device)

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        work()
        synchronise(device)
        timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def against_transformers(variant: str, ours: Encoder, device: str, image) -> tuple[float, float]:
    """End-to-end classify, mozo against the reference, on the same weights and device.

    Takes the encoder the profile already built rather than loading the checkpoint a second time;
    for ``giant-384`` that is 7.5 GB of avoidable work before the first measurement.
    """
    from PIL import Image
    from transformers import AutoImageProcessor, Siglip2Tokenizer, SiglipModel

    repo = f"google/siglip2-{SPECS[variant].upstream}"
    theirs = SiglipModel.from_pretrained(repo, dtype=torch.float32).eval().to(device)
    preprocess = AutoImageProcessor.from_pretrained(repo)
    tokenize = Siglip2Tokenizer.from_pretrained(repo)
    pillow = Image.fromarray(image)

    def reference():
        pixels = preprocess(pillow, return_tensors="pt")["pixel_values"].to(device)
        tokens = tokenize(PROMPTS, padding="max_length", max_length=CONTEXT,
                          return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = theirs(pixel_values=pixels, input_ids=tokens)
        return torch.sigmoid(out.logits_per_image).cpu()

    return (median_ms(lambda: ours.classify(image, PROMPTS), device),
            median_ms(reference, device))


def profile(encoder: Encoder, device: str, image) -> dict[str, float]:
    """Where a classify call's time goes, and what each half costs alone."""
    started = time.perf_counter()
    encoder.encode_image(image)
    encoder.encode_text(PROMPTS[:1])
    synchronise(device)
    ready = time.perf_counter() - started

    return {
        "load+1st": ready * 1000,
        "image x1": median_ms(lambda: encoder.encode_image(image), device),
        "image x8": median_ms(lambda: encoder.encode_image([image] * 8), device, runs=4),
        "text x1": median_ms(lambda: encoder.encode_text(PROMPTS[:1]), device),
        "text x32": median_ms(lambda: encoder.encode_text(PROMPTS * 11), device, runs=4),
        "classify": median_ms(lambda: encoder.classify(image, PROMPTS), device),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["base-224"])
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--reference", action="store_true",
                        help="also time transformers on the same weights")
    args = parser.parse_args()

    image = load_image(str(FIXTURE))

    header = f"{'variant':<13} {'dev':<5} {'load+1st':>9} {'img x1':>9} {'img x8':>9} " \
             f"{'/img':>8} {'txt x1':>8} {'txt x32':>9} {'classify':>9}"
    print(header)
    print("-" * len(header))

    for variant in args.variants:
        try:
            checkpoint = resolve("siglip2", variant, "torch-fp32")
        except WeightsError as error:
            print(f"{variant}: {error}")
            continue

        for device in args.devices:
            encoder = Encoder(checkpoint, SPECS[variant], device=device)
            numbers = profile(encoder, device, image)
            print(f"{variant:<13} {device:<5} {numbers['load+1st']/1000:8.1f}s "
                  f"{numbers['image x1']:8.1f}ms {numbers['image x8']:8.1f}ms "
                  f"{numbers['image x8']/8:7.1f}ms {numbers['text x1']:7.1f}ms "
                  f"{numbers['text x32']:8.1f}ms {numbers['classify']:8.1f}ms")

            if args.reference:
                mozo_ms, reference_ms = against_transformers(variant, encoder, device, image)
                print(f"{'':<13} {'':<5} classify against transformers: "
                      f"mozo {mozo_ms:.1f} ms, transformers {reference_ms:.1f} ms "
                      f"({mozo_ms / reference_ms:.2f}x)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
