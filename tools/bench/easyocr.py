#!/usr/bin/env python3
"""Measure what EasyOCR costs to serve, against the package it was extracted from.

Bootstrap tooling; never ships. ``tools/verify/easyocr.py`` answers whether mozo returns the same
strings as ``easyocr``; this answers what it costs to get them, and whether the split between
detection and recognition is worth the seam it introduces.

    python tools/bench/easyocr.py
    python tools/bench/easyocr.py --variants english korean --devices cpu mps

**The comparison is legitimate because both sides run the same weights on the same device.**
The same two checkpoints, the same images, the same torch. Upstream selects a device too --
its ``gpu`` argument is a selector rather than a boolean about CUDA, and it resolves mps on
Apple silicon -- so timing mozo on a GPU against upstream on a CPU would measure the two
devices and report the gap as mozo's. Both are timed end to end, from an
``HxWx3`` array to located strings, because that is the work a caller asks for. Upstream is timed
through ``detect`` and ``recognize`` on the same two arrays the vendor gets, for the same reason
the gate does it that way: timing ``readtext`` on a path would measure two different JPEG
decoders as well as two implementations.

The upstream side is built by ``tools/verify/easyocr.py``'s own ``upstream_reader``, imported
rather than restated. The thing being timed has to be the thing that was verified, and that
function is where the two pins live that make it so: ``quantize=False``, because upstream's
default is ``True`` and then swallows the failure, so on a machine with a quantization engine
this would time a qint8 recogniser against an fp32 one and call the difference mozo's; and the
language list that selects which checkpoint upstream loads at all.

Two numbers are reported per variant and device:

``read``       the whole set -- what serving a page costs.
``detect``     detection alone, so the split of that number has a denominator.

Nothing is cached, so there is no warm number to report. The reader holds no state between
calls: one image has one answer, and a cache would hash every pixel of every request to miss
every time.

Nothing here is a pass/fail gate. It produces the numbers a support matrix is built from.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

# Named after the package it compares against, like every other tool for this family, so its own
# directory has to come off the path before ``import easyocr`` can find the installed one.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if entry and Path(entry).resolve() != _HERE]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

from conftest import TEXT_FIXTURES  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.easyocr_deploy import SPECS, VARIANTS, Reader  # noqa: E402
from mozo.weights import resolve  # noqa: E402
from verify.easyocr import upstream_reader  # noqa: E402


def median_ms(work, iters: int) -> float:
    """Median of *iters* runs, in milliseconds, with one untimed warm-up."""
    work()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        work()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def pages() -> list[np.ndarray]:
    """The fixtures with text on them, as RGB arrays. The blank one is not a workload."""
    return [load_image(p) for p in sorted(TEXT_FIXTURES.glob("*.png"))
            if p.name != "blank.png"]


def run(variant: str, device: str, iters: int, compare: bool) -> None:
    reader = Reader(resolve("easyocr", variant, "torch-fp32"), SPECS[variant], device=device)
    workload = pages()
    print(f"\n{variant} on {device}  ({len(workload)} pages, median of {iters})")

    def whole() -> None:
        for rgb in workload:
            reader(rgb)

    def detect_only() -> None:
        for rgb in workload:
            reader.detect(rgb)

    total = median_ms(whole, iters)
    detect = median_ms(detect_only, iters)

    print(f"  {'mozo read':<22} {total:8.1f} ms   ({total / len(workload):.1f} ms/page)")
    print(f"  {'mozo detect only':<22} {detect:8.1f} ms   "
          f"({100 * detect / total:.0f}% of it)")

    if not compare:
        return

    try:
        up = upstream_reader(variant, device)
    except SystemExit as missing:
        print(f"  upstream               {missing}")
        return

    greys = [(rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)) for rgb in workload]

    def upstream() -> None:
        for rgb, grey in greys:
            h, f = up.detect(rgb)
            up.recognize(grey, h[0], f[0], reformat=False, allowlist=up.character)

    theirs = median_ms(upstream, iters)
    print(f"  {'upstream':<22} {theirs:8.1f} ms   "
          f"(mozo is {theirs / total:.2f}x its speed)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["english"],
                        help=f"default: english. Known: {VARIANTS}")
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--no-compare", action="store_true",
                        help="skip the upstream comparison")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    for variant in args.variants:
        if variant not in SPECS:
            raise SystemExit(f"unknown variant {variant!r}. Known: {VARIANTS}")
        for device in args.devices:
            run(variant, device, args.iters, not args.no_compare)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
