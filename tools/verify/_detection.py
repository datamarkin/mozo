#!/usr/bin/env python3
"""Check that mozo returns exactly what a detection vendor does.

Two paths reach the same weights. One is the vendored package used directly -- build a
``Detector``, hand it an image, read the boxes off it. The other is mozo: registry lookup, weights
resolution, adapter, runtime selection, PixelFlow result. Between them sit letterboxing, a runtime
choice, non-maximum suppression, a coordinate mapping and a result conversion, and any of those
could quietly change a number.

The comparison is exact for the torch runtime. Not "close": both paths run the same weights
through the same pre-processing and the same suppression, so every detection must agree exactly on
boxes, scores and class ids. A tolerance there would hide precisely the drift this exists to catch.

The vendor's side is quantised first, because mozo's is: PixelFlow truncates each box coordinate
to a whole pixel and rounds each score to three decimals, so those are the numbers a mozo user
actually receives and the only ones the two paths can be compared on. Comparing mozo's rounded
output against the vendor's full precision would report a difference on every box that is not
already an integer.

Graph runtimes are held to one pixel rather than zero, and the reason is that truncation, not the
executor. ``tools/export/*`` already verified each graph against the torch model to a hundredth of
a pixel at full precision; a box sitting at 632.9997 on one side and 633.0002 on the other is well
inside that and still truncates to 632 against 633.

This module is shared rather than copied per family, unlike the vendors it checks. It is a *gate*:
a second copy that nobody updated keeps exiting zero while checking something older than what it
is guarding, which is the one failure a gate must not have.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Done at import rather than in a function the scripts must remember to call between two other
# imports. ``conftest`` lives in tests/ because the quantisation rule it holds is a fact about
# mozo's result boundary, not about any tool. Local weights unless the caller says otherwise:
# these scripts check a tree you just built, and reaching for the published bucket would verify
# the wrong bytes.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

import numpy as np  # noqa: E402

import mozo  # noqa: E402
from conftest import as_pixelflow_reports  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import executable  # noqa: E402
from mozo.weights import artifacts, resolve  # noqa: E402

#: Serving threshold. Low enough that the comparison covers marginal detections, where two
#: implementations diverge first, rather than only the confident ones everything agrees on.
THRESHOLD = 0.05

#: What a graph pairing is allowed to move once quantised. One whole pixel, because a difference
#: far below a pixel can still land either side of a truncation boundary; the exporter is what
#: holds the graph itself to a hundredth of a pixel, and it does that before quantising.
BOX_TOLERANCE = 1.0
SCORE_TOLERANCE = 1e-3


def _vendor_detections(detector_class, family: str, variant: str, images: dict) -> dict:
    """Run the vendored package directly, with no mozo machinery between it and the weights."""
    detector = detector_class(resolve(family, variant, "torch-fp32"), device="cpu")
    found = {}
    for name, pixels in images.items():
        result = detector.predict(pixels, conf=THRESHOLD)
        boxes, scores = as_pixelflow_reports(result.boxes, result.scores)
        found[name] = (boxes, scores, result.class_ids, result.names)
    return found


def _mozo_detections(family: str, variant: str, images: dict, runtime: str) -> dict:
    """Run the same weights the way a user of mozo would.

    Through :class:`~mozo.ModelManager` rather than :func:`mozo.get_model`, because that is the
    path the server takes and the only one that accepts a runtime.
    """
    model = mozo.ModelManager().get_model(family, variant, device="cpu", runtime=runtime)
    found = {}
    for name, pixels in images.items():
        rows = model.predict(pixels, threshold=THRESHOLD).to_dict()
        found[name] = (
            np.array([row["bbox"] for row in rows], dtype=np.float64).reshape(-1, 4),
            np.array([row["confidence"] for row in rows], dtype=np.float64),
            np.array([row["class_id"] for row in rows], dtype=np.int64),
            [row["class_name"] for row in rows],
        )
    return found


def _compare(name: str, want: tuple, got: tuple, box_tol: float, score_tol: float) -> tuple[list, str]:
    """Return the ways two results disagree, and a one-line summary of how far apart they are."""
    want_boxes, want_scores, want_ids, want_names = want
    got_boxes, got_scores, got_ids, got_names = got

    if len(want_boxes) != len(got_boxes):
        return [f"{name}: {len(want_boxes)} detections one side, {len(got_boxes)} the other"], "counts differ"
    if not len(want_boxes):
        return [], "no detections"

    box_error = float(np.abs(want_boxes - got_boxes).max())
    score_error = float(np.abs(want_scores - got_scores).max())
    problems = []
    if not np.array_equal(want_ids, got_ids):
        problems.append(f"{name}: class ids differ")
    if want_names != got_names:
        problems.append(f"{name}: class names differ")
    if box_error > box_tol:
        problems.append(f"{name}: boxes differ by {box_error:g} px")
    if score_error > score_tol:
        problems.append(f"{name}: scores differ by {score_error:g}")
    return problems, f"boxes {box_error:g} px, scores {score_error:g}"


def _report(label: str, vendor: dict, other: dict, box_tol: float, score_tol: float) -> list[str]:
    """Compare every image and print a line each, returning everything that disagreed."""
    print(f"\n{label}")
    problems = []
    for name, want in vendor.items():
        found, detail = _compare(name, want, other[name], box_tol, score_tol)
        problems += found
        print(f"  {'FAIL' if found else 'ok':<4} {name:<24} {len(want[0]):>4} detections   {detail}")
    return problems


def run(family: str, description: str = "") -> int:
    """Compare *family*'s vendor against mozo over some images. Non-zero on any disagreement.

    The vendor is found from the family name, which is the same string everywhere -- the manifest
    key, the weights directory and ``mozo.vendors.<family>_deploy`` -- so a family script needs to
    say it once rather than also importing a class.

    Every runtime the variant publishes *and this machine can execute* is exercised. A family that
    publishes a graph mozo will not run here needs no special case, and neither does a host with
    no coremltools: :func:`~mozo.runtimes.executable` already answers both, and asking it is what
    keeps this runnable as a gate on a machine that is not a Mac.
    """
    detector_class = importlib.import_module(f"mozo.vendors.{family}_deploy").Detector
    parser = argparse.ArgumentParser(
        description=description or f"verify {family}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path, help="images to test (default: fixtures)")
    parser.add_argument("--variant", default="nano")
    args = parser.parse_args()

    images = args.images or sorted((ROOT / "tests" / "fixtures" / "images").glob("*.jpg"))
    missing = [image for image in images if not image.is_file()]
    if missing:
        raise SystemExit(f"no such image: {missing}")
    if not images:
        raise SystemExit("no images to test")

    print(f"{family}/{args.variant}, {len(images)} image(s), threshold {THRESHOLD}")
    published = artifacts(family, args.variant)
    print(f"published artifacts: {', '.join(published)}")

    # Decoded once and shared by every run below, so what is compared is the models rather than
    # several trips through the JPEG decoder.
    decoded = {path.name: load_image(str(path)) for path in images}
    vendor = _vendor_detections(detector_class, family, args.variant, decoded)
    problems = _report(
        "vendor direct  vs  mozo torch-fp32   (must be exact)",
        vendor, _mozo_detections(family, args.variant, decoded, "torch-fp32"), 0.0, 0.0)

    for runtime in executable(published):
        if runtime.startswith("torch"):
            continue
        problems += _report(
            f"vendor direct  vs  mozo {runtime:<12} (tolerance {BOX_TOLERANCE} px)",
            vendor, _mozo_detections(family, args.variant, decoded, runtime),
            BOX_TOLERANCE, SCORE_TOLERANCE)

    if problems:
        print("\nDISAGREEMENT:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("\nidentical on every image and every runtime")
    return 0
