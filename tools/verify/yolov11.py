#!/usr/bin/env python3
"""Check that mozo returns exactly what the vendor does, for YOLO11.

Two paths reach the same weights. One is the vendored package used directly -- build a
``Detector``, hand it an image, read the boxes off it. The other is mozo: registry lookup, weights
resolution, adapter, runtime selection, PixelFlow result. Between them sit letterboxing, a runtime
choice, non-maximum suppression, a coordinate mapping and a result conversion, and any of those
could quietly change a number.

    python tools/verify/yolov11.py                        # fixtures, nano, every runtime
    python tools/verify/yolov11.py --variant small
    python tools/verify/yolov11.py photo.jpg other.jpg    # your own images

The comparison is exact for the torch runtime. Not "close": both paths run the same weights
through the same pre-processing and the same suppression, so every detection must agree exactly on
boxes, scores and class ids. A tolerance there would hide precisely the drift this script exists
to catch.

The vendor's side is quantised first, because mozo's is: PixelFlow truncates each box coordinate
to a whole pixel and rounds each score to three decimals, so those are the numbers a mozo user
actually receives and the only ones the two paths can be compared on. Comparing mozo's rounded
output against the vendor's full precision would report a difference on every box that is not
already an integer.

ONNX is held to one pixel rather than zero, and the reason is that truncation, not the executor.
``tools/export/yolov11.py`` already verified the graph against the torch model to a hundredth of a
pixel at full precision; a box sitting at 632.9997 on one side and 633.0002 on the other is well
inside that and still truncates to 632 against 633.

Exits non-zero on any disagreement, so it can be run as a gate rather than read.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Local weights unless the caller points somewhere else: this script is for checking a tree you
# just built, and silently reaching for the published bucket instead would verify the wrong bytes.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

sys.path.insert(0, str(ROOT / "tests"))

import mozo  # noqa: E402
from conftest import as_pixelflow_reports  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.yolov11_deploy import Detector  # noqa: E402
from mozo.weights import artifacts, resolve  # noqa: E402

#: Serving threshold. Low enough that the comparison covers marginal detections, where two
#: implementations diverge first, rather than only the confident ones everything agrees on.
THRESHOLD = 0.05

#: What the ONNX pairing is allowed to move once quantised. One whole pixel, because a difference
#: far below a pixel can still land either side of a truncation boundary; the exporter is what
#: holds the graph itself to a hundredth of a pixel, and it does that before quantising.
ONNX_BOX_TOLERANCE = 1.0
ONNX_SCORE_TOLERANCE = 1e-3


def vendor_detections(variant: str, images: dict) -> dict[str, tuple]:
    """Run the vendored package directly, with no mozo machinery between it and the weights."""
    detector = Detector(resolve("yolov11", variant, "torch-fp32"), device="cpu")
    found = {}
    for name, pixels in images.items():
        result = detector.predict(pixels, conf=THRESHOLD)
        boxes, scores = as_pixelflow_reports(result.boxes, result.scores)
        found[name] = (boxes, scores, result.class_ids, result.names)
    return found


def mozo_detections(variant: str, images: dict, runtime: str) -> dict[str, tuple]:
    """Run the same weights the way a user of mozo would.

    Through :class:`~mozo.ModelManager` rather than :func:`mozo.get_model`, because that is the
    path the server takes and the only one that accepts a runtime.
    """
    model = mozo.ModelManager().get_model("yolov11", variant, device="cpu", runtime=runtime)
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


def compare(name: str, want: tuple, got: tuple, box_tol: float, score_tol: float) -> tuple[list, str]:
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


def report(label: str, vendor: dict, other: dict, box_tol: float, score_tol: float) -> list[str]:
    """Compare every image and print a line each, returning everything that disagreed."""
    print(f"\n{label}")
    problems = []
    for name, want in vendor.items():
        found, detail = compare(name, want, other[name], box_tol, score_tol)
        problems += found
        print(f"  {'FAIL' if found else 'ok':<4} {name:<24} {len(want[0]):>4} detections   {detail}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="*", type=Path, help="images to test (default: tests/fixtures/images)")
    parser.add_argument("--variant", default="nano")
    args = parser.parse_args()

    images = args.images or sorted((ROOT / "tests" / "fixtures" / "images").glob("*.jpg"))
    missing = [image for image in images if not image.is_file()]
    if missing:
        raise SystemExit(f"no such image: {missing}")
    if not images:
        raise SystemExit("no images to test")

    print(f"yolov11/{args.variant}, {len(images)} image(s), threshold {THRESHOLD}")
    published = artifacts("yolov11", args.variant)
    print(f"published artifacts: {', '.join(published)}")

    # Decoded once and shared by every run below, so what is compared is the models rather than
    # three trips through the JPEG decoder.
    decoded = {path.name: load_image(str(path)) for path in images}
    vendor = vendor_detections(args.variant, decoded)
    problems = report(
        "vendor direct  vs  mozo torch-fp32   (must be exact)",
        vendor, mozo_detections(args.variant, decoded, "torch-fp32"), 0.0, 0.0)

    if "onnx-fp32" in published:
        problems += report(
            f"vendor direct  vs  mozo onnx-fp32    (tolerance {ONNX_BOX_TOLERANCE} px)",
            vendor, mozo_detections(args.variant, decoded, "onnx-fp32"),
            ONNX_BOX_TOLERANCE, ONNX_SCORE_TOLERANCE)
    else:
        print("\nno onnx-fp32 published for this variant; skipping that pairing")

    if problems:
        print("\nDISAGREEMENT:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("\nidentical on every image and every runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
