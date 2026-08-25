#!/usr/bin/env python3
"""Measure every published RF-DETR artifact against upstream's own implementation.

Bootstrap tooling; never ships. It answers two questions that the export-time check cannot:
whether an artifact still agrees with the *upstream library* (not merely with the torch model we
exported it from), and what it costs to run.

    python tools/bench/rfdetr.py --images /path/to/photos
    python tools/bench/rfdetr.py --images /path/to/photos --variants small --devices mps

The baseline is the installed ``rfdetr`` package, run on the same photographs. Agreement is
reported at the level a user sees -- how many detections match, how well the boxes line up, how
far the scores moved -- because raw tensor deltas say nothing about whether a result changed.
Detections are matched by best IoU rather than by position, since two runtimes may legitimately
order equal-scoring queries differently.

Latency is measured per image after warm-up, and reported as the median over ``--iters`` runs.
Nothing here is a pass/fail gate; it produces the numbers a support matrix is built from.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
# This file is called rfdetr.py, so its own directory would shadow the package it benchmarks.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

from mozo.image import load_image  # noqa: E402
from mozo.runtimes import runnable  # noqa: E402
from mozo.weights import WeightsError, artifacts  # noqa: E402

VARIANTS = ["nano", "small", "medium", "large", "seg-nano", "seg-small", "seg-medium", "seg-large",
            "keypoint-preview"]

#: Upstream exposes each variant as its own class.
UPSTREAM_CLASS = {
    "nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge",
    "seg-nano": "RFDETRSegNano", "seg-small": "RFDETRSegSmall",
    "seg-medium": "RFDETRSegMedium", "seg-large": "RFDETRSegLarge",
    "keypoint-preview": "RFDETRKeypointPreview",
}

THRESHOLD = 0.5

#: A pair of boxes this close is the same detection seen twice, not two findings.
MATCH_IOU = 0.5

#: Iterations discarded before timing starts. See :func:`measure`.
WARMUP = 12


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two xyxy boxes."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if overlap <= 0:
        return 0.0
    areas = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return float(overlap / (areas - overlap))


def compare(baseline: list[dict], ours: list[dict]) -> dict:
    """Match two images' detections by best IoU and describe how far apart they are.

    Returns:
        Counts of matched and unmatched detections, the worst IoU among matches, the worst score
        movement, and how often the matched pair disagreed on the class.
    """
    unmatched = list(range(len(ours)))
    ious, score_deltas, kp_deltas, class_flips = [], [], [], 0

    for want in baseline:
        best, best_iou = None, 0.0
        for index in unmatched:
            value = _iou(want["box"], ours[index]["box"])
            if value > best_iou:
                best, best_iou = index, value
        if best is None or best_iou < MATCH_IOU:
            continue
        got = ours[best]
        unmatched.remove(best)
        ious.append(best_iou)
        score_deltas.append(abs(want["score"] - got["score"]))
        class_flips += int(want["class_id"] != got["class_id"])
        # Compared only for a pair that already matched by box: keypoints belonging to two
        # different people are not a delta, they are a mismatch, and the IoU test above is what
        # rules that out. One side is enough to test -- a run compares a variant against itself,
        # so either both containers carry joints or neither does.
        if want["keypoints"] is not None:
            kp_deltas.append(float(np.abs(want["keypoints"] - got["keypoints"]).max()))

    return {
        "baseline": len(baseline),
        "ours": len(ours),
        "matched": len(ious),
        "missed": len(baseline) - len(ious),
        "extra": len(unmatched),
        "worst_iou": min(ious) if ious else None,
        "worst_score_delta": max(score_deltas) if score_deltas else None,
        "worst_keypoint_delta": max(kp_deltas) if kp_deltas else None,
        "class_flips": class_flips,
    }


def _as_records(detections) -> list[dict]:
    """Normalise a PixelFlow ``Detections`` into plain comparable records.

    ``keypoints`` is ``(K, 3)`` as ``(x, y, confidence)`` for a keypoint variant and ``None`` for
    every other, which is what the models themselves return -- a detection variant is not a
    keypoint variant with its joints missing.
    """
    return [{"box": np.asarray(d.bbox, dtype=float), "score": float(d.confidence),
             "class_id": int(d.class_id),
             "keypoints": None if not d.keypoints else np.array(
                 [[k.x, k.y, k.confidence] for k in d.keypoints], dtype=float)}
            for d in detections]


def upstream_baseline(variant: str, images: list[Path], device: str) -> dict[str, list[dict]]:
    """Run the installed ``rfdetr`` package and return its detections per image.

    The device only affects how long this takes: upstream returns identical detections on CPU and
    MPS (measured: same counts, worst score delta 0.00000), while MPS is 2.4x faster. So the
    baseline is computed once, on whatever is quickest, and reused for every comparison.
    """
    import rfdetr
    from PIL import Image

    model = getattr(rfdetr, UPSTREAM_CLASS[variant])(device=device)
    results = {}
    for path in images:
        out = model.predict(Image.open(path).convert("RGB"), threshold=THRESHOLD)
        boxes, scores, joints = _unpack(out)
        results[path.name] = [
            {"box": np.asarray(box, dtype=float), "score": float(score), "class_id": int(cls),
             "keypoints": None if joints is None else joints[i]}
            for i, (box, score, cls) in enumerate(zip(boxes, scores, out.class_id))
        ]
    del model
    return results


def _unpack(out) -> tuple:
    """Return upstream's boxes, object scores and joints, whichever container it answered in.

    Read off the object rather than looked up by variant name. A keypoint model answers with an
    ``sv.KeyPoints``, which spells two of these fields differently -- boxes move into
    ``.data["xyxy"]``, and ``.confidence`` becomes per-*joint* so the object score is
    ``.detection_confidence``. Reading the detection container's names off one silently compares
    the wrong numbers, and the difference is a property of what came back, not of which name was
    asked for: a table of keypoint variants would be a third place to update when upstream
    publishes the rest of the curve, and the one nobody would think to.

    Joints are stacked once per image into ``(N, K, 3)`` -- the shape mozo's side already
    produces -- rather than per detection, and are ``None`` for a container that has none.
    """
    if not hasattr(out, "xy"):
        return out.xyxy, out.confidence, None
    joints = np.concatenate(
        [np.asarray(out.xy, dtype=float),
         np.asarray(out.keypoint_confidence, dtype=float)[..., None]], axis=-1)
    return out.data["xyxy"], out.detection_confidence, joints


def measure(model, images: list[np.ndarray], iters: int) -> float:
    """Return median per-image latency in milliseconds, after warm-up.

    Times an already-built model rather than building its own: loading a second copy leaves the
    first one's memory in flight.

    Warm-up is generous because Metal needs it. Three iterations was enough for the small
    variants and left the larger ones reading ~80% high; the number must not depend on how much
    work happened to run before this call.
    """
    for _ in range(WARMUP):
        model.predict(images[0], threshold=THRESHOLD)

    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        for image in images:
            model.predict(image, threshold=THRESHOLD)
        timings.append((time.perf_counter() - start) * 1000 / len(images))
    return statistics.median(timings)


def run(variant: str, runtime: str, device: str, paths: list[Path], arrays: list[np.ndarray],
        baseline: dict, iters: int) -> dict:
    """Compare and time one (variant, runtime, device) combination."""
    from mozo.adapters.rfdetr import RFDETRPredictor

    model = RFDETRPredictor(variant, device=device, runtime=runtime)
    agreement = [compare(baseline[path.name], _as_records(model.predict(array, threshold=THRESHOLD)))
                 for path, array in zip(paths, arrays)]
    latency = measure(model, arrays, iters)
    del model

    matched = sum(a["matched"] for a in agreement)
    expected = sum(a["baseline"] for a in agreement)
    worst_ious = [a["worst_iou"] for a in agreement if a["worst_iou"] is not None]

    return {
        "variant": variant, "runtime": runtime, "device": device,
        "images": len(paths),
        "baseline_detections": expected,
        "our_detections": sum(a["ours"] for a in agreement),
        "matched": matched,
        "recall": matched / expected if expected else 1.0,
        "missed": sum(a["missed"] for a in agreement),
        "extra": sum(a["extra"] for a in agreement),
        "class_flips": sum(a["class_flips"] for a in agreement),
        "worst_iou": min(worst_ious) if worst_ious else None,
        "worst_score_delta": max((a["worst_score_delta"] for a in agreement
                                  if a["worst_score_delta"] is not None), default=None),
        "worst_keypoint_delta": max((a["worst_keypoint_delta"] for a in agreement
                                     if a["worst_keypoint_delta"] is not None), default=None),
        "ms": latency,
        "fps": 1000.0 / latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="directory of photographs")
    parser.add_argument("--limit", type=int, default=10, help="how many images to use")
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    parser.add_argument("--devices", nargs="*", default=["mps", "cpu"])
    parser.add_argument("--runtimes", nargs="*", default=None,
                        help="restrict to these artifact keys (default: everything runnable)")
    parser.add_argument("--baseline-device", default="mps", help="where to run the upstream baseline")
    parser.add_argument("--iters", type=int, default=3, help="timed passes over the image set")
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "rfdetr.json")
    args = parser.parse_args()

    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:args.limit]
    if len(paths) < args.limit:
        print(f"warning: only {len(paths)} images available", file=sys.stderr)
    # Our model takes mozo's contract (RGB); the upstream baseline reads the same files
    # itself through PIL, so the two never share an array.
    arrays = [load_image(str(p)) for p in paths]
    print(f"{len(paths)} images, {args.iters} timed passes each\n")

    rows = []
    for variant in args.variants:
        try:
            published = runnable(artifacts("rfdetr", variant))
            if args.runtimes:
                published = [key for key in published if key in args.runtimes]
        except WeightsError:
            print(f"{variant}: not published, skipping")
            continue

        started = time.perf_counter()
        baseline = upstream_baseline(variant, paths, args.baseline_device)
        print(f"=== rfdetr/{variant}: upstream found {sum(len(v) for v in baseline.values())} "
              f"detections over {len(paths)} images ({time.perf_counter() - started:.0f}s)")

        for device in args.devices:
            for runtime in published:
                try:
                    row = run(variant, runtime, device, paths, arrays, baseline, args.iters)
                except Exception as error:  # a runtime that cannot run here is a result, not a crash
                    print(f"    {device:4} {runtime:11} FAILED  {type(error).__name__}: {str(error)[:80]}")
                    rows.append({"variant": variant, "runtime": runtime, "device": device,
                                 "error": f"{type(error).__name__}: {error}"})
                    continue
                rows.append(row)
                iou = f"{row['worst_iou']:.3f}" if row["worst_iou"] is not None else "n/a"
                delta = f"{row['worst_score_delta']:.4f}" if row["worst_score_delta"] is not None else "n/a"
                kp = (f"  kp Δ {row['worst_keypoint_delta']:.4f}"
                      if row["worst_keypoint_delta"] is not None else "")
                print(f"    {device:4} {runtime:11} {row['matched']:3}/{row['baseline_detections']:<3} matched  "
                      f"worst IoU {iou}  score Δ {delta}{kp}  "
                      f"{row['ms']:7.1f} ms  {row['fps']:5.1f} fps")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
