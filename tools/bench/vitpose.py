#!/usr/bin/env python3
"""Measure every published ViTPose artifact against upstream's own implementation.

Bootstrap tooling; never ships. It answers two questions: whether the extraction still agrees with
``transformers`` -- the implementation it was taken from -- and what it costs to run.

    python tools/bench/vitpose.py --images tests/fixtures/images
    python tools/bench/vitpose.py --images /path/to/photos --variants small --devices cpu

The baseline is the installed ``transformers`` package, run on the same photographs **and the same
boxes**. That last part is what makes the comparison about this model: a top-down pose estimator
answers a question about a box it was given, so giving the two sides different boxes would measure
the detector instead. The boxes come from ``rfdetr/medium``, which is not part of what is being
measured -- it is a fixed source of people.

Because the boxes are shared, joints correspond row for row and person for person. There is no
matching step and no IoU: the numbers reported are how far a joint moved, in the frame's own
pixels, and how far its confidence moved.

**Latency is reported per image and per person.** N boxes are N crops through one forward pass, so
per-image cost depends on how crowded the photograph is; per-person is the number that transfers.
Nothing here is a pass/fail gate; it produces the numbers ``PROVENANCE.md`` states.
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
# This file is called vitpose.py, so its own directory would shadow the package it benchmarks.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import DETECTOR, person_boxes  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import runnable  # noqa: E402
from mozo.vendors.vitpose_deploy import get_spec  # noqa: E402
from mozo.vendors.vitpose_deploy.predictor import EXPERT  # noqa: E402
from mozo.weights import WeightsError, artifacts  # noqa: E402

VARIANTS = ["small", "base", "large", "huge"]

#: mozo's variant name -> the Hugging Face repository the baseline loads. The same mapping
#: ``tools/fetch/vitpose.py`` publishes from; stated again here because this script must be able
#: to run against a checkpoint mozo has not published yet.
UPSTREAM_REPO = {variant: f"usyd-community/vitpose-plus-{variant}" for variant in VARIANTS}

#: Iterations discarded before timing starts. Generous because Metal needs it.
WARMUP = 5


def people(paths: list[Path], device: str) -> dict[str, np.ndarray]:
    """Person boxes per image, from the detector this bench holds fixed with the export gate."""
    boxes = {image.name: found for image, found in person_boxes(paths, device).items()}
    print(f"{DETECTOR[0]}/{DETECTOR[1]} found {sum(len(v) for v in boxes.values())} people "
          f"over {len(paths)} images\n")
    return boxes


def upstream_baseline(variant: str, paths: list[Path], arrays: list[np.ndarray],
                      boxes: dict[str, np.ndarray], device: str) -> dict[str, np.ndarray]:
    """Run ``transformers`` over the same images and boxes, returning ``(N, K, 3)`` per image."""
    import torch
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    repo = UPSTREAM_REPO[variant]
    processor = AutoImageProcessor.from_pretrained(repo)
    model = VitPoseForPoseEstimation.from_pretrained(repo).eval().to(device)

    results = {}
    for path, array in zip(paths, arrays):
        xyxy = boxes[path.name]
        if not len(xyxy):
            results[path.name] = np.zeros((0, get_spec(variant).keypoints, 3), dtype=np.float32)
            continue
        # Upstream takes COCO's (x, y, w, h); mozo takes xyxy everywhere.
        xywh = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in xyxy]
        inputs = processor(array, boxes=[xywh], return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(pixel_values=inputs["pixel_values"],
                            dataset_index=torch.full((len(xywh),), EXPERT, device=device))
        posed = processor.post_process_pose_estimation(outputs, boxes=[xywh])[0]
        results[path.name] = np.stack([
            np.concatenate([person["keypoints"].cpu().numpy(),
                            person["scores"].cpu().numpy()[:, None]], axis=-1) for person in posed
        ]) if posed else np.zeros((0, get_spec(variant).keypoints, 3), dtype=np.float32)
    del model
    return results


def compare(baseline: np.ndarray, ours: np.ndarray) -> dict:
    """How far the joints moved. Row for row: the two ran on the same boxes."""
    if baseline.shape != ours.shape:
        raise SystemExit(f"upstream returned {baseline.shape} and mozo {ours.shape}; "
                         "the two were not given the same boxes")
    if not len(baseline):
        return {"joints": 0, "worst_position": None, "worst_confidence": None}
    return {
        "joints": int(baseline.shape[0] * baseline.shape[1]),
        "worst_position": float(np.abs(baseline[..., :2] - ours[..., :2]).max()),
        "worst_confidence": float(np.abs(baseline[..., 2] - ours[..., 2]).max()),
    }


def measure(model, arrays: list[np.ndarray], detections: list, iters: int) -> float:
    """Return median per-image latency in milliseconds, after warm-up."""
    for _ in range(WARMUP):
        model.predict(arrays[0], detections[0])

    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        for array, found in zip(arrays, detections):
            model.predict(array, found)
        timings.append((time.perf_counter() - start) * 1000 / len(arrays))
    return statistics.median(timings)


def run(variant: str, runtime: str, device: str, paths: list[Path], arrays: list[np.ndarray],
        boxes: dict[str, np.ndarray], baseline: dict[str, np.ndarray], iters: int) -> dict:
    """Compare and time one (variant, runtime, device) combination."""
    import pixelflow as pf

    from mozo.adapters.vitpose import ViTPosePredictor

    model = ViTPosePredictor(variant, device=device, runtime=runtime)
    detections = [
        pf.detections.from_arrays(boxes=boxes[path.name],
                                  scores=[1.0] * len(boxes[path.name]))
        for path in paths
    ]

    agreement = []
    for path, array, found in zip(paths, arrays, detections):
        posed = model.predict(array, found)
        joints = np.array([[[joint.x, joint.y, joint.confidence] for joint in row.keypoints]
                           for row in posed], dtype=np.float64).reshape(len(posed), -1, 3)
        agreement.append(compare(baseline[path.name], joints))

    latency = measure(model, arrays, detections, iters)
    del model

    counted = sum(len(boxes[path.name]) for path in paths)
    return {
        "variant": variant, "runtime": runtime, "device": device,
        "images": len(paths),
        "people": counted,
        "joints": sum(a["joints"] for a in agreement),
        "worst_position": max((a["worst_position"] for a in agreement
                               if a["worst_position"] is not None), default=None),
        "worst_confidence": max((a["worst_confidence"] for a in agreement
                                 if a["worst_confidence"] is not None), default=None),
        "ms_per_image": latency,
        "ms_per_person": latency * len(paths) / counted if counted else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="directory of photographs")
    parser.add_argument("--limit", type=int, default=10, help="how many images to use")
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    parser.add_argument("--devices", nargs="*", default=["mps", "cpu"])
    parser.add_argument("--runtimes", nargs="*", default=None,
                        help="restrict to these artifact keys (default: everything runnable)")
    parser.add_argument("--baseline-device", default="cpu", help="where to run the upstream baseline")
    parser.add_argument("--iters", type=int, default=3, help="timed passes over the image set")
    parser.add_argument("--out", type=Path, default=ROOT / "bench" / "vitpose.json")
    args = parser.parse_args()

    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:args.limit]
    if len(paths) < args.limit:
        print(f"warning: only {len(paths)} images available", file=sys.stderr)
    arrays = [load_image(str(p)) for p in paths]
    print(f"{len(paths)} images, {args.iters} timed passes each\n")

    boxes = people(paths, args.baseline_device)

    rows = []
    for variant in args.variants:
        try:
            published = runnable(artifacts("vitpose", variant))
            if args.runtimes:
                published = [key for key in published if key in args.runtimes]
        except WeightsError:
            print(f"{variant}: not published, skipping")
            continue

        started = time.perf_counter()
        baseline = upstream_baseline(variant, paths, arrays, boxes, args.baseline_device)
        print(f"=== vitpose/{variant}: upstream posed "
              f"{sum(len(v) for v in baseline.values())} people "
              f"({time.perf_counter() - started:.0f}s)")

        for device in args.devices:
            for runtime in published:
                try:
                    row = run(variant, runtime, device, paths, arrays, boxes, baseline, args.iters)
                except Exception as error:  # noqa: BLE001 - a device that cannot run is a result
                    print(f"  {runtime:12s} {device:4s} unavailable: {error}")
                    continue
                rows.append(row)
                print(f"  {runtime:12s} {device:4s} "
                      f"{row['ms_per_image']:7.1f} ms/image  "
                      f"{row['ms_per_person']:6.1f} ms/person  "
                      f"worst joint {row['worst_position']:.2e} px  "
                      f"worst confidence {row['worst_confidence']:.2e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
