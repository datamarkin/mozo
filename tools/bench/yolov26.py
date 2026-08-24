#!/usr/bin/env python3
"""What YOLO26 costs, against the implementation its checkpoints come from.

    python tools/bench/yolov26.py
    python tools/bench/yolov26.py --variants nano small --devices cpu mps
    python tools/bench/yolov26.py --images /path/to/photos --iters 50

Bootstrap tooling; never ships. `tools/verify/yolov26_reference.py` answers whether the vendored
package agrees with `ultralytics`; this answers what it costs, on the same weights, the same
photograph and the same device, so the number describes the extraction rather than the model.

**The reference model is built by that script, not by this one.** `reference_model` carries the
version pin, the `end2end` assertion and the checkpoint-suffix workaround, and importing it is
what keeps the thing being timed the thing that was verified -- the same reason
`tools/bench/easyocr.py` imports its reader from `tools/verify/easyocr.py`. A private copy here
had already lost the pin and the assertion and hardcoded `imgsz` to 640, which under 8.3.222
would have timed the one-to-many head against the vendor's full decode and printed the ratio as
a speed-up.

Four things are timed rather than one. The forward pass is what a graph runtime would replace; the
letterbox and the coordinate mapping are what it would not, and a breakdown is what lets a
regression be placed. This family also carries its **selection inside the network** -- the anchor
grid, the box decode and a two-stage top-k are part of the forward pass, where the siblings do
non-maximum suppression outside it -- so the share the head takes is the number worth having, and
no sibling bench reports it.

Two measurement traps, both inherited from `vendoring.md` §8 and both real:

- **Synchronise the device.** Metal queues asynchronously, so an unsynchronised timer measures
  submission and not work. SigLIP 2 first reported itself 23% slower on MPS for exactly this
  reason, and was 11% faster once synchronised.
- **Interleave the two sides.** Timing all of one and then all of the other lets thermal drift
  land on whichever ran second. One iteration each, alternating, moves it onto both -- so every
  stage goes in one `interleaved` group, including the two that are almost free. Splitting them
  across groups is how the comparison and the breakdown it is compared against stop sharing a
  thermal history.

Nothing here is a gate. It produces the numbers a support matrix is built from.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
# This file is called yolov26.py, so its own directory would shadow the package it benchmarks.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# Local weights unless the caller says otherwise, as every sibling bootstrap tool sets: without
# it this measures bytes fetched from the published bucket while the reference gate measures the
# tree you just built, and the two quietly describe different checkpoints.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from mozo.image import load_image  # noqa: E402
from mozo.vendors.yolov26_deploy.image import letterbox, to_original  # noqa: E402
from mozo.vendors.yolov26_deploy.network import build_detector  # noqa: E402
from mozo.weights import resolve  # noqa: E402
from verify.yolov26_reference import reference_model  # noqa: E402

VARIANTS = ["nano", "small", "medium", "large", "xlarge"]

#: The photograph timed by default. Spelled out rather than imported from ``tests/conftest.py``
#: for the reason ``tools/verify/yolov26_reference.py`` gives: the environment that has
#: `ultralytics` is usually not the one that has pytest.
FIXTURE = ROOT / "tests" / "fixtures" / "images" / "example.jpg"

CONF = 0.25

#: Iterations discarded before timing. Enough for lazy allocation, kernel selection and, on Metal,
#: the first pipeline compile -- all of which land on iteration one and none of which recur.
WARMUP = 8


def synchronise(device: str) -> None:
    """Wait for the device to finish, so a timer measures work rather than submission."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def available(devices: list[str]) -> list[str]:
    """Keep the devices this machine can actually run."""
    usable = {"cpu": True,
              "mps": torch.backends.mps.is_available(),
              "cuda": torch.cuda.is_available()}
    return [d for d in devices if usable.get(d)]


def time_once(call, device: str) -> float:
    """One synchronised call, in milliseconds."""
    synchronise(device)
    start = time.perf_counter()
    call()
    synchronise(device)
    return (time.perf_counter() - start) * 1000


def interleaved(calls: dict[str, object], device: str, iters: int) -> dict[str, float]:
    """Time several alternatives against each other, one iteration each in turn.

    Returns the median per call. Alternating rather than running each to completion is what stops
    thermal drift landing entirely on whichever went second -- over a few hundred iterations a
    machine warms measurably, and a bench that reads 10% either way must not be one of the things
    creating the 10%.
    """
    for call in calls.values():
        for _ in range(WARMUP):
            call()
    samples: dict[str, list[float]] = {name: [] for name in calls}
    for _ in range(iters):
        for name, call in calls.items():
            samples[name].append(time_once(call, device))
    return {name: statistics.median(values) for name, values in samples.items()}


def reference_call(model, image: np.ndarray, imgsz: int, device: str):
    """A callable running the already-built reference on *image*, preprocessing its own way.

    Takes the model rather than a checkpoint, so a directory of photographs loads `ultralytics`
    once per device instead of once per image. *imgsz* comes from the vendor network, so both
    sides are always letterboxed to the same square.
    """
    from ultralytics.data.augment import LetterBox

    def call():
        padded = LetterBox((imgsz, imgsz), auto=False, scaleup=True)(image=image[:, :, ::-1])
        chw = np.ascontiguousarray(padded.transpose(2, 0, 1)[::-1], dtype=np.float32)
        with torch.no_grad():
            return model(torch.from_numpy(chw).div_(255.0)[None].to(device))

    return call


def stages(network, image: np.ndarray, device: str, iters: int, theirs=None) -> dict[str, float]:
    """Split the call into the parts a graph runtime would and would not replace.

    Every stage, including the reference, goes in one ``interleaved`` group. The mozo total is
    ``letterbox + network``, summed from two medians rather than timed a third time: an end-to-end
    call is those two and nothing else, so timing it separately spent a full forward pass per
    iteration to re-derive a number already measured -- 33 of them per image per device.
    """
    imgsz = network.imgsz
    batch, gain, pad_x, pad_y = letterbox(image, imgsz)
    batch = batch.to(device)

    def convolutions():
        """Everything up to and including the head's convolutions, stopping before the decode.

        Mirrors ``Yolo.forward`` including its release schedule -- an earlier copy kept only
        ``keep`` and freed nothing, so it held four activations the real forward drops and
        measured a network that allocates more than the shipped one does.
        """
        with torch.no_grad():
            held, current = {}, batch
            for index, (layer, source) in enumerate(zip(network.model[:-1], network.sources[:-1])):
                current = layer(network._gather(held, current, source))
                for spent in network.release.get(index, ()):
                    del held[spent]
                if index in network.keep:
                    held[index] = current
            feats = network._gather(held, current, network.sources[-1])
            return network.model[-1](feats), [(int(f.shape[2]), int(f.shape[3])) for f in feats]

    def whole():
        with torch.no_grad():
            return network(batch)

    raw, shapes = convolutions()
    budget = min(network.max_det, sum(h * w for h, w in shapes))

    def selection():
        """The anchor grid, the box decode and the two-stage top-k, on a fixed input.

        Timed directly rather than as ``network`` minus ``convolutions``. That subtraction is what
        this replaced: the head is well under a percent of an 82 ms CPU forward, so the difference
        of two medians is dominated by the noise on the larger of them and came out *negative* --
        a number that is not small, but wrong.
        """
        with torch.no_grad():
            boxes, logits = network._decode(raw, shapes)
            return network._select(boxes, logits, budget)

    rows = whole()

    def post():
        kept = rows[0].float().cpu()
        kept = kept[kept[:, 4] > CONF]
        return to_original(kept[:, :4], gain, pad_x, pad_y, image.shape[:2])

    calls = {
        "network": whole,
        "backbone+head convs": convolutions,
        "decode+topk": selection,
        "letterbox": lambda: letterbox(image, imgsz),
        "to_original": post,
    }
    if theirs is not None:
        calls["ultralytics"] = theirs
    timed = interleaved(calls, device, iters)
    timed["mozo"] = timed["letterbox"] + timed["network"]
    return timed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", nargs="+", default=["nano"], choices=VARIANTS)
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    parser.add_argument("--images", type=Path, default=None, help="a directory of photographs")
    parser.add_argument("--iters", type=int, default=25)
    arguments = parser.parse_args()

    paths = sorted(p for p in arguments.images.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
        if arguments.images else [FIXTURE]
    devices = available(arguments.devices)
    if not devices:
        raise SystemExit(f"none of {arguments.devices} is available on this machine")

    try:
        import ultralytics
        version = ultralytics.__version__
    except ImportError:
        version = None

    print(f"torch       {torch.__version__}")
    print(f"reference   ultralytics {version or 'not installed -- vendor timings only'}")
    print(f"devices     {', '.join(devices)}")
    print(f"images      {', '.join(p.name for p in paths)}")
    print(f"iterations  {arguments.iters} after {WARMUP} warm-up\n")

    images = {path.name: load_image(str(path)) for path in paths}

    for variant in arguments.variants:
        checkpoint = Path(resolve("yolov26", variant, "torch-fp32"))
        # Built once per variant. ``.to(device)`` below is all the device loop needs, and
        # rebuilding per device re-read the whole checkpoint -- 119 MB twice over on xlarge.
        network = build_detector(str(checkpoint), fuse=True).eval()

        for device in devices:
            network.to(device)
            # Loaded once per device rather than once per image: a directory of twenty
            # photographs previously reloaded `ultralytics` twenty times.
            reference = reference_model(checkpoint, device) if version else None

            for name, image in images.items():
                theirs = reference_call(reference, image, network.imgsz, device) if reference \
                    else None
                timed = stages(network, image, device, arguments.iters, theirs)
                print(f"yolov26/{variant}  {device}  {name}")
                for label in ("letterbox", "backbone+head convs", "decode+topk", "to_original"):
                    share = (f"   {timed[label] / timed['network'] * 100:>5.1f}% of the network"
                             if label == "decode+topk" else "")
                    print(f"  {label:<22}{timed[label]:>8.2f} ms{share}")
                print(f"  {'network, total':<22}{timed['network']:>8.2f} ms")
                print(f"  {'mozo end to end':<22}{timed['mozo']:>8.2f} ms")
                if theirs:
                    ratio = timed["ultralytics"] / timed["mozo"]
                    print(f"  {'vs ultralytics':<22}{timed['ultralytics']:>8.2f} ms"
                          f"   mozo is {ratio:.2f}x {'faster' if ratio > 1 else 'slower'}")
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
