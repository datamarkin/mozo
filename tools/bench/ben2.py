#!/usr/bin/env python3
"""Measure what BEN2 costs to serve, against the implementation it was extracted from.

Bootstrap tooling; never ships. ``tools/verify/ben2.py`` answers whether mozo returns the same
numbers as upstream; this answers what it costs to get them.

    python tools/bench/ben2.py --upstream /path/to/BEN2 --weights weights/ben2/base/<rev>
    python tools/bench/ben2.py --upstream ... --weights ... --devices cpu mps

**The comparison is legitimate because both sides run the same weights** -- the same checkpoint,
the same photograph, the same device, the same torch. Any gap is this extraction against
upstream's own file, which is the only thing mozo can be responsible for.

**Every measurement synchronises the device.** Metal queues asynchronously, so a timer that stops
when Python returns measures how fast work was *submitted*. SigLIP 2 reported itself 23% slower on
MPS for exactly this reason before the synchronise was added.

**Both sides are timed end to end**, from an ``HxWx3`` array to a matte, because that is the work
a caller asks for. Timing a bare forward would flatter whichever side does more of its work
outside one -- and here that is upstream, whose ``inference`` also does the PIL resize, the
normalise and the compositing.

**Upstream is forced onto the float32 path.** ``BEN_Base.forward`` carries
``@torch.autocast(device_type="cuda")`` and ``inference`` picks its normalisation from
``torch.cuda.is_available()``, so on a CUDA box the two sides would be comparing fp16 against
fp32 -- a precision difference reported as a speed difference. Neither the extraction nor this
benchmark claims the fp16 path.

Nothing here is a pass/fail gate. It produces the numbers a support matrix is built from.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

from common import FIXTURES  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

WARMUP = 2      # the model is 5 backbone passes; two is enough to settle allocation
TRIALS = 5      # interleaved, so thermal drift moves both sides together


def synchronise(device: str) -> None:
    """Drain the device queue, so a timer measures work rather than submission."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _once(call, device: str) -> float:
    """One synchronised wall-clock sample, in milliseconds."""
    start = time.perf_counter()
    call()
    synchronise(device)
    return (time.perf_counter() - start) * 1000


def time_it(call, device: str, trials: int = TRIALS) -> float:
    """Median wall-clock milliseconds over *trials*, warmed up and synchronised."""
    for _ in range(WARMUP):
        call()
    synchronise(device)
    return statistics.median(_once(call, device) for _ in range(trials))


def time_pair(left, right, device: str, trials: int = TRIALS) -> tuple[float, float]:
    """Time two calls **alternately**, one trial each, and return both medians.

    Interleaved rather than one side then the other, which is not a refinement: on a laptop that
    throttles, a block of five runs of A followed by five of B measures the machine cooling down
    or heating up and reports it as a difference between A and B. Alternating makes any drift
    common to both. This function exists because the first version of this file timed them in
    blocks while its own docstring claimed otherwise.
    """
    for _ in range(WARMUP):
        left(); right()
    synchronise(device)

    lefts, rights = [], []
    for _ in range(trials):
        lefts.append(_once(left, device))
        rights.append(_once(right, device))
    return statistics.median(lefts), statistics.median(rights)


def bench_device(upstream, ours, rgb: np.ndarray, pil: Image.Image, device: str) -> None:
    """One device: the two implementations, then where mozo's own time goes."""
    from mozo.vendors.ben2_deploy.image import postprocess, preprocess

    print(f"\n--- {device} ---")

    ours_ms, up_ms = time_pair(
        lambda: ours.matte(rgb),
        lambda: upstream.inference(pil.copy(), refine_foreground=False), device)
    ratio = up_ms / ours_ms if ours_ms else float("nan")
    print(f"  end to end     mozo {ours_ms:8.1f} ms   upstream {up_ms:8.1f} ms   "
          f"{ratio:.2f}x")

    # Where mozo's time goes. Not what you would guess: the five Swin-B passes are under a third
    # of the forward, and the cross-attention decoder is the rest. MCLM and MCRM leave
    # need_weights=True on every nn.MultiheadAttention call, which takes torch's unfused branch
    # and materialises the whole attention matrix -- at the shallowest rung that is 16,384 query
    # tokens per quadrant. Reproduced deliberately; see PROVENANCE.md.
    tensor = preprocess(rgb, device=device)
    pre_ms = time_it(lambda: preprocess(rgb, device=device), device)
    with torch.inference_mode():
        fwd_ms = time_it(lambda: ours.model(tensor), device)
        matte = ours.model(tensor)  # reused by the postprocess and refine timings below
    post_ms = time_it(lambda: postprocess(matte, rgb.shape[:2]), device)
    print(f"  breakdown      preprocess {pre_ms:6.1f}   forward {fwd_ms:8.1f}   "
          f"postprocess {post_ms:6.1f}")

    # The backbone alone, to say how much of the forward is Swin and how much is the decoder.
    with torch.inference_mode():
        from mozo.vendors.ben2_deploy.blocks import image2patches, rescale_to

        quads = image2patches(tensor[:1])
        glb = rescale_to(tensor[:1], scale_factor=0.5, interpolation="bilinear")
        five = torch.cat((quads, glb), dim=0)
        back_ms = time_it(lambda: ours.model.backbone(five), device)
    share = 100 * back_ms / fwd_ms if fwd_ms else float("nan")
    print(f"  of which       Swin-B x5 {back_ms:8.1f} ms ({share:.0f}% of the forward)")

    # What refine costs, since it is the one argument with a real price. Timed directly rather
    # than as the difference between two ``cutout`` calls: that phrasing ran the network ten more
    # times -- around a minute on CPU -- to price two box blurs that do not involve it at all.
    from mozo.vendors.ben2_deploy.image import refine_foreground

    alpha = postprocess(matte, rgb.shape[:2])
    refine_ms = time_it(lambda: refine_foreground(rgb, alpha), device, trials=3)
    print(f"  refine=True    {refine_ms:8.1f} ms extra at {rgb.shape[1]}x{rgb.shape[0]} "
          f"(two full-resolution box blurs; the forward is unchanged)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=FIXTURES / "example.jpg")
    parser.add_argument("--devices", nargs="+", default=None,
                        help="default: cpu, plus mps when available")
    args = parser.parse_args()

    devices = args.devices or (["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"])

    sys.path.insert(0, str(args.upstream))
    import BEN2

    checkpoint = args.weights / "torch-fp32.pth" if args.weights.is_dir() else args.weights
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = blob["model_state_dict"] if "model_state_dict" in blob else blob

    pil = Image.open(args.image).convert("RGB")
    rgb = np.asarray(pil)
    print(f"image {args.image.name}  {rgb.shape[1]}x{rgb.shape[0]}   torch {torch.__version__}")
    print(f"trials {TRIALS} (median), warmup {WARMUP}, device synchronised")

    from mozo.vendors.ben2_deploy import Predictor

    for device in devices:
        upstream = BEN2.BEN_Base().eval()
        upstream.load_state_dict(state, strict=True)
        upstream = upstream.to(device)

        ours = Predictor.from_pretrained(checkpoint, device=device)
        bench_device(upstream, ours, rgb, pil, device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
