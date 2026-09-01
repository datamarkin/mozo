#!/usr/bin/env python3
"""Build BEN2's ONNX and CoreML graphs, measure them, and publish neither.

    python tools/export/ben2.py --weights weights/ben2/base/<revision>
    python tools/export/ben2.py --weights ... --keep /tmp/ben2-graphs   # leave the artifacts

This runs once, on a machine you control, and never ships. **It writes nothing into
``weights/``**, which makes it unlike every other tool in this directory -- and that is the point.
§6 of ``plans/vendoring.md`` says a measured "no" is a finding and an unmeasured one is a gap, so
the negative result lives in a script anyone can re-run rather than in a paragraph anyone can
doubt.

``EXECUTES = ("torch",)`` in ``mozo/adapters/ben2.py`` is what this file justifies.

## What was measured, so nobody re-derives it

Both graphs build. Neither is publishable, for different reasons.

**ONNX (fp32, 408.9 MB, opset 17)** is *slower than torch* and does not hold parity:

    forward       6180 ms against torch's 5455 ms -- 0.88x
    parity        max|d| 4.879e-05, MAE 6.112e-07
    alpha         1 grey level on 0.0146% of pixels
    peak RSS      10.8 GB

An artifact ``select_runtime`` would never choose, that would also put two disagreeing copies of
the same model in users' hands. Either reason alone settles it.

**CoreML (fp32 mlprogram)** is the fastest thing here and wrong where it matters:

    forward       386 ms against torch-on-MPS's 601 ms -- 1.56x faster
    compute units CPU_AND_GPU and ALL are identical, so the ANE contributes nothing
    parity        max|d| 7.853e-01, MAE 6.692e-03
    alpha         7.8% of pixels differ by >1 grey level, 1.5% by >32, 0.14% by >128

A difference image is black across every interior and glows along every silhouette: a sub-pixel
shift in the alpha ramp, not a structural error. The matte is visually correct. It is still the
one place a matting model may not differ -- the soft edge is the entire product, which is why this
model exists rather than a segmenter.

**Both failures were chased to their cause; neither is fixable here.**

CoreML's divergence is the *GPU runtime*, not the conversion. Fourteen op classes convert alone to
within 9.5e-07, the MIL program has zero fp16 casts, and macOS14 and macOS15 diverge identically --
but the same .mlpackage on ``CPU_ONLY`` reaches 9.27e-06 with 0.000% of alpha pixels off, against
7.85e-01 and 7.801% on ``CPU_AND_GPU``. ``compute_precision=FLOAT32`` is honoured on CPU and ignored
on the GPU. The error lands on edges because the head ends in a sigmoid: interiors are saturated and
hide it, edges sit where the slope is steepest. Exact costs 5034 ms, which is no faster than torch
on CPU; the 386 ms needs the GPU that will not do float32.

ONNX's 4.9e-05 is not ``onnxruntime``'s optimizer -- disabling every fusion gives 4.718e-05 -- so it
is the kernels' own accumulation order. Core ML on CPU reaches 9.27e-06 on the same model, five
times closer. The slowness is structural: 13,328 nodes and an unfused attention materialising a
16,384 x 5,376 matrix per quadrant, which no available fusion covers and which parity forbids
changing.

## Two changes to the vendor made an export possible at all

Both are in ``blocks.py`` and both are the tracer failing to see a constant that is one. Neither
moves a number in eager -- re-measured after each, ``torch.equal`` with zero delta on every
fixture -- and ``mozo/vendors/ben2_deploy/PROVENANCE.md`` records them:

* ``round(h / ...)`` -> ``round(int(h) / ...)``, because ``round()`` on a traced Tensor raises.
* ``F.adaptive_avg_pool2d`` -> ``F.avg_pool2d`` where the division is exact, because ONNX cannot
  lower adaptive pooling once a reshape has hidden the input's static shape. Bit-identical at all
  fifteen (shape, target) pairs this model uses; ``tools/verify/ben2.py`` checks that.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import FIXTURES  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.ben2_deploy import Predictor  # noqa: E402
from mozo.vendors.ben2_deploy.image import postprocess, preprocess  # noqa: E402

#: Above this the graph is not publishable. Not a tolerance the gate accepts -- mozo's bar for a
#: published artifact is exactness, and this only decides how loudly the report says no.
PARITY_CEILING = 0.0


def synchronise(device: str) -> None:
    """Drain the device queue, so a timer measures work rather than submission.

    Both branches, not just Metal: an accelerator queue that is never drained reports how fast
    work was *submitted*, and a CUDA-only version of this bug is what made SigLIP 2 believe it
    was 23% slower on MPS until the synchronise was added.
    """
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def timed(call, trials: int = 3, warmup: int = 1, device: str = "cpu") -> float:
    """Median milliseconds, warmed and synchronised."""
    for _ in range(warmup):
        call()
    synchronise(device)
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        call()
        synchronise(device)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def report(label: str, reference: torch.Tensor, actual: torch.Tensor, shape, ms: float,
           against: float, against_label: str) -> None:
    """Print one artifact's verdict: how fast, how wrong, and what that does to the alpha."""
    delta = (actual - reference).abs()
    ref_alpha, got_alpha = postprocess(reference, shape), postprocess(actual, shape)
    grey = np.abs(ref_alpha.astype(int) - got_alpha.astype(int))

    print(f"\n  {label}")
    print(f"    speed      {ms:8.1f} ms   against {against_label} {against:8.1f} ms   "
          f"{against / ms:.2f}x")
    print(f"    parity     max|d| {delta.max().item():.3e}   MAE {delta.mean().item():.3e}")
    print(f"    alpha      identical={np.array_equal(ref_alpha, got_alpha)}   "
          f">1 grey level on {100 * (grey > 1).mean():.3f}% of pixels, "
          f"max delta {grey.max()}")
    verdict = "PUBLISHABLE" if delta.max().item() <= PARITY_CEILING and ms < against else "not published"
    print(f"    verdict    {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True,
                        help="a revision directory holding torch-fp32.pth, or a checkpoint path")
    parser.add_argument("--image", type=Path, default=FIXTURES / "example.jpg")
    parser.add_argument("--keep", type=Path, default=None,
                        help="write the graphs here instead of a temporary directory")
    parser.add_argument("--skip", nargs="*", default=(), choices=["onnx", "coreml"])
    args = parser.parse_args()

    checkpoint = args.weights / "torch-fp32.pth" if args.weights.is_dir() else args.weights
    predictor = Predictor.from_pretrained(checkpoint, device="cpu")
    rgb = load_image(args.image)
    tensor = preprocess(rgb)
    shape = rgb.shape[:2]

    print(f"image {args.image.name}  {rgb.shape[1]}x{rgb.shape[0]}   torch {torch.__version__}")
    with torch.no_grad():
        reference = predictor.model(tensor)
    torch_ms = timed(lambda: predictor.model(tensor))
    print(f"  torch cpu    {torch_ms:8.1f} ms   (the reference every row below is measured against)")

    scratch = args.keep or Path(tempfile.mkdtemp(prefix="ben2-export-"))
    scratch.mkdir(parents=True, exist_ok=True)

    if "onnx" not in args.skip:
        path = scratch / "ben2.onnx"
        start = time.perf_counter()
        torch.onnx.export(predictor.model, (tensor,), str(path), opset_version=17,
                          input_names=["image"], output_names=["matte"], dynamo=False)
        print(f"\n  onnx exported in {time.perf_counter() - start:.0f}s, "
              f"{path.stat().st_size / 1e6:.1f} MB")

        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        array = tensor.numpy()
        got = torch.from_numpy(session.run(None, {name: array})[0])
        ms = timed(lambda: session.run(None, {name: array}))
        report("onnx-fp32", reference, got, shape, ms, torch_ms, "torch-cpu")

    if "coreml" not in args.skip and sys.platform == "darwin":
        import coremltools as ct

        path = scratch / "ben2.mlpackage"
        start = time.perf_counter()
        traced = torch.jit.trace(predictor.model, tensor, strict=False)
        model = ct.convert(
            traced,
            inputs=[ct.TensorType(name="image", shape=tuple(tensor.shape), dtype=np.float32)],
            outputs=[ct.TensorType(name="matte", dtype=np.float32)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS14,
        )
        model.save(str(path))
        print(f"\n  coreml converted in {time.perf_counter() - start:.0f}s")

        # CoreML's competition is torch on Metal, not torch on CPU: it is the artifact
        # ``_PREFERENCE`` would put first on Apple silicon, so beating the CPU proves nothing.
        mps = Predictor.from_pretrained(checkpoint, device="mps")
        mps_tensor = preprocess(rgb, device="mps")

        def run_mps():
            with torch.no_grad():
                out = mps.model(mps_tensor)
            torch.mps.synchronize()
            return out

        mps_ms = timed(run_mps, device="mps")
        drift = (run_mps().cpu() - reference).abs().max().item()
        print(f"  torch mps    {mps_ms:8.1f} ms   and already {drift:.3e} from the cpu reference, "
              f"which is why parity is claimed for cpu alone")

        feed = {"image": tensor.numpy()}
        for units, label in [(ct.ComputeUnit.CPU_AND_GPU, "coreml-fp32 (CPU_AND_GPU)"),
                             (ct.ComputeUnit.ALL, "coreml-fp32 (ALL, ANE allowed)")]:
            loaded = ct.models.MLModel(str(path), compute_units=units)
            got = torch.from_numpy(np.asarray(loaded.predict(feed)["matte"]))
            ms = timed(lambda: loaded.predict(feed))
            report(label, reference, got, shape, ms, mps_ms, "torch-mps")

    built = sorted(child.name for child in scratch.iterdir()) if scratch.exists() else []
    if built:
        print(f"\nNothing was published. {', '.join(built)} left in {scratch}"
              f"{' (delete when done)' if args.keep is None else ''}.")
    else:
        scratch.rmdir()
        print("\nNothing was built and nothing was published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
