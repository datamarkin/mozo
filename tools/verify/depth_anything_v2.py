#!/usr/bin/env python3
"""Verify the vendored Depth Anything V2 against the upstream repository it was extracted from.

    python tools/verify/depth_anything_v2.py --images /path/to/photos
    python tools/verify/depth_anything_v2.py --images /path/to/photos --variants small large

Four checks, in increasing strength, plus a timing pass:

1. **Standalone** -- no import outside stdlib, ``torch``, ``torchvision``, ``numpy`` and ``cv2``,
   and no absolute self-import. The vendored tree has to be movable and dependency-light or the
   extraction bought nothing.
2. **Structural** -- state-dict keys and shapes identical to upstream, for every variant.
3. **Forward pass** -- same weights, same input tensor, ``max|delta|`` over the raw output.
4. **End to end** -- real photographs through our ``Predictor.predict`` against upstream's
   ``infer_image``, which is the whole pipeline including the cv2 resize and the resize back.

Checks 3 and 4 must be exactly zero. This is an extraction, not a reimplementation: anything
above zero means a line drifted, and "close enough" is how a drifted line survives.

**The baseline must not be older than the extraction.** An outdated upstream will argue
confidently for behaviour that upstream itself has since abandoned, and it is very convincing
while it does so. This script therefore refuses to run against a commit other than the one
recorded in PROVENANCE.md unless you override it deliberately.

Both sides are pinned to the same device. Upstream's ``image2tensor`` probes the host and moves
the tensor itself, so it is patched here for the duration of the comparison -- an MPS-vs-CPU
comparison measures backend kernels, not two implementations.
"""

from __future__ import annotations

import argparse
import ast
import json
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# This file is named for the package it verifies, so its own directory has to leave sys.path
# before ``import depth_anything_v2`` can reach upstream's package instead of this script.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))  # ...and mozo itself has to be importable without an install

# A full run takes minutes per variant; line-buffer so progress is visible when redirected.
sys.stdout.reconfigure(line_buffering=True)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

VENDOR = ROOT / "mozo" / "vendors" / "depth_anything_v2_deploy"

def _pinned_commit() -> str:
    """The commit PROVENANCE.md records, read from the file so the two cannot disagree."""
    text = (VENDOR / "PROVENANCE.md").read_text()
    for line in text.splitlines():
        if "Upstream commit" in line:
            return line.split("`")[1]
    raise SystemExit("PROVENANCE.md does not record an upstream commit")


#: What the vendored tree is allowed to import. Everything else is a dependency we shed.
ALLOWED_THIRD_PARTY = {"torch", "torchvision", "numpy", "cv2"}

WARMUP = 12  # MPS needs more than a couple of passes before its timings settle


@contextmanager
def pinned_to(device: str):
    """Make upstream's host probing agree with the device we chose.

    ``image2tensor`` hardcodes ``cuda if available else mps if available else cpu`` and moves the
    tensor there itself. Patching the two probes is the smallest change that lets the comparison
    happen on one device, and it touches nothing else in the forward path.
    """
    real_cuda, real_mps = torch.cuda.is_available, torch.backends.mps.is_available
    torch.cuda.is_available = lambda: device == "cuda"
    torch.backends.mps.is_available = lambda: device == "mps"
    try:
        yield
    finally:
        torch.cuda.is_available, torch.backends.mps.is_available = real_cuda, real_mps


def _optional_imports(tree: ast.Module) -> set[int]:
    """Node ids of imports wrapped in ``try: ... except ImportError``.

    An import that is allowed to fail is not a dependency -- it is a fast path the package can
    do without. DINOv2 guards its xformers imports this way, and the fallback is the plain
    attention its own base class defines.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        caught = [h.type for h in node.handlers]
        names = {n.id for t in caught if t is not None for n in ast.walk(t) if isinstance(n, ast.Name)}
        if not names <= {"ImportError", "ModuleNotFoundError"} or not names:
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    guarded.add(id(inner))
    return guarded


def check_standalone() -> list[str]:
    """Return complaints about the vendored tree's imports; empty means it is self-contained."""
    problems = []
    for path in sorted(VENDOR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        optional = _optional_imports(tree)
        for node in ast.walk(tree):
            if id(node) in optional:
                continue
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative -- exactly what we want
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name in ("depth_anything_v2", "depth_anything_v2_deploy", "mozo"):
                    problems.append(f"{path.relative_to(ROOT)}: absolute self-import {name!r}")
                elif name not in ALLOWED_THIRD_PARTY and name not in sys.stdlib_module_names:
                    problems.append(f"{path.relative_to(ROOT)}: imports {name!r}")
    return problems


_UPSTREAM_TREES: dict[Path, object] = {}


def load_upstream(root: Path):
    """Import ``depth_anything_v2.dpt`` from one upstream tree.

    Upstream ships two trees under the same package name -- the repository root for relative
    depth, and ``metric_depth/`` for the metric models, whose ``DepthAnythingV2`` is the only
    one that accepts ``max_depth``. They cannot both be on ``sys.path`` at once, so each is
    imported with the module cache cleared and then kept. This duplication is exactly what
    ``dpt.py``'s six-line modification removes on our side; the baseline still has to reach
    into whichever copy corresponds to the variant under test.
    """
    root = root.resolve()
    if root not in _UPSTREAM_TREES:
        for name in [n for n in sys.modules if n.split(".")[0] == "depth_anything_v2"]:
            del sys.modules[name]
        sys.path.insert(0, str(root))
        try:
            import depth_anything_v2.dpt as module
        finally:
            sys.path.remove(str(root))
        _UPSTREAM_TREES[root] = module
    return _UPSTREAM_TREES[root]


def build_pair(variant: str, weights: Path, device: str, upstream: Path):
    """Build ours and upstream's model for *variant* from the same checkpoint file."""
    from mozo.vendors.depth_anything_v2_deploy import Predictor, get_spec

    spec = get_spec(variant)
    ours = Predictor.from_pretrained(variant, weights=weights, device=device)

    upstream_module = load_upstream(upstream if spec.relative else upstream / "metric_depth")
    theirs = upstream_module.DepthAnythingV2(
        encoder=spec.encoder,
        features=spec.features,
        out_channels=list(spec.out_channels),
        **({} if spec.max_depth is None else {"max_depth": spec.max_depth}),
    )
    theirs.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return ours, theirs.to(device).eval(), spec


def compare_structure(ours, theirs) -> list[str]:
    """Return complaints about state-dict shape; empty means the architectures match."""
    a, b = ours.model.state_dict(), theirs.state_dict()
    problems = []
    if set(a) != set(b):
        for key in sorted(set(a) ^ set(b)):
            problems.append(f"key only in {'ours' if key in a else 'upstream'}: {key}")
    for key in sorted(set(a) & set(b)):
        if a[key].shape != b[key].shape:
            problems.append(f"{key}: {tuple(a[key].shape)} vs {tuple(b[key].shape)}")
    return problems


@torch.inference_mode()
def compare_forward(ours, theirs, device: str, shape=(1, 3, 518, 728)) -> float:
    """Max absolute difference on the raw forward pass, given one identical input tensor."""
    torch.manual_seed(0)
    x = torch.randn(*shape).to(device)
    return float((ours.model(x) - theirs(x)).abs().max())


@torch.inference_mode()
def compare_end_to_end(ours, theirs, images: list[np.ndarray], device: str) -> tuple[float, list]:
    """Max absolute difference over full-pipeline depth maps, and the shapes actually fed."""
    worst = 0.0
    shapes = []
    for image in images:
        # Spelled out rather than calling ``ours.predict``, which would preprocess a second time
        # purely so the shape below could be read off it -- a full cvtColor, INTER_CUBIC resize
        # and host-to-device upload per image, per variant.
        tensor, size = ours.preprocess(image)
        mine = ours.postprocess(ours.model(tensor), size)
        with pinned_to(device):
            upstream = theirs.infer_image(image)
        worst = max(worst, float(np.abs(mine - upstream).max()))
        shapes.append(tuple(tensor.shape[-2:]))
    return worst, shapes


@torch.inference_mode()
def measure(predictor, images: list[np.ndarray], iters: int) -> float:
    """Median per-image latency in milliseconds, after warm-up."""
    for _ in range(WARMUP):
        predictor.predict(images[0])
    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        for image in images:
            predictor.predict(image)
        timings.append((time.perf_counter() - start) * 1000 / len(images))
    return statistics.median(timings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="directory of photographs")
    parser.add_argument("--limit", type=int, default=10, help="how many images to use")
    parser.add_argument("--variants", nargs="*", default=None, help="default: all nine")
    parser.add_argument("--device", default=None, help="default: mps, cuda, or cpu, whichever exists")
    parser.add_argument("--iters", type=int, default=3, help="timed passes over the image set")
    parser.add_argument("--upstream", type=Path, default=ROOT.parent / "Depth-Anything-V2",
                        help="checkout of DepthAnything/Depth-Anything-V2")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    parser.add_argument("--revision", default="2026-08-19")
    parser.add_argument("--out", type=Path, default=None, help="write results as JSON")
    parser.add_argument("--allow-commit-drift", action="store_true",
                        help="run against an upstream other than the extracted commit")
    args = parser.parse_args()

    # mozo's own detection, not a second copy of it: these numbers describe the device mozo
    # would have picked, so the two must not be able to disagree.
    from mozo.device import get_default_device

    device = args.device or get_default_device()

    if not args.upstream.is_dir():
        raise SystemExit(
            f"no upstream checkout at {args.upstream}. Clone it:\n"
            f"  git clone https://github.com/DepthAnything/Depth-Anything-V2 {args.upstream}"
        )
    head = subprocess.run(["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    pinned = _pinned_commit()
    if head != pinned and not args.allow_commit_drift:
        raise SystemExit(
            f"upstream is at {head[:12]}, PROVENANCE.md records {pinned[:12]}.\n"
            f"A baseline older than the extraction will disagree with it and sound right doing so.\n"
            f"Check it out ('git -C {args.upstream} checkout {pinned}') or pass --allow-commit-drift."
        )

    from mozo.vendors.depth_anything_v2_deploy import MODEL_SPECS

    variants = args.variants or list(MODEL_SPECS)
    paths = sorted(p for p in args.images.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:args.limit]
    if not paths:
        raise SystemExit(f"no images in {args.images}")
    # BGR on both sides, deliberately. This compares the *vendor* against upstream, and the
    # vendor keeps upstream's contract verbatim -- ``cv2.imread`` order, converted internally.
    # mozo's RGB contract lives one layer up, in the adapter; it is not in this comparison.
    images = [cv2.imread(str(p)) for p in paths]

    print(f"device {device}, upstream {head[:12]}, {len(images)} images, {args.iters} timed passes\n")

    print("standalone:")
    problems = check_standalone()
    for problem in problems:
        print(f"  FAIL {problem}")
    if not problems:
        print(f"  ok — {len(list(VENDOR.rglob('*.py')))} files import only "
              f"{', '.join(sorted(ALLOWED_THIRD_PARTY))} and the stdlib\n")

    results = []
    for variant in variants:
        weights = args.weights_dir / "depth_anything_v2" / variant / args.revision / "torch-fp32.pth"
        if not weights.is_file():
            print(f"=== {variant}: no checkpoint at {weights}, skipped")
            continue

        ours, theirs, spec = build_pair(variant, weights, device, args.upstream)

        structure = compare_structure(ours, theirs)
        forward = compare_forward(ours, theirs, device)
        end_to_end, shapes = compare_end_to_end(ours, theirs, images, device)
        ms = measure(ours, images, args.iters)

        ok = not structure and forward == 0.0 and end_to_end == 0.0
        unit = spec.unit or "relative"
        print(f"=== {variant:<14} {spec.encoder}  {unit:<8} "
              f"{'IDENTICAL' if ok else 'MISMATCH'}")
        print(f"    state dict   {len(ours.model.state_dict())} tensors, "
              f"{'identical' if not structure else f'{len(structure)} differences'}")
        print(f"    forward      max|delta| {forward:g}")
        print(f"    end to end   max|delta| {end_to_end:g}   over {len(images)} photographs")
        print(f"    speed        {ms:.1f} ms   {1000 / ms:.1f} fps   "
              f"input {min(s[0] for s in shapes)}-{max(s[0] for s in shapes)}"
              f"x{min(s[1] for s in shapes)}-{max(s[1] for s in shapes)}")
        for problem in structure[:5]:
            print(f"    FAIL {problem}")

        results.append({
            "variant": variant, "encoder": spec.encoder, "unit": spec.unit, "device": device,
            "images": len(images), "state_dict_tensors": len(ours.model.state_dict()),
            "structure_differences": structure, "forward_max_delta": forward,
            "end_to_end_max_delta": end_to_end, "ms": ms, "fps": 1000 / ms,
            "identical": ok,
        })
        del ours, theirs

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.out}")

    identical = sum(r["identical"] for r in results)
    print(f"\n{identical}/{len(results)} variants bit-identical to upstream"
          + ("" if not problems else f", {len(problems)} standalone violations"))
    return 0 if identical == len(results) and not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
