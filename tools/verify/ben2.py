#!/usr/bin/env python3
"""Verify the vendored BEN2 against the upstream repository it was extracted from.

    python tools/verify/ben2.py --upstream /path/to/BEN2 --weights weights/ben2/base/<rev>
    python tools/verify/ben2.py --upstream /path/to/BEN2 --weights ... --images /path/to/photos

Six checks, in increasing strength:

1. **Standalone** -- no import outside stdlib, ``torch``, ``numpy``, ``cv2`` and ``PIL``, and no
   absolute self-import. Upstream additionally requires ``timm`` and ``einops``; the point of the
   extraction is that neither survives, so an import of either here is a regression.
2. **The reshape rewrites** -- all nine ``einops.rearrange`` calls that were replaced by
   ``view``/``permute``/``reshape``, compared against ``einops`` itself on random tensors, plus
   the round trip ``patches2image(image2patches(x)) == x``. This is the riskiest change in the
   package: a wrong axis order reassembles the quadrants into the wrong corners and still looks
   like a matte.
3. **Structural** -- state-dict keys and shapes identical to upstream, and a strict load.
4. **Forward pass** -- same weights, same input tensor, ``torch.equal`` on the raw matte.
5. **Every stage, end to end** -- the preprocessed tensor, the matte, the postprocessed alpha,
   and both of upstream's documented paths (``refine_foreground`` off and on), on real
   photographs including the awkward ones.
6. **Falsification** -- ten perturbations of the vendored code, each of which must fail, and each
   at the stage it actually reaches. A gate that has never failed has not been shown to work, and
   one whose probe misses is worse: two of these originally patched ``blocks`` where the names are
   defined, while ``network.py`` binds them at import, so they reached nothing and reported
   "nothing moved" -- indistinguishable from a correct subject.

Checks 3 to 5 must be exactly zero: this is an extraction, not a reimplementation.

**Both sides are pinned to CPU and float32.** Upstream is two models -- ``BEN_Base.forward``
carries ``@torch.autocast(device_type="cuda", dtype=torch.float16)`` and ``inference`` picks its
normalisation with ``torch.cuda.is_available()`` -- so "bit-exact" is undefined until the device
is pinned. mozo claims the float32 path and nothing else, and this script is where that is
enforced rather than hoped for.

**Upstream cannot run every fixture.** A 1x1 image drives ``postprocess_image`` into a
zero-dimensional array (``np.squeeze`` with no axis) which ``ToPILImage`` rejects, and a matte the
model reads as uniform divides by ``max - min`` == 0 and casts ``nan`` to uint8. Both are recorded
as upstream failures rather than treated as mozo's -- and both are the reason the vendor's
postprocess squeezes an explicit axis and guards the denominator.

**The baseline must not be older than the extraction.** This script refuses to run against a
commit other than the one recorded in PROVENANCE.md unless you override it deliberately.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# This file is named for the package it verifies, so its own directory has to leave sys.path
# before ``import BEN2`` can reach upstream's module instead of this script.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

sys.stdout.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from common import FIXTURES, fixtures  # noqa: E402
from mozo.vendors.ben2_deploy.image import postprocess, preprocess  # noqa: E402

VENDOR = ROOT / "mozo" / "vendors" / "ben2_deploy"

#: What the vendored tree is allowed to import. ``timm`` and ``einops`` are deliberately absent.
ALLOWED_THIRD_PARTY = {"torch", "numpy", "cv2", "PIL"}


def _pinned_commit() -> str:
    """The commit PROVENANCE.md records, read from the file so the two cannot disagree."""
    for line in (VENDOR / "PROVENANCE.md").read_text().splitlines():
        if line.startswith("Upstream commit:"):
            return line.split("`")[1]
    raise SystemExit("PROVENANCE.md does not record an upstream commit")


# --------------------------------------------------------------------------------------------
# 1. standalone


def check_standalone() -> list[str]:
    """Return complaints about the vendored tree's imports; empty means it is self-contained."""
    problems = []
    for path in sorted(VENDOR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in ALLOWED_THIRD_PARTY and root not in sys.stdlib_module_names:
                        problems.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative -- what we want
                    continue
                root = (node.module or "").split(".")[0]
                if root == "mozo":
                    problems.append(f"{path.name}: imports mozo ({node.module})")
                elif root not in ALLOWED_THIRD_PARTY and root not in sys.stdlib_module_names:
                    problems.append(f"{path.name}: from {node.module} import ...")
    return problems


# --------------------------------------------------------------------------------------------
# 2. the reshape rewrites


def check_rewrites() -> list[tuple[str, bool]]:
    """Every ``rearrange`` that was replaced, against ``einops`` on random tensors."""
    from einops import rearrange

    from mozo.vendors.ben2_deploy.blocks import _bchw_to_hwbc, image2patches, patches2image

    torch.manual_seed(0)
    results = []

    x = torch.randn(3, 7, 16, 20)
    results.append(("image2patches", torch.equal(
        image2patches(x), rearrange(x, "b c (hg h) (wg w) -> (hg wg b) c h w", hg=2, wg=2))))

    y = torch.randn(12, 7, 8, 10)
    results.append(("patches2image", torch.equal(
        patches2image(y), rearrange(y, "(hg wg b) c h w -> b c (hg h) (wg w)", hg=2, wg=2))))

    results.append(("patches2image inverts image2patches", torch.equal(patches2image(image2patches(x)), x)))

    z = torch.randn(2, 9, 4, 6)
    results.append(("b c h w -> (h w) b c", torch.equal(
        _bchw_to_hwbc(z), rearrange(z, "b c h w -> (h w) b c"))))

    w = torch.randn(24, 2, 9)
    ours = w.view(4, 6, 2, 9).permute(2, 3, 0, 1).contiguous()
    results.append(("(h w) b c -> b c h w", torch.equal(
        ours, rearrange(w, "(h w) b c -> b c h w", h=4, w=6))))

    q = torch.randn(16, 16, 1, 11)
    ours = q.view(2, 8, 2, 8, 1, 11).permute(1, 3, 0, 2, 4, 5).reshape(64, 4, 11)
    results.append(("(ng h) (nw w) b c -> (h w) (ng nw b) c", torch.equal(
        ours, rearrange(q, "(ng h) (nw w) b c -> (h w) (ng nw b) c", ng=2, nw=2))))

    n = torch.randn(4, 13, 5, 7)
    results.append(("nl c h w -> nl c (h w)", torch.equal(
        n.flatten(2), rearrange(n, "nl c h w -> nl c (h w)"))))

    m = torch.randn(4, 13, 35)
    results.append(("nl c nphw -> nl nphw 1 c", torch.equal(
        m.permute(0, 2, 1).unsqueeze(2), rearrange(m, "nl c nphw -> nl nphw 1 c"))))

    results.append(("nl c h w -> nl (h w) 1 c", torch.equal(
        n.flatten(2).permute(0, 2, 1).unsqueeze(2), rearrange(n, "nl c h w -> nl (h w) 1 c"))))

    return results


#: Every (input shape, pooling target) pair the model reaches, derived from the frozen 1024 input.
#: ``_pool`` substitutes ``avg_pool2d`` for ``adaptive_avg_pool2d`` at each, which is what lets the
#: model export at all -- see ``tools/export/ben2.py``. The substitution is exact only because
#: every ratio divides evenly, so it is checked rather than argued.
POOLING = (
    ((1, 128, 32, 32), [(16, 16), (4, 4), (2, 2)]),      # MCLM
    ((4, 128, 16, 16), [(16, 16), (8, 8), (4, 4)]),      # dec_blk4
    ((4, 128, 32, 32), [(32, 32), (16, 16), (8, 8)]),    # dec_blk3
    ((4, 128, 64, 64), [(64, 64), (32, 32), (16, 16)]),  # dec_blk2 and dec_blk1
)


def check_pooling() -> list[tuple[str, bool]]:
    """``_pool`` against ``F.adaptive_avg_pool2d`` at every shape the model uses."""
    import torch.nn.functional as F

    from mozo.vendors.ben2_deploy.blocks import _pool

    torch.manual_seed(0)
    results = []
    for shape, targets in POOLING:
        x = torch.randn(*shape)
        for target in targets:
            same = torch.equal(_pool(x, target), F.adaptive_avg_pool2d(x, target))
            results.append((f"{shape} -> {target}", same))
    return results


# --------------------------------------------------------------------------------------------
# 3-5. against upstream


def _attempt(call):
    """Run an upstream call, reporting a crash or a silent NaN as a result rather than raising."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = call()
        except Exception as error:
            return None, type(error).__name__
        if any("invalid value" in str(w.message) for w in caught):
            return value, "NaN"
        return value, None


def compare(upstream, ours, images: list[Path]) -> tuple[int, int, list[str]]:
    """Every stage on every photograph. Returns (exact, compared, notes)."""
    import BEN2

    exact = compared = 0
    notes: list[str] = []
    columns = ("pixels", "matte", "alpha", "cutout", "refine")
    print(f"\n{'fixture':18s} " + " ".join(f"{c:>9}" for c in columns))

    for path in images:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        cells = []

        # -- the preprocessed tensor
        up_image, h_, w_, _ = BEN2.rgb_loader_refiner(Image.open(path))
        up_tensor = BEN2.img_transform32(up_image).unsqueeze(0)
        our_tensor = preprocess(rgb)
        cells.append(torch.equal(up_tensor, our_tensor))

        # -- the raw matte
        with torch.no_grad():
            up_matte = upstream(up_tensor)
            # Same no-grad context as upstream's: the raw module is being called here rather than
            # ``Predictor.matte``, which carries its own ``inference_mode``, and a tensor that
            # requires grad cannot reach ``postprocess``'s ``.numpy()``.
            our_matte = ours.model(our_tensor)
        cells.append(torch.equal(up_matte, our_matte))

        # -- the postprocessed alpha. im_size=[w_, h_] is upstream's swapped-name (H, W).
        up_alpha, note = _attempt(lambda: BEN2.postprocess_image(up_matte, im_size=[w_, h_]))
        # From the matte already computed above rather than a second ``ours.matte`` call, which
        # would re-run the whole network to reach a tensor sitting in a local.
        our_alpha = postprocess(our_matte, rgb.shape[:2], stretch=True)
        cells.append(note or np.array_equal(up_alpha, our_alpha))

        # -- end to end, both of upstream's documented paths. Each cutout is computed once.
        our_cut = ours.cutout(rgb, stretch=True, refine=False)
        up_cut, note = _attempt(
            lambda: np.asarray(upstream.inference(Image.open(path), refine_foreground=False)))
        cells.append(note or (up_cut.shape == our_cut.shape and np.array_equal(up_cut, our_cut)))

        our_ref = ours.cutout(rgb, refine=True)
        up_ref, note = _attempt(
            lambda: np.asarray(upstream.inference(Image.open(path), refine_foreground=True)))
        cells.append(note or (up_ref.shape == our_ref.shape and np.array_equal(up_ref, our_ref)))

        rendered = []
        for column, value in zip(columns, cells):
            if value is True:
                rendered.append("exact"); exact += 1; compared += 1
            elif value is False:
                rendered.append("DIFFER"); compared += 1
            else:
                rendered.append(value)
                notes.append(f"{path.name}/{column}: upstream {value}")
        print(f"{path.name:18s} " + " ".join(f"{c:>9}" for c in rendered))

    return exact, compared, notes


# --------------------------------------------------------------------------------------------
# 6. falsification

def check_falsifiable(upstream, ours, image: Path) -> list[tuple[str, bool, str, str]]:
    """Perturb one thing at a time, confirm the gate notices, restore.

    Each entry names the stage that *must* move. Getting the stage right matters as much as
    catching the change: a perturbation to the normalisation that first shows up in the matte
    would mean the pixel comparison is not actually running.

    The perturbations are chosen to be individually plausible -- every one of them is something a
    reader could talk themselves into while "tidying" the extraction, and none of them raises.
    """
    import torch.nn.functional as F

    import BEN2

    from mozo.vendors.ben2_deploy import blocks, image as vendor_image, network, swin

    rgb = np.asarray(Image.open(image).convert("RGB"))
    up_image, _, _, _ = BEN2.rgb_loader_refiner(Image.open(image))
    up_tensor = BEN2.img_transform32(up_image).unsqueeze(0)
    with torch.no_grad():
        up_matte = upstream(up_tensor)

    def observe() -> tuple[str, float]:
        """First stage that moved, and by how much.

        A perturbation can change a tensor's *shape* rather than its values -- resizing to 1000
        instead of 1024 does -- and subtracting those raises rather than reporting a difference.
        A shape change is the loudest possible difference, so it counts as one.
        """
        tensor = preprocess(rgb)
        if tensor.shape != up_tensor.shape:
            return "pixels", float("inf")
        delta = (tensor - up_tensor).abs().max().item()
        if delta:
            return "pixels", delta
        with torch.no_grad():
            matte = ours.model(tensor)
        if matte.shape != up_matte.shape:
            return "matte", float("inf")
        delta = (matte - up_matte).abs().max().item()
        return ("matte", delta) if delta else ("nothing", 0.0)

    def set_attr(module, name, value):
        original = getattr(module, name)
        setattr(module, name, value)
        return lambda: setattr(module, name, original)

    bicubic = Image.BICUBIC
    real_gelu, real_interpolate = F.gelu, F.interpolate

    #: (label, how to apply it, which stage must notice)
    cases = (
        ("normalisation mean 0.485 -> 0.585",
         lambda: set_attr(vendor_image, "MEAN", (0.585, 0.456, 0.406)), "pixels"),
        ("normalisation std 0.229 -> 0.230",
         lambda: set_attr(vendor_image, "STD", (0.230, 0.224, 0.225)), "pixels"),
        ("resize filter LANCZOS -> BICUBIC",
         lambda: set_attr(Image, "LANCZOS", bicubic), "pixels"),
        ("input side 1024 -> 1000",
         lambda: set_attr(vendor_image, "INPUT", 1000), "pixels"),
        ("patches2image quadrant order reversed",
         lambda: set_attr(blocks, "patches2image",
                          lambda x, f=blocks.patches2image: f(x.flip(0))), "matte"),
        # Patched on `network`, not `blocks`: network.py does `from .blocks import rescale_to`,
        # so it holds its own binding and patching the source module would fire nothing. That is
        # itself worth a perturbation -- a gate whose probe misses is indistinguishable from a
        # gate whose subject is correct.
        ("global downscale gains antialias=True",
         lambda: set_attr(network, "rescale_to",
                          lambda x, scale_factor=2, interpolation="nearest": real_interpolate(
                              x, scale_factor=scale_factor, mode=interpolation,
                              **({"antialias": True} if interpolation != "nearest" else {}))),
         "matte"),
        ("resize_as bilinear -> nearest",
         lambda: set_attr(network, "resize_as",
                          lambda x, y, interpolation="nearest": real_interpolate(
                              x, size=y.shape[-2:], mode="nearest")), "matte"),
        ("GELU -> tanh approximation",
         lambda: set_attr(F, "gelu", lambda x, approximate="none": real_gelu(x, approximate="tanh")),
         "matte"),
        ("Swin attention scale x 1.0000001",
         lambda: _scale_attention(ours.model, swin, 1.0000001), "matte"),
        ("LayerNorm eps 1e-5 -> 1e-6",
         lambda: _layernorm_eps(ours.model, 1e-6), "matte"),
    )

    results = []
    for label, apply, expected in cases:
        restore = apply()
        try:
            moved, delta = observe()
        finally:
            restore()
        shown = "shape" if delta == float("inf") else (f"{delta:.2e}" if delta else "-")
        results.append((label, moved == expected, moved, shown))

    return results


def _scale_attention(model, swin, factor: float):
    """Nudge every window-attention scale, and hand back the undo.

    The smallest perturbation in the set, and the one that matters most: it is the size of
    difference an "equivalent" rewrite of the attention introduces, and the reason this package
    keeps upstream's scale-then-matmul order rather than the algebraically identical alternative.
    """
    touched = [(m, m.scale) for m in model.modules() if isinstance(m, swin.WindowAttention)]
    for module, scale in touched:
        module.scale = scale * factor
    return lambda: [setattr(m, "scale", s) for m, s in touched]


def _layernorm_eps(model, eps: float):
    """Retune every LayerNorm's epsilon, and hand back the undo."""
    touched = [(m, m.eps) for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    for module, _ in touched:
        module.eps = eps
    return lambda: [setattr(m, "eps", e) for m, e in touched]



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", type=Path, required=True,
                        help="checkout of github.com/PramaLLC/BEN2")
    parser.add_argument("--weights", type=Path, required=True,
                        help="directory holding torch-fp32.pth, or a checkpoint path")
    parser.add_argument("--images", type=Path, default=None,
                        help="directory of photographs; defaults to the shared fixtures")
    parser.add_argument("--allow-any-commit", action="store_true",
                        help="run against a baseline other than the one PROVENANCE.md pins")
    args = parser.parse_args()

    torch.manual_seed(0)

    # -- 0. the baseline
    pinned = _pinned_commit()
    head = subprocess.run(["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    print(f"upstream  {args.upstream}")
    print(f"  HEAD    {head or '(not a git checkout)'}")
    print(f"  pinned  {pinned}")
    if head != pinned and not args.allow_any_commit:
        raise SystemExit(
            f"\nupstream is at {head}, PROVENANCE.md pins {pinned}. An outdated baseline argues "
            f"confidently for behaviour upstream has abandoned. Pass --allow-any-commit if you "
            f"mean it.")
    print(f"  torch   {torch.__version__}   device cpu, dtype float32")

    # -- 1. standalone
    problems = check_standalone()
    print(f"\n[1] standalone: {'ok' if not problems else 'FAILED'}")
    for problem in problems:
        print(f"    {problem}")

    # -- 2. the reshape rewrites, and the pooling substitution
    rewrites = check_rewrites()
    print(f"\n[2] reshape rewrites vs einops: {sum(ok for _, ok in rewrites)}/{len(rewrites)} exact")
    for label, ok in rewrites:
        if not ok:
            print(f"    FAILED {label}")

    pooling = check_pooling()
    print(f"    _pool vs adaptive_avg_pool2d: {sum(ok for _, ok in pooling)}/{len(pooling)} exact")
    for label, ok in pooling:
        if not ok:
            print(f"    FAILED {label}")

    sys.path.insert(0, str(args.upstream))
    import BEN2

    checkpoint = args.weights / "torch-fp32.pth" if args.weights.is_dir() else args.weights
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = blob["model_state_dict"] if "model_state_dict" in blob else blob

    upstream = BEN2.BEN_Base().eval()
    upstream.load_state_dict(state, strict=True)

    from mozo.vendors.ben2_deploy import Predictor

    ours = Predictor.from_pretrained(checkpoint, device="cpu")

    # -- 3. structural
    up_keys = {k: tuple(v.shape) for k, v in upstream.state_dict().items()}
    our_keys = {k: tuple(v.shape) for k, v in ours.model.state_dict().items()}
    print(f"\n[3] structural: {len(our_keys)} keys, "
          f"{'identical' if up_keys == our_keys else 'DIFFER'}")
    for key in sorted(set(up_keys) ^ set(our_keys)):
        print(f"    only one side has {key}")

    # -- 4/5. forward and end to end
    # ``common.fixtures`` rather than a private glob: it refuses an empty directory by name,
    # where a copy of the glob reports "0/0 comparisons exact" and then dies on ``images[0]``.
    images = fixtures() if args.images is None else sorted(
        p for p in args.images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"no photographs in {args.images}")
    exact, compared, notes = compare(upstream, ours, images)
    print(f"\n[4/5] {exact}/{compared} comparisons exact")
    for note in notes:
        print(f"    {note}   (upstream limit, not a mozo divergence -- see PROVENANCE.md)")

    # -- 6. falsification
    print("\n[6] falsification -- each must fail, at its own stage")
    falsified = check_falsifiable(upstream, ours, images[0])
    for label, ok, moved, delta in falsified:
        print(f"    {'caught' if ok else 'MISSED'}  {label:40s} {moved:>8s}  {delta:>9s}")

    good = (not problems and all(ok for _, ok in rewrites) and all(ok for _, ok in pooling)
            and up_keys == our_keys
            and exact == compared and all(ok for _, ok, _, _ in falsified))
    print(f"\n{'PASS' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
