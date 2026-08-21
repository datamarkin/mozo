#!/usr/bin/env python3
"""Check that mozo's EasyOCR returns exactly what the published model does.

Three paths reach the same weights and this compares all three.

``mozo/vendors/easyocr_deploy`` is one: build a ``Reader``, hand it an image, read the strings
off it. ``easyocr`` is the second -- the Apache-2.0 package this was extracted from, driven
through its own ``Reader``. And mozo itself is the third: registry lookup, weights resolution,
adapter, runtime selection, PixelFlow result. Between the first two sit two rewritten networks,
a preprocessing rewrite, an OpenCV postprocessing rewrite and a CTC decoder; between the first
and third sit a hull, a quad conversion and a result conversion. Any of those could quietly
change a number.

**The comparison against ``easyocr`` is exact.** Not "close": a tolerance would hide precisely
the drift this exists to catch. Six divergences found while building this package were each at a
magnitude a tolerance would have swallowed -- most sharply, batching the recogniser rather than
running one crop per forward moves the logits by 1.4e-05, which is enough to flip a marginal
character and invisible in any box.

**Upstream is driven at the level where the inputs are unambiguous**: ``detect`` on the same RGB
array, then ``recognize`` on the same greyscale page, rather than ``readtext`` on a path.
Upstream's path entry decodes greyscale with ``cv2.imread(..., IMREAD_GRAYSCALE)``, which is
libjpeg's direct-to-grey and differs from converting its own decoded RGB by up to 7 levels on a
JPEG; its array entry documents its input as BGR and would channel-swap instead. Neither is a
statement about the model, and comparing through either would measure the decoder. Handing both
sides the same two arrays measures the arithmetic.

**Two more things are pinned, for reasons that are not obvious.**

``quantize=False``. Upstream's reader defaults to ``quantize=True`` on CPU and then swallows the
result in a bare ``except``: where torch has a quantization engine the recogniser really is
qint8, and where it has none -- as on this machine, which raises ``NoQEngine`` -- it silently is
not. Left at the default, this gate would compare against a different model on a different
machine and blame mozo.

``allowlist=character``. Upstream masks every character outside the languages the caller asked
for, so its output depends on a language list rather than on the weights. A mozo variant is a
checkpoint and decodes its whole alphabet; passing the full charset as the allowlist is how
upstream is asked the same question. Without it the two disagree correctly, and the failure
reads like a decoder bug -- see the vendor's PROVENANCE.md.

Run from the repository root::

    python tools/verify/easyocr.py                    # english
    python tools/verify/easyocr.py latin korean       # others
    python tools/verify/easyocr.py --all
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent

# This file is named after the package it compares against -- every gate is named for its family
# -- and Python puts a script's own directory first on the path, so ``import easyocr`` would find
# this module instead of the installed one. Dropping that entry is what makes the name safe.
_HERE = Path(__file__).resolve().parent
sys.path[:] = [entry for entry in sys.path if entry and Path(entry).resolve() != _HERE]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from mozo.adapters.easyocr import EasyOCRPredictor  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.easyocr_deploy import SPECS, VARIANTS, Reader  # noqa: E402
from mozo.vendors.easyocr_deploy import boxes as _boxes  # noqa: E402
from mozo.vendors.easyocr_deploy import image as _image  # noqa: E402
from mozo.weights import resolve  # noqa: E402

from conftest import FIXTURE, TEXT_FIXTURES  # noqa: E402

#: Which languages to ask upstream for so that it loads the recogniser mozo publishes as this
#: variant. The list is upstream's selector, not a property of the weights, which is the whole
#: reason a mozo variant is named after the script instead.
LANGUAGES = {
    "english": ["en"],
    "latin": ["en", "fr", "de", "es"],
    "chinese-simplified": ["ch_sim", "en"],
    "japanese": ["ja", "en"],
    "korean": ["ko", "en"],
}


def images() -> list[Path]:
    """Every fixture, photograph last."""
    rendered = sorted(p for p in TEXT_FIXTURES.glob("*.png"))
    if not rendered:
        raise SystemExit(f"no fixtures in {TEXT_FIXTURES}")
    return rendered + ([FIXTURE] if FIXTURE.is_file() else [])


def upstream_reader(variant: str):
    """Upstream's reader for *variant*, on CPU, unquantized."""
    try:
        import easyocr
    except ImportError:
        raise SystemExit(
            "This gate compares against the published package. Install it with:\n"
            "    pip install easyocr"
        ) from None
    return easyocr.Reader(LANGUAGES[variant], gpu=False, verbose=False, quantize=False)


class Comparison:
    """Collects stage-by-stage verdicts and prints one line each."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.count = 0

    def check(self, name: str, want, got) -> bool:
        self.count += 1
        same = _identical(want, got)
        if not same:
            self.failures.append(name)
            print(f"    {name:52s} DIFFERS  {_delta(want, got)}")
        return same

    def note(self, name: str, text: str) -> None:
        print(f"    {name:52s} {text}")


def _array(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=object if _ragged(value) else float)
    return np.asarray(value)


def _ragged(value) -> bool:
    try:
        np.asarray(value, dtype=float)
        return False
    except (ValueError, TypeError):
        return True


def _identical(want, got) -> bool:
    if isinstance(want, str) or isinstance(got, str):
        return want == got
    if isinstance(want, float) or isinstance(got, float):
        return float(want) == float(got)
    a, b = _array(want), _array(got)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _delta(want, got) -> str:
    a, b = _array(want), _array(got)
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    try:
        return f"max|delta| = {np.abs(a.astype(float) - b.astype(float)).max()}"
    except (ValueError, TypeError):
        return f"{want!r} vs {got!r}"


def against_upstream(variant: str, checkpoint: Path, comparison: Comparison) -> None:
    """Compare every stage, from the preprocessed tensor to the decoded string."""
    from easyocr.imgproc import normalizeMeanVariance, resize_aspect_ratio
    from easyocr.craft_utils import adjustResultCoordinates, getDetBoxes
    from easyocr.utils import get_image_list

    up = upstream_reader(variant)
    mine = Reader(checkpoint, SPECS[variant], device="cpu")

    for path in images():
        rgb = load_image(path)
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        print(f"  {path.name}")

        # -- preprocessing -------------------------------------------------------------
        up_resized, up_ratio, _ = resize_aspect_ratio(
            rgb, 2560, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0)
        up_batch = torch.from_numpy(
            np.array([np.transpose(normalizeMeanVariance(up_resized), (2, 0, 1))]))
        my_batch, my_ratio = _image.for_detector(rgb)
        comparison.check("detector input", up_batch, my_batch)
        comparison.check("resize ratio", up_ratio, my_ratio)

        # -- detector forward ----------------------------------------------------------
        with torch.no_grad():
            up_heat, _feature = up.detector(up_batch)
            my_heat = mine.detector(my_batch)
        comparison.check("heatmaps", up_heat, my_heat)

        region = my_heat[0][:, :, 0].numpy()
        affinity = my_heat[0][:, :, 1].numpy()

        # -- box extraction ------------------------------------------------------------
        up_quads, _, _ = getDetBoxes(region, affinity, 0.7, 0.4, 0.4, False, False)
        my_quads = _boxes.quads(region, affinity)
        comparison.check(f"quads (n={len(my_quads)})", up_quads, my_quads)

        up_polys = [np.array(b).astype(np.int32).reshape(-1)
                    for b in adjustResultCoordinates(up_quads, 1 / up_ratio, 1 / up_ratio)]
        my_polys = _boxes.rescale(my_quads, my_ratio)
        comparison.check("rescaled", up_polys, my_polys)

        up_h, up_f = up.detect(rgb)
        # What ``mine.detect`` computes from the values already in hand, rather than a second
        # and third forward pass of the same image -- 3.3 s each on the photograph, times every
        # fixture times every variant.
        my_h, my_f = _boxes.group(my_polys)
        comparison.check(f"grouped horizontal (n={len(my_h)})", up_h[0], my_h)
        comparison.check(f"grouped free (n={len(my_f)})", up_f[0], my_f)

        # -- crops, logits and the read --------------------------------------------------
        for index, (line, is_free) in enumerate(
                [(b, False) for b in my_h] + [(q, True) for q in my_f]):
            up_crops, up_width = get_image_list(
                [] if is_free else [line], [line] if is_free else [], grey, model_height=64)
            cut = _image.line_image(line, grey, is_free=is_free)
            comparison.check(f"line {index} kept", len(up_crops) == 1, cut is not None)
            if cut is None:
                continue
            _quad, my_crop, my_width = cut
            comparison.check(f"line {index} padded width", float(up_width), float(my_width))
            comparison.check(f"line {index} crop", up_crops[0][1], my_crop)

        # -- the whole pipeline ----------------------------------------------------------
        up_result = up.recognize(grey, up_h[0], up_f[0], reformat=False,
                                 allowlist=up.character)
        my_result = mine.read(grey, my_h, my_f)
        if not comparison.check("region count", len(up_result), len(my_result)):
            continue
        for index, ((quad, text, score), region_out) in enumerate(zip(up_result, my_result)):
            comparison.check(f"region {index} quad", quad, region_out.quad)
            comparison.check(f"region {index} text", text, region_out.text)
            comparison.check(f"region {index} confidence", float(score), region_out.confidence)
        comparison.note("read", " | ".join(repr(r.text) for r in my_result) or "(nothing)")


def against_mozo(variant: str, comparison: Comparison) -> None:
    """Compare the vendor against mozo's own adapter, through PixelFlow.

    PixelFlow rounds coordinates to two decimals and confidences to three, which is the only
    door mozo's numbers come through, so the vendor's values are put through the same rounding
    rather than the comparison being given a tolerance.
    """
    model = EasyOCRPredictor(variant, device="cpu")
    reader = Reader(resolve("easyocr", variant, "torch-fp32"), SPECS[variant], device="cpu")

    for path in images():
        rgb = load_image(path)
        expected = reader(rgb)
        found = model.predict(path)
        print(f"  {path.name}")
        if not comparison.check("adapter count", len(expected), len(found)):
            continue
        for index, (region, detection) in enumerate(zip(expected, found)):
            quad = np.asarray(region.quad, dtype=float)
            hull = [round(float(v), 2) for v in
                    (quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max())]
            comparison.check(f"adapter {index} bbox", hull, detection.bbox)
            comparison.check(f"adapter {index} segments", quad.round(2), detection.segments)
            comparison.check(f"adapter {index} text", region.text, detection.text)
            comparison.check(f"adapter {index} confidence",
                             round(region.confidence, 3), detection.confidence)
            comparison.check(f"adapter {index} class_name", "None", str(detection.class_name))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="*", default=None,
                        help=f"variant names (default: english). Known: {VARIANTS}")
    parser.add_argument("--all", action="store_true", help="every published variant")
    args = parser.parse_args()

    wanted = VARIANTS if args.all else (args.variants or ["english"])
    unknown = [v for v in wanted if v not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}. Known: {VARIANTS}")

    warnings.filterwarnings("ignore")
    torch.backends.cudnn.enabled = False  # the LSTM is the one place a backend may reorder

    comparison = Comparison()
    for variant in wanted:
        checkpoint = resolve("easyocr", variant, "torch-fp32")
        print(f"\n=== {variant} vs the published package ===")
        against_upstream(variant, checkpoint, comparison)
        print(f"\n=== {variant} vs mozo's adapter, through PixelFlow ===")
        against_mozo(variant, comparison)

    print()
    if comparison.failures:
        print(f"{len(comparison.failures)} of {comparison.count} comparisons differ. EasyOCR: FAIL")
        for name in comparison.failures[:20]:
            print(f"  {name}")
        return 1
    print(f"{comparison.count} comparisons, every one identical to the published model. "
          f"EasyOCR: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
