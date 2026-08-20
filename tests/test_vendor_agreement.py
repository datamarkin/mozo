"""Invariants every detection vendor must satisfy, and must agree with the others about.

mozo's detection vendors are deliberately independent copies of each other -- see any of their
PROVENANCE.md files for why. That buys reproducibility against each upstream and costs the usual
thing: four copies drift, and the ways they drift are invisible. A vendor that letterboxes slightly
differently still runs, still finds the right objects, and reports every one of them in the wrong
place, so nothing else in the suite fails.

This file is the compensating control. Each invariant here is measured rather than asserted, and
every vendor is discovered rather than listed. That has now been paid off once: the fourth family
landed already covered, without a line added here.

**The letterbox and its inverse must agree about where the image was put.** For a source whose
scaled side is odd the spare space is a half pixel, and rounding it one way in the placement and
the other way in the inverse costs ``0.5 / gain`` source pixels -- 1.5 px on the fixture
photograph, on every box. Three harvests arrived disagreeing about exactly this, one of them with
a docstring arguing for the wrong answer. Decided by counting the border rows the letterbox
actually wrote and requiring the inverse to remove precisely those.

**Suppression must separate classes even at negative coordinates**, which is where a
too-narrow class-separating band stops separating.

**The vendors that suppress must agree on their defaults**, because mozo's adapters do not pass an
overlap threshold and the vendor default is therefore what every served prediction uses.

The last two apply only to the families that suppress. YOLO26 is trained to fire once per object
and returns a ranked detection list from the network itself, so it has no ``suppress`` and no
``iou`` -- a fact about the architecture, not a hole in the coverage. :func:`suppresses` is the one
place that is decided, and both tests report the family they skipped rather than dropping it in
silence.

The placement invariants are qualified the same way, by :func:`letterboxes`. SAM 2 resizes
straight to a square and distorts the aspect ratio rather than padding, so there is no border to
place and nothing here to be right about. Two predicates, two architectural facts, each read in
one place -- and :func:`test_more_than_one_vendor_was_discovered` holds both to finding somebody,
because a suite that skips everything reads exactly like a suite that passes everything.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

import mozo.vendors

#: Every vendor package with an image module, found rather than listed. Discovery is the point: a
#: hand-maintained list is the same failure mode one level up -- the fourth detection family would
#: land with the same silent 1.5 px error and a green suite because nobody remembered to add a
#: line here. Packages rather than modules, so the tests below derive whichever module they need
#: from one place instead of doing string surgery on each other's names.
VENDOR_PACKAGES = sorted(f"mozo.vendors.{path.parent.name}"
                         for path in Path(mozo.vendors.__file__).parent.glob("*/image.py"))

#: Source shapes. Three pad to a half pixel -- ``1281x1920`` is the fixture photograph, 106.5 rows;
#: ``1000x999`` and ``641x640`` pad 0.5 columns -- and three pad whole. The whole-pixel ones pass
#: under either convention and are here as the control: ``1080x810`` and ``720x1280`` are the two
#: photographs both upstream parity suites used, which is exactly why neither could see the bug.
SHAPES = [(1281, 1920), (1000, 999), (641, 640), (333, 777), (1080, 810), (720, 1280)]

SIZE = 640


@pytest.fixture(params=VENDOR_PACKAGES)
def vendor(request):
    """One vendor's image module: letterboxing and the coordinate mapping."""
    return importlib.import_module(f"{request.param}.image")


def letterboxes(package: str) -> bool:
    """Whether *package* letterboxes at all, asked once for the same reason as :func:`suppresses`.

    A family that resizes straight to a square has no border to place and no padding to undo, so
    it exposes no ``letterbox`` and no ``BORDER``. SAM 2 is the first: it squashes to 1024x1024
    and distorts the aspect ratio, which is what it was trained under. Every invariant below is
    about where a border was written, so on such a family there is nothing to be right or wrong
    about -- as distinct from being right by accident, which is what a vacuous pass would look
    like.
    """
    return hasattr(importlib.import_module(f"{package}.image"), "letterbox")


def suppresses(package: str) -> bool:
    """Whether *package* suppresses at all, asked once so two tests cannot disagree about it.

    A family whose head fires once per object has nothing to suppress, so it exposes no
    ``suppress`` and its ``detect`` takes no overlap threshold. Both follow from one architectural
    fact, and reading it in one place is what stops a family that keeps an ignored ``iou=`` kwarg
    for compatibility from being skipped by one test and included by the other.
    """
    return hasattr(importlib.import_module(f"{package}.image"), "suppress")


def _placement(canvas: np.ndarray, border: int) -> tuple[int, int]:
    """Return the top and left border actually written, by finding the first non-border line."""
    rows = ~np.all(canvas == border, axis=1)
    columns = ~np.all(canvas == border, axis=0)
    return int(np.argmax(rows)), int(np.argmax(columns))


def _letterboxed(vendor, height: int, width: int):
    """Letterbox a black image of *height* x *width* and return the canvas plus the placement.

    The one door all three placement tests go through, so it is also where a family that does not
    letterbox is declined -- once, rather than in each of them, on the same reasoning that puts
    :func:`suppresses` in a single place.
    """
    if not letterboxes(vendor.__package__):
        pytest.skip(f"{vendor.__name__} resizes to a square: no border is written to check")
    # Black, so no pixel of the content can be mistaken for the grey border.
    batch, gain, pad_x, pad_y = vendor.letterbox(np.zeros((height, width, 3), np.uint8), SIZE)
    canvas = np.rint(batch[0, 0].numpy() * 255).astype(int)
    return canvas, gain, pad_x, pad_y


@pytest.mark.parametrize(("height", "width"), SHAPES)
def test_the_inverse_removes_the_border_that_was_written(vendor, height, width):
    """``to_original`` must subtract the placement the letterbox actually used.

    This is the whole invariant. A box drawn on the content's top-left corner in canvas
    coordinates has to come back as the source image's top-left corner, exactly.
    """
    canvas, gain, pad_x, pad_y = _letterboxed(vendor, height, width)
    top, left = _placement(canvas, vendor.BORDER)

    corner = torch.tensor([[float(left), float(top), float(left) + 1, float(top) + 1]])
    back = vendor.to_original(corner, gain, pad_x, pad_y, (height, width))

    assert float(back[0, 0]) == pytest.approx(0.0, abs=1e-4)
    assert float(back[0, 1]) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize(("height", "width"), SHAPES)
def test_the_reported_padding_is_the_padding_that_was_written(vendor, height, width):
    """The placement ``letterbox`` reports must be the one it wrote, not a rounding of it.

    Checked separately from the round trip because the two can cancel: a vendor that reports a
    fractional pad and subtracts the same fractional pad round-trips perfectly while placing the
    image somewhere else, and every box it returns is off by the difference.
    """
    canvas, _, pad_x, pad_y = _letterboxed(vendor, height, width)
    top, left = _placement(canvas, vendor.BORDER)

    assert (pad_x, pad_y) == (left, top)


@pytest.mark.parametrize(("height", "width"), SHAPES)
def test_the_content_fills_what_is_left(vendor, height, width):
    """The border on the far side must be what the near side leaves over.

    Without this, a placement and an inverse that both floor would still pass above while losing
    a row off the bottom of the image.
    """
    canvas, gain, pad_x, pad_y = _letterboxed(vendor, height, width)
    rows = np.flatnonzero(~np.all(canvas == vendor.BORDER, axis=1))
    columns = np.flatnonzero(~np.all(canvas == vendor.BORDER, axis=0))

    assert len(rows) == round(height * gain)
    assert len(columns) == round(width * gain)
    assert rows[0] == pad_y and columns[0] == pad_x


def test_a_half_pixel_of_padding_is_actually_exercised():
    """Shapes that pad to a half pixel must stay in the table, or none of this proves anything.

    Both upstream parity suites were run on images that pad to whole pixels -- 80.0 and 140.0 --
    where every convention agrees. A shape list that drifted back to only such images would leave
    these tests green and blind. Not parametrized over vendors: this is a property of the table,
    not of anyone's letterbox.
    """
    fractional = 0
    for height, width in SHAPES:
        gain = min(SIZE / height, SIZE / width)
        spare_y = (SIZE - round(height * gain)) / 2
        spare_x = (SIZE - round(width * gain)) / 2
        fractional += bool(spare_y % 1 or spare_x % 1)

    assert fractional >= 3


def test_more_than_one_vendor_was_discovered():
    """Discovery that silently finds nothing would make every test above vacuously pass.

    The second assertion guards the skip in :func:`_letterboxed` the same way: a predicate that
    stopped recognising anyone -- a renamed function, a vendor layout change -- would turn every
    placement test into a skip, and a suite of skips reads green.
    """
    assert len(VENDOR_PACKAGES) >= 2, f"expected several vendors, found {VENDOR_PACKAGES}"
    placing = [package for package in VENDOR_PACKAGES if letterboxes(package)]
    assert len(placing) >= 2, f"expected several letterboxing vendors, found {placing}"


def test_classes_cannot_suppress_each_other_across_the_padded_edge(vendor):
    """Suppression must separate classes even where the letterbox puts boxes at negative pixels.

    Each vendor shifts every class into its own band of coordinates so one class cannot suppress
    another, then runs a single pass. The width of that band has to cover the full span of the
    boxes. One harvest sized it as ``max + 1``, which is only the full span when the smallest
    coordinate is zero -- and a detection running off the padded edge produces negative ones, at
    which point two classes overlap inside the same shifted range and delete each other.

    No fixture photograph shows this: on mozo's, the smallest letterbox-space coordinate is 0.3.
    So it is constructed instead -- two identical boxes of different classes, lying almost entirely
    at negative coordinates. Both must survive, because non-maximum suppression is per class.

    The box has to sit mostly *below* the origin, not merely straddle it. A box spanning -100..100
    still separates under the narrow band, because the shift it produces (101) is large next to the
    box (200) and leaves an overlap of only 0.14. Spanning -1000..10 makes the narrow shift 11
    against a box of 1010, an overlap of 0.98, which suppresses.
    """
    if not suppresses(vendor.__package__):
        pytest.skip(f"{vendor.__name__} is NMS-free: its network returns a detection list")

    # (4 + classes, anchors): two anchors, two classes, centre-form boxes sharing one position
    # that runs far off the left and top edges.
    prediction = torch.zeros(6, 2)
    corner = torch.tensor([-495.0, -495.0, 1010.0, 1010.0])      # corners -1000..10
    prediction[:4, 0] = corner
    prediction[:4, 1] = corner                                   # identical box
    prediction[4, 0] = 0.9                                       # anchor 0 is class 0
    prediction[5, 1] = 0.8                                       # anchor 1 is class 1

    boxes, _, labels = vendor.suppress(prediction, conf=0.5, iou=0.7, max_det=300)

    assert float(boxes.min()) < 0, "the constructed boxes must straddle the origin to test anything"
    assert len(boxes) == 2, "two classes at the same place must both survive; they suppressed each other"
    assert set(int(label) for label in labels) == {0, 1}


def test_the_vendors_agree_on_their_suppression_defaults():
    """One family's boxes should not overlap differently from another's by accident.

    A harvest arrived defaulting to ``iou=0.45`` where its siblings used ``0.7`` -- and where its
    own recorded parity table had been measured at ``0.7``. mozo's adapters do not pass an overlap
    threshold, so the vendor default is what every served prediction actually uses.

    Invisible to the family suites: at their 0.25 threshold the two values return identical
    detections on the fixture photograph, and only diverge at 0.05 and below.
    """
    defaults = {}
    for package in VENDOR_PACKAGES:
        if not suppresses(package):
            print(f"skipping {package}: NMS-free, nothing to overlap")
            continue
        parameters = inspect.signature(importlib.import_module(f"{package}.model").detect).parameters
        defaults[package] = tuple(parameters[name].default for name in ("conf", "iou", "max_det"))

    assert len(defaults) >= 2, f"expected several suppressing vendors, found {sorted(defaults)}"
    assert len(set(defaults.values())) == 1, f"vendors disagree on suppression defaults: {defaults}"
    # Pinned to a value, not only to each other. Agreement alone would let a coordinated drift --
    # a new harvest arrives at 0.45 and the others are "fixed" to match, which is how the
    # assertion above invites you to resolve a failure -- silently change the conditions every
    # published artifact was verified under, because tools/export passes no overlap threshold and
    # takes these very defaults.
    assert next(iter(defaults.values())) == (0.25, 0.7, 300)
