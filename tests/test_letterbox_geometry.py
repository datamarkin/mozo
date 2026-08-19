"""The letterbox and its inverse must agree about where the image was put.

Every detection family that letterboxes has two halves of one decision: the pre-processor places
the resized image somewhere on a square canvas, and the post-processor subtracts that placement
to get back to source pixels. If the two disagree the model still runs, still finds the right
objects, and reports every one of them in the wrong place -- so nothing else in the suite fails.

The disagreement is available: for a source whose scaled side is odd, the spare space is a half
pixel, and rounding it one way in the placement and the other way in the inverse costs half a
canvas pixel, which is ``0.5 / gain`` source pixels. On the fixture photograph that is 1.5 px on
every box. Two vendors harvested from the same upstream disagreed about exactly this, which is
what these tests exist to stop.

They decide it by measurement rather than by assertion: count the border rows the letterbox
actually wrote, and require the inverse to remove precisely those. Neither half is trusted to
describe itself.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

import mozo.vendors

#: Every vendor that letterboxes, found rather than listed. Discovery is the point: this file is
#: the compensating control for vendors being deliberately independent copies of each other, and
#: the bug it exists to catch was one copy drifting from another. A hand-maintained list is the
#: same failure mode one level up -- the third detection family would land with the same silent
#: 1.5 px error and a green suite because nobody remembered to add a line here.
VENDORS = sorted(f"mozo.vendors.{path.parent.name}.image"
                 for path in Path(mozo.vendors.__file__).parent.glob("*/image.py"))

#: Source shapes. Three pad to a half pixel -- ``1281x1920`` is the fixture photograph, 106.5 rows;
#: ``1000x999`` and ``641x640`` pad 0.5 columns -- and three pad whole. The whole-pixel ones pass
#: under either convention and are here as the control: ``1080x810`` and ``720x1280`` are the two
#: photographs both upstream parity suites used, which is exactly why neither could see the bug.
SHAPES = [(1281, 1920), (1000, 999), (641, 640), (333, 777), (1080, 810), (720, 1280)]

SIZE = 640


@pytest.fixture(params=VENDORS)
def vendor(request):
    """One vendor's letterboxing module."""
    return importlib.import_module(request.param)


def _placement(canvas: np.ndarray, border: int) -> tuple[int, int]:
    """Return the top and left border actually written, by finding the first non-border line."""
    rows = ~np.all(canvas == border, axis=1)
    columns = ~np.all(canvas == border, axis=0)
    return int(np.argmax(rows)), int(np.argmax(columns))


def _letterboxed(vendor, height: int, width: int):
    """Letterbox a black image of *height* x *width* and return the canvas plus the placement."""
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
    """Discovery that silently finds nothing would make every test above vacuously pass."""
    assert len(VENDORS) >= 2, f"expected several letterboxing vendors, found {VENDORS}"
