"""Drawing what a model found onto the image it found it in.

Every node here is one call to :mod:`pixelflow.annotate`, which owns the drawing. What this module
contributes is the declaration: which ports, which parameters, and what the editor should offer for
each. Nothing about how a box is rasterised lives in mozo.

Every annotator draws into the array it is given and returns that same array -- PixelFlow says so in
each of their docstrings, and for a video pipeline it is the right default. A workflow node cannot
behave that way: one image is commonly wired to several nodes at once, and a diamond that drew boxes
on one branch and blurred faces on the other would have each branch painting over the other's work.
So each node here copies before it draws, which is what declaring ``-> Image`` means.
``tests/workflow/test_pixelflow_nodes.py`` holds every node that takes an image to that, not only
these. The copy costs 0.54 ms on a 1920x1281 photograph, against the 60 ms encoding the result
costs -- it is not the expensive part of anything.

Colours arrive as ``"#RRGGBB"`` because that is what a colour picker produces, and go to PixelFlow
as an ``(r, g, b)`` tuple. No channel swap: PixelFlow writes the tuple through to the array
untouched, and mozo's arrays are RGB -- verified rather than assumed, since a silently swapped red
and blue is exactly the class of bug ``mozo.image``'s docstring exists to prevent.

Several PixelFlow arguments take ``None`` to mean "scale this with the image", which is better than
any fixed number on a photograph whose size is not known in advance. Those are annotated
``int | None`` and pass straight through, so the catalogue says which parameters may be left unset
rather than leaving a reader to know that ``thickness=0`` was a code for automatic.
"""

from __future__ import annotations

from typing import Literal, Optional

import pixelflow as pf

from ..node import Color, Detections, Image
from ..registry import node

#: Where a label sits relative to its box. PixelFlow reads the vertical half from the prefix and
#: the horizontal half from the rest, so these nine are the combinations that mean something.
LABEL_POSITIONS = Literal[
    "top_left", "top_center", "top_right",
    "center_left", "center", "center_right",
    "bottom_left", "bottom_center", "bottom_right",
]


@node(category="Annotate")
def draw_boxes(image: Image, detections: Detections, thickness: Optional[int] = None,
               color: Optional[Color] = None) -> Image:
    """Draw a box around each detection."""
    return pf.annotate.box(image.copy(), detections, thickness=thickness, colors=_colors(color))


@node(category="Annotate")
def draw_labels(image: Image, detections: Detections, position: LABEL_POSITIONS = "top_left",
                font_scale: Optional[float] = None, padding: int = 6,
                text_color: Color = "#FFFFFF", background: Optional[Color] = None) -> Image:
    """Write each detection's class and score beside it."""
    return pf.annotate.label(image.copy(), detections, position=position, font_scale=font_scale,
                             padding=padding, text_color=_rgb(text_color),
                             bg_color=_rgb(background))


@node(category="Annotate")
def draw_masks(image: Image, detections: Detections, opacity: float = 0.5,
               color: Optional[Color] = None) -> Image:
    """Shade each detection's mask over the image."""
    return pf.annotate.mask(image.copy(), detections, opacity=opacity, colors=_colors(color))


@node(category="Annotate")
def draw_polygons(image: Image, detections: Detections, thickness: Optional[int] = None,
                  color: Optional[Color] = None) -> Image:
    """Outline each detection's mask rather than filling it."""
    return pf.annotate.polygon(image.copy(), detections, thickness=thickness,
                               colors=_colors(color))


@node(category="Annotate")
def draw_keypoints(image: Image, detections: Detections, radius: Optional[int] = None,
                   thickness: Optional[int] = None, color: Optional[Color] = None,
                   show_names: bool = False, min_confidence: float = 0.0) -> Image:
    """Mark each detection's keypoints."""
    return pf.annotate.keypoint(image.copy(), detections, radius=radius, thickness=thickness,
                                colors=_colors(color), show_names=show_names,
                                min_confidence=min_confidence)


@node(category="Annotate")
def draw_skeleton(image: Image, detections: Detections, thickness: Optional[int] = None,
                  color: Optional[Color] = None, min_confidence: float = 0.0) -> Image:
    """Join each detection's keypoints into a skeleton."""
    return pf.annotate.keypoint_skeleton(image.copy(), detections, thickness=thickness,
                                         colors=_colors(color), min_confidence=min_confidence)


@node(category="Annotate")
def blur_regions(image: Image, detections: Detections, kernel_size: Optional[int] = None,
                 padding_percent: float = 0.05) -> Image:
    """Blur whatever was detected -- faces, plates, anything you would rather not publish."""
    return pf.annotate.blur(image.copy(), detections, kernel_size=kernel_size,
                            padding_percent=padding_percent)


@node(category="Annotate")
def pixelate_regions(image: Image, detections: Detections, pixel_size: Optional[int] = None,
                     padding_percent: float = 0.05) -> Image:
    """Pixelate whatever was detected, where a blur would look like a mistake."""
    return pf.annotate.pixelate(image.copy(), detections, pixel_size=pixel_size,
                                padding_percent=padding_percent)


def _rgb(color: Optional[str]) -> Optional[tuple]:
    """``"#RRGGBB"`` as ``(r, g, b)``. Unset stays unset, which PixelFlow reads as "you choose".

    Left as None, PixelFlow colours each detection by its class, which is what you want on a
    photograph with several kinds of thing in it -- so the useful default is no colour at all.
    """
    if color is None:
        return None
    text = color.lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a colour like '#RRGGBB', got {color!r}")
    return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))


def _colors(color: Optional[str]) -> Optional[list]:
    """The same, as the one-item list the annotators take when every detection shares a colour."""
    resolved = _rgb(color)
    return [resolved] if resolved else None
