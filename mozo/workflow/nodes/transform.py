"""Changing the image, and changing what was found in it to match.

Two kinds of node, and the difference is the whole point of the module. A geometric change to an
image invalidates every box drawn on it: rotate the photograph and a box that fitted a face now
sits over an ear. So each geometry node comes in two forms -- one that moves only pixels, and one
that moves the detections with them and produces both.

That is why a node may declare more than one output. PixelFlow's detection-aware transforms return
the new image *and* the moved detections, because those are one operation and splitting them into
two nodes would do the arithmetic twice from two sets of parameters that could disagree.

``normalize`` and ``standardize`` are deliberately absent. Both return float arrays rather than the
``HxWx3`` ``uint8`` an image port carries, and nothing downstream -- annotating, saving, a model --
takes one. They are model preprocessing, and every mozo adapter already does its own.
"""

from __future__ import annotations

from typing import Literal, Optional

import pixelflow.transforms as transforms

from mozo.text import comma_separated

from ..node import Detections, Image
from ..registry import node

#: Which edge a padding fraction is measured against.
REFERENCE = Literal["shorter", "longer", "width", "height"]


# --- Pixels only -------------------------------------------------------------------------------

@node(category="Transform")
def rotate(image: Image, angle: float = 90.0) -> Image:
    """Rotate the image about its centre, in degrees."""
    return transforms.rotate(image, angle)


@node(category="Transform")
def flip_horizontal(image: Image) -> Image:
    """Mirror the image left to right."""
    return transforms.flip_horizontal(image)


@node(category="Transform")
def flip_vertical(image: Image) -> Image:
    """Mirror the image top to bottom."""
    return transforms.flip_vertical(image)


@node(category="Transform")
def crop(image: Image, left: int = 0, top: int = 0, right: Optional[int] = None,
         bottom: Optional[int] = None) -> Image:
    """Cut a rectangle out of the image. Right and bottom default to the image's own edges."""
    return transforms.crop(image, _box(image, left, top, right, bottom))


@node(category="Transform")
def to_grayscale(image: Image) -> Image:
    """Drain the colour, keeping three channels so the result is still an image."""
    return transforms.to_grayscale(image, keep_channels=True)


# --- Exposure ----------------------------------------------------------------------------------

@node(category="Adjust")
def enhance_clahe(image: Image, clip_limit: float = 2.0, tile_size: int = 8) -> Image:
    """Lift local contrast, which brings detail out of shadow without blowing out the rest."""
    return transforms.clahe(image, clip_limit=clip_limit, tile_size=(tile_size, tile_size))


@node(category="Adjust")
def auto_contrast(image: Image, cutoff: float = 1.0) -> Image:
    """Stretch the histogram to the full range, ignoring the brightest and darkest few percent."""
    return transforms.auto_contrast(image, cutoff=cutoff)


@node(category="Adjust")
def gamma_correction(image: Image, gamma: float = 1.0) -> Image:
    """Brighten or darken without clipping. Below 1 brightens, above 1 darkens."""
    return transforms.gamma_correction(image, gamma)


# --- Pixels and detections together --------------------------------------------------------------

@node(category="Transform")
def rotate_with_detections(image: Image, detections: Detections,
                           angle: float = 90.0) -> tuple[Image, Detections]:
    """Rotate the image and carry the detections round with it."""
    return transforms.rotate_detections(image, detections, angle)


@node(category="Transform")
def flip_horizontal_with_detections(image: Image,
                                    detections: Detections) -> tuple[Image, Detections]:
    """Mirror the image left to right, and the detections with it."""
    return transforms.flip_horizontal_detections(image, detections)


@node(category="Transform")
def flip_vertical_with_detections(image: Image,
                                  detections: Detections) -> tuple[Image, Detections]:
    """Mirror the image top to bottom, and the detections with it."""
    return transforms.flip_vertical_detections(image, detections)


@node(category="Transform")
def crop_with_detections(image: Image, detections: Detections, left: int = 0, top: int = 0,
                         right: Optional[int] = None,
                         bottom: Optional[int] = None) -> tuple[Image, Detections]:
    """Cut a rectangle out, keeping the detections that fall inside it."""
    return transforms.crop_detections(
        image, detections, _box(image, left, top, right, bottom))


@node(category="Transform")
def align_by_keypoints(image: Image, detections: Detections, first: str = "left_eye",
                       second: str = "right_eye", target_angle: float = 0.0,
                       detection_index: int = 0) -> tuple[Image, Detections]:
    """Rotate so that two named keypoints sit at a chosen angle -- how a face is straightened."""
    return transforms.rotate_to_align(image, detections, first, second,
                                      target_angle=target_angle,
                                      detection_index=detection_index)


@node(category="Transform")
def crop_around_detections(image: Image, detections: Detections, padding: float = 0.0) -> Image:
    """Cut out one image per detection.

    Produces a batch from a single image, which every node downstream then runs once per crop --
    the engine's fan-out reads a list on a port as many values on one wire.
    """
    return transforms.crop_around_detections(image, detections, padding=padding)


# --- Detections only -----------------------------------------------------------------------------

@node(category="Transform")
def pad_detections(detections: Detections, padding: float = 0.1,
                   reference: REFERENCE = "shorter") -> Detections:
    """Grow every box by a fraction of its own size."""
    return transforms.add_padding(detections, padding, reference=reference)


@node(category="Transform")
def bbox_from_keypoints(detections: Detections,
                        keypoints: Optional[str] = None) -> Detections:
    """Redraw each box to enclose its keypoints. Unset uses all of them."""
    return transforms.update_bbox_from_keypoints(
        detections, keypoint_names=comma_separated(keypoints) or None)


def _box(image, left: int, top: int, right: Optional[int], bottom: Optional[int]) -> list:
    """A crop rectangle, where an unset right or bottom edge means the image's own."""
    height, width = image.shape[:2]
    return [left, top, width if right is None else right, height if bottom is None else bottom]
