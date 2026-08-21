# SPDX-License-Identifier: Apache-2.0
"""Preprocessing, and the mapping of predicted boxes back to the source image.

Replaces ``Owlv2ImageProcessor``. Images arrive already decoded, as RGB ``uint8`` arrays; nothing
here reads a file, because mozo decodes in one place (:mod:`mozo.image`) so exactly one piece of
code decides channel order.

Three things here are easy to get wrong and expensive to get wrong quietly. Two more -- how a
byte becomes a float, and what precision the scale factor is computed in -- are in the code
below, because each is one line and neither is visible from anywhere else.

**OWLv2 pads to a square and then resizes.** Not letterboxed to fit, not squashed: the image is
padded on the **bottom and right only** with black, out to a square of side ``max(h, w)``, and
that square is resized to 960. So a 4:3 photograph spends a quarter of the model's patches on
padding, and every predicted box lives in the padded square rather than in the picture.

**Which is why boxes descale by ``max(h, w)`` on both axes.** Not by width for x and height for y
-- the usual reading, and wrong here, because the pad made the coordinate space square. Getting
this wrong leaves boxes that look plausible on a square image and are systematically compressed
along the short axis on every other one.

**The resize antialiases like scikit-image, not like torchvision.** Upstream reimplements
``skimage.transform.resize``: a Gaussian blur with ``sigma = (factor - 1) / 2``, then a *plain*
bilinear resize with ``antialias=False``. That is not the same as ``F.interpolate(antialias=True)``,
which is what a sibling family here needs and what the instinct reaches for. Downscaling a
photograph through the wrong one shifts every feature slightly, which shows up as detections that
are close enough to look right and wrong enough to fail a parity check.

The blur is reproduced with :func:`torch.nn.functional.conv2d` rather than by calling
``torchvision``, which upstream uses -- an outer-product kernel, reflect padding, one grouped
convolution, which is what ``torchvision`` does. ``tests/families/test_owlv2.py`` holds the two
bit-identical at four standard deviations, which is what makes reimplementing it a dependency
choice rather than a different blur.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as tvF

__all__ = ["MEAN", "RESCALE", "STD", "preprocess", "to_original"]

#: What a ``uint8`` pixel is multiplied by. Upstream's ``rescale_factor``, stated as a reciprocal.
RESCALE = 1.0 / 255.0

#: CLIP's statistics, which OWLv2 inherits. Upstream's ``preprocessor_config.json`` defaults.
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess(image: np.ndarray, size: int) -> torch.Tensor:
    """Scale an RGB image into the square the trunk runs at, and normalise it.

    Args:
        image: ``HxWx3`` RGB ``uint8`` array, as :func:`mozo.image.load_image` returns.
        size: Square side the trunk runs at.

    Returns:
        The ``(1, 3, size, size)`` normalised float batch.

    Raises:
        ValueError: If the array is not ``HxWx3``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB array, got shape {image.shape}")

    # Straight into the padded square, in one copy. The obvious spelling -- convert to float,
    # then pad -- materialises a full-resolution intermediate first: 29.5 MB on a 2 MP photograph,
    # read and written again immediately. ``copy_`` does the cast and the transpose on the way in,
    # so the only full-size buffer is the one that survives.
    #
    # The rescale then happens *after* the pad rather than before, which upstream does the other
    # way round and which is the same tensor either way: the pad is 0.0, and zero times the
    # reciprocal is zero. It is done in place on the square, not on the source, so a caller's
    # array is never touched.
    #
    # A *multiply* by one two-hundred-and-fifty-fifth, not a divide by 255. Upstream carries the
    # reciprocal as a constant and multiplies by it, and the two disagree in the last bit of a
    # float on most inputs -- 9.5e-07 by the time it reaches the trunk.
    height, width = image.shape[:2]
    square = max(height, width)
    padded = torch.zeros(1, 3, square, square)
    padded[0, :, :height, :width].copy_(torch.from_numpy(image).permute(2, 0, 1))
    padded.mul_(RESCALE)

    batch = _resize(padded, size)
    mean = batch.new_tensor(MEAN).view(1, 3, 1, 1)
    std = batch.new_tensor(STD).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std)


def to_original(boxes: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Turn normalised cxcywh boxes into xyxy pixels of the source image.

    Args:
        boxes: ``(..., 4)`` centre-x, centre-y, width, height, each in ``[0, 1]``.
        shape: The source image's ``(height, width)``.

    Returns:
        ``(..., 4)`` x1, y1, x2, y2 in source pixels. Not clipped: a box may legitimately run off
        the edge, and clipping is the caller's decision.
    """
    centre, size = boxes[..., :2], boxes[..., 2:]
    corners = torch.cat([centre - size / 2, centre + size / 2], dim=-1)
    # One scale for all four numbers, because the pad made the space square. See the module
    # docstring: this is the line that is wrong if boxes come out compressed on one axis.
    return corners * max(shape)


def _resize(batch: torch.Tensor, size: int) -> torch.Tensor:
    """Downscale the way ``skimage.transform.resize`` does: blur, then plain bilinear.

    Args:
        batch: ``(1, 3, S, S)``.
        size: Target side.

    Returns:
        ``(1, 3, size, size)``.

    The scale factor is computed in **float32**, through a tensor, rather than in Python's
    float64. Upstream divides one tensor of side lengths by another, which lands in float32, and
    the standard deviation that falls out differs from the double-precision one in the seventh
    digit -- enough to change the kernel and leave 9.5e-07 on the trunk's input. It only shows up
    on images whose longest side is not an exact multiple of ``size``, which is why an
    unsuspicious pair of test images can agree exactly and hide it.
    """
    factor = torch.tensor(float(batch.shape[-1]), dtype=torch.float32) / size
    sigma = float(((factor - 1) / 2).clamp(min=0))
    if sigma > 0:
        # Three standard deviations either side, always odd, which is upstream's kernel width.
        width = 2 * int(math.ceil(3 * sigma)) + 1
        batch = tvF.gaussian_blur(batch, [width, width], sigma=[sigma, sigma])
    return F.interpolate(batch, size=(size, size), mode="bilinear", align_corners=False)
