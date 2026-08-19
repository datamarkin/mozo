# SPDX-License-Identifier: Apache-2.0
"""Letterboxing and the mapping back to original image coordinates.

Images arrive already decoded, as RGB ``uint8`` arrays. Nothing here reads a file: mozo decodes in
one place (:mod:`mozo.image`) so that exactly one piece of code decides channel order, and a numpy
array carries nothing that would let this module check the decision afterwards.

The resizing and padding below are model maths, not image handling -- they reproduce the letterbox
the network was trained under, and changing the interpolation would cost accuracy silently.

There is no ``suppress`` here, and its absence is the point. This family's head is trained to fire
once per object, so the network returns a detection list directly and no box ever suppresses
another. What the siblings do between the forward pass and the coordinate mapping, this family
does not do at all.

The placement is reported floored, and :func:`to_original` subtracts exactly what was written. The
harvest returned the unrounded half and argued for it: "rounding it would shift every box by up to
half a pixel back in the original image whenever the total padding is odd". That has it backwards.
The content is written at ``math.floor(spare_y)``, so the floor *is* the offset between the two
coordinate systems; the unrounded value describes a placement that never happened, and costs
``0.5 / gain`` source pixels -- 1.5 px on mozo's fixture photograph, on every box.
``tests/test_vendor_agreement.py`` measures the border actually written and holds every vendor to
it.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

#: Grey used for the letterbox border.
BORDER = 114


def letterbox(image: np.ndarray, size: int) -> tuple[torch.Tensor, float, int, int]:
    """Scale an RGB image to fit a ``size`` square, pad it out, and return the batch and placement.

    Args:
        image: ``HxWx3`` RGB ``uint8`` array, as :func:`mozo.image.load_image` returns.
        size: Side of the square the network is run at.

    Returns:
        The ``(1, 3, size, size)`` float batch scaled to ``[0, 1]``, the scale factor applied, and
        the left and top padding -- everything :func:`to_original` needs to undo the placement.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB array, got shape {image.shape}")
    height, width = image.shape[:2]
    gain = min(size / height, size / width)
    scaled = cv2.resize(image, (round(width * gain), round(height * gain)), interpolation=cv2.INTER_LINEAR)
    spare_x, spare_y = (size - scaled.shape[1]) / 2, (size - scaled.shape[0]) / 2
    padded = cv2.copyMakeBorder(
        scaled,
        math.floor(spare_y),
        math.ceil(spare_y),
        math.floor(spare_x),
        math.ceil(spare_x),
        cv2.BORDER_CONSTANT,
        value=(BORDER, BORDER, BORDER),
    )
    # ``ascontiguousarray`` with a dtype change always copies, so ``chw`` is fresh and the
    # scaling can be done in place rather than allocating a second copy of the batch.
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(chw).div_(255.0)[None], gain, math.floor(spare_x), math.floor(spare_y)


def to_original(boxes: torch.Tensor, gain: float, pad_x: int, pad_y: int, shape: tuple[int, int]) -> torch.Tensor:
    """Undo the letterbox placement and clip the boxes to the original image."""
    height, width = shape
    boxes = (boxes - boxes.new_tensor([pad_x, pad_y, pad_x, pad_y])) / gain
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    return boxes
