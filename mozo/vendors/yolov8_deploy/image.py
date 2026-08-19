# SPDX-License-Identifier: Apache-2.0
"""Letterboxing, box filtering and the mapping back to original image coordinates.

Images arrive already decoded, as RGB ``uint8`` arrays. Nothing here reads a file: mozo decodes in
one place (:mod:`mozo.image`) so that exactly one piece of code decides channel order, and a numpy
array carries nothing that would let this module check the decision afterwards.

The resizing and padding below are model maths, not image handling -- they reproduce the letterbox
the network was trained under, and changing the interpolation would cost accuracy silently.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

#: Grey used for the letterbox border.
BORDER = 114


def letterbox(image: np.ndarray, size: int) -> tuple[torch.Tensor, float, float, float]:
    """Scale an RGB image to fit a ``size`` square, pad it out, and return the batch tensor and placement.

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
    # scaling can be in place. Dividing out of place allocated a second 4.9 MB tensor per call.
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(chw).div_(255.0)[None], gain, math.floor(spare_x), math.floor(spare_y)


def suppress(prediction: torch.Tensor, conf: float, iou: float, max_det: int) -> tuple[torch.Tensor, ...]:
    """Keep the confident, non-overlapping detections of one image's raw head output.

    ``prediction`` is ``(4 + classes, anchors)`` holding centre-form boxes and per-class scores.
    """
    boxes, scores = prediction[:4].T, prediction[4:].T
    best, labels = scores.max(1)
    chosen = best > conf
    boxes, best, labels = boxes[chosen], best[chosen], labels[chosen]

    half = boxes[:, 2:] / 2
    corners = torch.cat((boxes[:, :2] - half, boxes[:, :2] + half), 1)
    if not corners.numel():
        return corners, best, labels

    # Shift each class into its own coordinate band so boxes of different classes cannot suppress one another.
    band = (corners.max() - corners.min() + 1) * labels[:, None]
    shifted = corners + band
    areas = (corners[:, 2] - corners[:, 0]) * (corners[:, 3] - corners[:, 1])
    order = best.argsort(descending=True, stable=True)
    kept = []
    while order.numel() and len(kept) < max_det:
        first = order[0]
        kept.append(first)
        rest = order[1:]
        top_left = torch.maximum(shifted[first, :2], shifted[rest, :2])
        bottom_right = torch.minimum(shifted[first, 2:], shifted[rest, 2:])
        overlap = (bottom_right - top_left).clamp(min=0).prod(1)
        order = rest[overlap / (areas[first] + areas[rest] - overlap) <= iou]
    index = torch.stack(kept) if kept else torch.zeros(0, dtype=torch.long, device=corners.device)
    return corners[index], best[index], labels[index]


def to_original(boxes: torch.Tensor, gain: float, pad_x: float, pad_y: float, shape: tuple[int, int]) -> torch.Tensor:
    """Undo the letterbox placement and clip the boxes to the original image."""
    height, width = shape
    boxes = (boxes - boxes.new_tensor([pad_x, pad_y, pad_x, pad_y])) / gain
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
    return boxes
