# SPDX-License-Identifier: Apache-2.0
"""Turning a frame and some person boxes into model input.

Replaces ``VitPoseImageProcessor``'s preprocessing half. Images arrive already decoded, as RGB
``uint8`` arrays; nothing here reads a file, because mozo decodes in one place
(:mod:`mozo.image`) so exactly one piece of code decides channel order.

**The crop is bigger than the box.** :func:`box_to_center_and_scale` widens or heightens the box
to the model's 3:4 aspect ratio and then pads it by a further 1.25x. For a 50x140 person that
comes out around 131x175 -- roughly forty pixels of width and thirty-five of height the detector's
box never contained, taken from the surrounding frame. This is why this package is handed the full
frame and a box rather than a crop: a caller who cropped tightly has already discarded the pixels
the model wants, and a wrist just outside the box goes with them.

**The warp is exact bilinear, written here rather than called.** Upstream reaches for
``scipy.ndimage.affine_transform``; mozo does not depend on SciPy, and would not add a dependency
of that size for one resample. :func:`warp` is the same operation -- inverse map, bilinear gather,
zero outside, round to ``uint8`` -- and ``tests/families/test_vitpose.py`` holds the two
bit-identical on real photographs, which is what makes reimplementing it a dependency choice
rather than a different warp.

``cv2.warpAffine`` was the obvious alternative, is already a mozo dependency, and is what the
original ViTPose used. It is not used here because OpenCV quantises sampling coordinates to 1/32
of a pixel: measured against an exactly-known answer it is off by 0.032 where this is off by
3e-14, and that difference reached 1.1 pixels in the final joint positions. Matching the
extraction source is worth more than matching the ancestor it approximates.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["ASPECT", "MEAN", "NORMALIZE_FACTOR", "PADDING_FACTOR", "RESCALE", "STD",
           "box_to_center_and_scale", "preprocess", "warp", "warp_matrix"]

#: ImageNet's statistics, which ViTPose inherits. The published ``preprocessor_config.json``.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: What a ``uint8`` pixel is multiplied by. Upstream's ``rescale_factor``, stated as a reciprocal.
RESCALE = 1.0 / 255.0

#: What a box's width and height are divided by to become a "scale". Upstream's
#: ``normalize_factor``: an mmpose convention with no geometric meaning, carried because the
#: numbers it produces are what :func:`~.postprocess.to_frame` multiplies back out.
NORMALIZE_FACTOR = 200.0

#: How much larger than the aspect-corrected box the crop is taken. Upstream's ``padding_factor``.
PADDING_FACTOR = 1.25

#: The input's aspect ratio, width over height. Every box is widened or heightened to match it
#: before the crop, so that a person is never stretched.
ASPECT = 192 / 256


def box_to_center_and_scale(box: np.ndarray, aspect: float = ASPECT
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Encode an ``xyxy`` box as the centre and scale the warp is built from.

    Args:
        box: ``(4,)`` as ``x1, y1, x2, y2`` in the frame's own pixels.
        aspect: Target width over height. The box is grown along whichever axis is short.

    Returns:
        ``(centre, scale)``, both ``(2,)`` float32. ``scale`` is in units of
        :data:`NORMALIZE_FACTOR` pixels, which is upstream's convention rather than a choice.
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    width, height = x2 - x1, y2 - y1
    center = np.array([x1 + width * 0.5, y1 + height * 0.5], dtype=np.float32)

    if width > aspect * height:
        height = width / aspect
    elif width < aspect * height:
        width = height * aspect

    scale = np.array([width / NORMALIZE_FACTOR, height / NORMALIZE_FACTOR], dtype=np.float32)
    return center, scale * PADDING_FACTOR


def warp_matrix(center: np.ndarray, scale: np.ndarray, height: int, width: int) -> np.ndarray:
    """The 2x3 affine taking the padded box onto the model's input rectangle.

    Upstream's ``get_warp_matrix``, with its rotation fixed at zero -- rotation is a training-time
    augmentation and no inference path sets it. The destination is ``size - 1`` rather than
    ``size``, which is the "unbiased data processing" convention: coordinates are treated as
    samples of a continuous field at pixel centres, so the *last* pixel maps to the far edge.
    Getting this off by one shifts every joint by half a pixel of the crop.

    The widths are upstream's too, and they are not incidental: the source rectangle is rounded to
    float32 before it is divided, and the division itself happens in float64. Computed the tidier
    way -- all float64, or all float32 -- the matrix lands one bit away, which turns into a
    one-level difference on about two pixels in ten thousand. Small, and no longer exact.
    """
    source = (scale * NORMALIZE_FACTOR).astype(np.float32)
    span = (center * 2.0).astype(np.float32)
    scale_x = np.float64(width - 1.0) / source[0]
    scale_y = np.float64(height - 1.0) / source[1]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = scale_x
    matrix[1, 1] = scale_y
    matrix[0, 2] = scale_x * (-0.5 * span[0] + 0.5 * source[0])
    matrix[1, 2] = scale_y * (-0.5 * span[1] + 0.5 * source[1])
    return matrix


def warp(image: np.ndarray, matrix: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resample *image* through *matrix* into a ``height x width`` crop.

    Bilinear, with anything outside the frame reading as black, and the result rounded back to
    ``uint8`` -- which is what upstream does, and is not free: the model sees 256 levels, not the
    exact interpolant. Reproduced rather than improved on, because the checkpoint was trained
    against it.

    Args:
        image: ``HxWx3`` RGB ``uint8``.
        matrix: ``(2, 3)`` mapping source pixels to destination pixels.
        height: Destination rows.
        width: Destination columns.

    Returns:
        ``height x width x 3`` RGB ``uint8``.
    """
    inverse = np.linalg.inv(np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float64))
    if inverse[0, 1] or inverse[1, 0]:
        raise ValueError(
            "warp is axis-aligned: the source x of a destination pixel must depend only on its "
            "column, and its y only on its row. Rotation is a training-time augmentation that "
            "no inference path sets, so ``warp_matrix`` never produces one."
        )

    # Two vectors rather than two grids. The transform has no rotation, so every column shares an
    # x and every row shares a y -- computing them per pixel would build six full-size float64
    # arrays (mgrid, x, y, left, top, fractions) to hold 192 and 256 distinct values.
    source_height, source_width = image.shape[:2]
    x = inverse[0, 0] * np.arange(width, dtype=np.float64) + inverse[0, 2]
    y = inverse[1, 1] * np.arange(height, dtype=np.float64) + inverse[1, 2]

    left = np.floor(x).astype(np.int64)
    top = np.floor(y).astype(np.int64)
    fraction_x = (x - left)[None, :, None]
    fraction_y = (y - top)[:, None, None]

    # **A sample outside the frame is black, not a blend toward black.** Upstream's border rule is
    # SciPy's ``mode="constant"``, which is not what the name suggests: a point at x = -0.2 does
    # not come back one fifth of the way to the first column, it comes back as the constant. Only
    # ``grid-constant`` interpolates against it. Blending is the natural way to write this and is
    # wrong by up to 68 levels along the edge of any crop that runs off the frame -- which is most
    # of them, since the crop is deliberately larger than the box.
    inside = ((y >= 0) & (y <= source_height - 1))[:, None] & \
             ((x >= 0) & (x <= source_width - 1))[None, :]

    def sample(row: np.ndarray, column: np.ndarray) -> np.ndarray:
        # Clamped only to keep the gather in bounds. Where the clamp bites, either the point is
        # outside and masked off below, or it sits exactly on the last row or column, where the
        # fraction is zero and the duplicated neighbour is multiplied away.
        gathered = image[np.clip(row, 0, source_height - 1)[:, None],
                         np.clip(column, 0, source_width - 1)[None, :]]
        return gathered.astype(np.float64)

    near_x, near_y = 1 - fraction_x, 1 - fraction_y
    upper = sample(top, left) * near_x + sample(top, left + 1) * fraction_x
    lower = sample(top + 1, left) * near_x + sample(top + 1, left + 1) * fraction_x
    crop = upper * near_y + lower * fraction_y
    crop *= inside[..., None]
    return np.rint(crop, out=crop).astype(np.uint8)


def preprocess(
    image: np.ndarray, boxes: np.ndarray, height: int, width: int
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Crop one person per box and normalise them into a batch.

    Args:
        image: ``HxWx3`` RGB ``uint8``, the whole frame.
        boxes: ``(N, 4)`` xyxy in the frame's own pixels.
        height: The model's input height.
        width: The model's input width.

    Returns:
        ``(pixel_values, centres, scales)`` -- a ``(N, 3, height, width)`` float32 tensor, and the
        two ``(N, 2)`` arrays :func:`~.postprocess.to_frame` needs to put joints back where they
        came from.
    """
    aspect = width / height
    centers = np.zeros((len(boxes), 2), dtype=np.float32)
    scales = np.zeros((len(boxes), 2), dtype=np.float32)
    crops = np.zeros((len(boxes), height, width, 3), dtype=np.uint8)
    for index, box in enumerate(boxes):
        center, scale = box_to_center_and_scale(np.asarray(box, dtype=np.float64), aspect=aspect)
        centers[index] = center
        scales[index] = scale
        crops[index] = warp(image, warp_matrix(center, scale, height, width), height, width)

    # Upstream folds the 1/255 into the statistics rather than applying it first, so a byte is
    # normalised in one step against ``mean/rescale`` and ``std/rescale``. Rescaling separately
    # gives the same number to within float32 rounding, and this gives it exactly.
    mean = torch.tensor([value / RESCALE for value in MEAN]).view(1, 3, 1, 1)
    std = torch.tensor([value / RESCALE for value in STD]).view(1, 3, 1, 1)
    # ``permute`` leaves the batch channels-last, which every consumer then has to undo: ONNX
    # Runtime repacks it into a fresh buffer on each call, and a torch matmul would too. Making it
    # contiguous once here costs a copy that was going to happen anyway.
    batch = torch.from_numpy(crops).permute(0, 3, 1, 2).float().contiguous()
    return batch.sub_(mean).div_(std), centers, scales
