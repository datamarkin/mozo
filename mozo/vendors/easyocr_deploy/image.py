"""Everything between an image and a tensor, on both sides of the pipeline.

Two separate preprocessing paths live here because the two networks want different things.
The detector takes the whole page, resized once and padded to a multiple of 32. The recogniser
takes one rectified line at a time, greyscale, at a fixed height.

The order of operations in both is upstream's, down to which library does which resize. That is
not fussiness: :func:`crop_to_height` and :func:`align` each resize the same crop, with
different filters, and collapsing them into one changes pixels.
"""

from __future__ import annotations

__all__ = ["CANVAS_SIZE", "MODEL_HEIGHT", "for_detector", "line_image", "align",
           "adjust_contrast_grey"]

import math

import cv2
import numpy as np
import torch

#: Longest side the detector will look at. Bigger pages are scaled down to this before the
#: forward pass and the boxes are scaled back up afterwards.
CANVAS_SIZE = 2560

#: The recogniser's fixed input height. Crops are resized to this and their width follows the
#: aspect ratio.
MODEL_HEIGHT = 64

#: ImageNet statistics, in the 0-255 domain the raw uint8 image is already in -- so this is one
#: subtract and one divide rather than a rescale followed by a normalise.
_MEAN = np.array([0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0], dtype=np.float32)
_STD = np.array([0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0], dtype=np.float32)

# The same two constants as tensors. Broadcasting a ``(3,)`` over a contiguous ``(H, W, 3)``
# lands in numpy's slow inner loop -- 34 of the 37 ms this function spent on a 2 MP page. The
# subtract and divide are elementwise and correctly rounded either way, so the result is
# bit-identical; only the loop differs.
_MEAN_T = torch.from_numpy(_MEAN)
_STD_T = torch.from_numpy(_STD)

# Upstream writes `interpolation=Image.Resampling.LANCZOS` when resizing a crop with OpenCV.
# PIL's LANCZOS is 1 and so is cv2.INTER_LINEAR, while cv2's own Lanczos is 4 -- so that call
# has always been bilinear, whatever it reads like. It is spelled correctly here and left
# bilinear, because matching the published model matters more than matching the identifier
# somebody meant to type. Changing this to INTER_LANCZOS4 breaks parity; see PROVENANCE.md.
_CROP_INTERPOLATION = cv2.INTER_LINEAR


def for_detector(image: np.ndarray) -> tuple[torch.Tensor, float]:
    """An RGB page to the detector's input tensor, and the scale that was applied.

    Returns ``(batch, ratio)``. Multiply a coordinate in the network's output space by
    ``1 / ratio`` to get back to the original image -- :func:`.boxes.rescale` does exactly that.
    """
    height, width = image.shape[:2]
    longest = max(height, width)

    target = min(longest, CANVAS_SIZE)
    ratio = target / longest
    target_h, target_w = int(height * ratio), int(width * ratio)
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # The network downsamples by 32, so the input has to divide by it. The padding goes bottom
    # and right and is zeros -- not an edge replicate -- which reads as black, and black is
    # background to a detector trained on documents.
    padded_h = target_h + (-target_h % 32)
    padded_w = target_w + (-target_w % 32)
    canvas = np.zeros((padded_h, padded_w, image.shape[2]), dtype=np.float32)
    canvas[:target_h, :target_w] = resized

    # Through a tensor that shares ``canvas``'s memory, so the return below is unchanged.
    torch.from_numpy(canvas).sub_(_MEAN_T).div_(_STD_T)
    return torch.from_numpy(canvas.transpose(2, 0, 1)[None]), ratio


def _aspect(width: float, height: float) -> float:
    """Width over height, flipped up the other way when the crop is taller than it is wide.

    A ratio below one would resize to a sub-pixel width, so vertical text is measured on its
    long axis instead and rotated into shape by the caller.
    """
    ratio = width / height
    return 1.0 / ratio if ratio < 1.0 else ratio


def _four_point_transform(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp a quadrilateral onto an upright rectangle of its own longest sides."""
    tl, tr, br, bl = quad
    width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                      dtype="float32")
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(image, matrix, (width, height))


def crop_to_height(crop: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    """Resize one crop to :data:`MODEL_HEIGHT`, keeping its aspect ratio."""
    ratio = width / height
    if ratio < 1.0:
        # Taller than wide. Resized so its *width* is MODEL_HEIGHT and its height follows,
        # which leaves the text running down a tall strip rather than squeezed into a short
        # one. The recogniser still reads it left to right and mostly gets it wrong; upstream
        # does this, so it is what parity means.
        ratio = _aspect(width, height)
        return cv2.resize(crop, (MODEL_HEIGHT, int(MODEL_HEIGHT * ratio)),
                          interpolation=_CROP_INTERPOLATION), ratio
    return cv2.resize(crop, (int(MODEL_HEIGHT * ratio), MODEL_HEIGHT),
                      interpolation=_CROP_INTERPOLATION), ratio


def line_image(line, grey: np.ndarray, *, is_free: bool):
    """Cut one located line out of the greyscale page.

    Returns ``(quad, crop, width)``, or ``None`` if the line is degenerate. ``width`` is what the
    crop will be padded to -- a whole multiple of :data:`MODEL_HEIGHT`, which for a single line is
    that line's own aspect ratio rounded up.

    One line at a time, because that is what upstream does whenever it is on CPU or its batch
    size is one, which is its default. Its other path pads a whole page's crops to the widest
    among them; this one gives each line a width that depends on nothing but itself.

    A tilted line is rectified with a perspective warp; a level one is a plain slice, clamped to
    the page.
    """
    max_y, max_x = grey.shape

    if is_free:
        warped = _four_point_transform(grey, np.array(line, dtype="float32"))
        height, width = warped.shape[:2]
        if height <= 0 or width <= 0 or int(MODEL_HEIGHT * _aspect(width, height)) == 0:
            return None
        crop, ratio = crop_to_height(warped, width, height)
        quad = line
    else:
        x_min, x_max = max(0, line[0]), min(line[1], max_x)
        y_min, y_max = max(0, line[2]), min(line[3], max_y)
        width, height = x_max - x_min, y_max - y_min
        if height <= 0 or width <= 0 or int(MODEL_HEIGHT * _aspect(width, height)) == 0:
            return None
        crop, ratio = crop_to_height(grey[y_min:y_max, x_min:x_max], width, height)
        quad = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

    return quad, crop, math.ceil(max(ratio, 1.0)) * MODEL_HEIGHT


def _contrast(grey: np.ndarray) -> tuple[float, float, float]:
    """Spread between the 10th and 90th percentile, and those two percentiles."""
    high, low = np.percentile(grey, 90), np.percentile(grey, 10)
    return (high - low) / np.maximum(10, high + low), high, low


def adjust_contrast_grey(grey: np.ndarray, target: float = 0.4) -> np.ndarray:
    """Stretch a washed-out crop, or return it untouched if it is already contrasty enough.

    This is the second half of the low-confidence retry: a crop the recogniser was unsure about
    is put through here and read again, and the better of the two answers wins.
    """
    contrast, high, low = _contrast(grey)
    if contrast >= target:
        return grey
    scale = 200.0 / np.maximum(10, high - low)
    stretched = (grey.astype(int) - low + 25) * scale
    return np.clip(stretched, 0, 255).astype(np.uint8)


def align(crop: np.ndarray, width: int, contrast: float = 0.0) -> torch.Tensor:
    """One crop to a ``(1, 1, MODEL_HEIGHT, width)`` batch, scaled to [-1, 1].

    The crop is resized a second time here, with PIL's bicubic filter rather than the bilinear
    one :func:`crop_to_height` used. Two resizes with two filters is what the published model
    does and collapsing them moves pixels.

    Padding replicates the crop's **last column** across the remainder rather than filling with
    zeros. Zero padding would read as a black bar, which the recogniser is happy to decode as
    characters; a repeated final column reads as more of whatever the line ended with.
    """
    from PIL import Image

    if contrast > 0:
        crop = adjust_contrast_grey(crop, target=contrast)

    image = Image.fromarray(crop, "L")
    source_w, source_h = image.size
    target_w = min(math.ceil(MODEL_HEIGHT * source_w / source_h), width)
    resized = image.resize((target_w, MODEL_HEIGHT), Image.BICUBIC)

    tensor = torch.from_numpy(np.array(resized, dtype=np.float32) / 255.0)[None]
    tensor.sub_(0.5).div_(0.5)

    batch = torch.zeros(1, 1, MODEL_HEIGHT, width)
    batch[0, :, :, :target_w] = tensor
    if target_w < width:
        batch[0, :, :, target_w:] = tensor[:, :, target_w - 1:target_w]
    return batch
