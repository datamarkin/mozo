# ------------------------------------------------------------------------
# BEN2 -- Background Erase Network
# Copyright (c) 2025 Prama LLC. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
#
# refine_foreground() is Photoroom's blur-fusion foreground estimator, which
# BEN2.py credits to https://github.com/Photoroom/fast-foreground-estimation
# ------------------------------------------------------------------------
"""Pixels in, matte out: everything either side of the network.

Extracted from ``BEN2.py`` lines 1161-1368. Small, and almost every line of it is a trap.

**The resize squashes, and that is the model.** Upstream resizes to a square 1024x1024 with PIL's
LANCZOS filter and does not preserve aspect ratio. A 4000x500 panorama is squashed, matted, and
unsquashed. Letterboxing instead is the obvious improvement and moves every pixel of the result,
because the weights were trained on squashed images.

**PIL's LANCZOS is not reproducible in torch.** Its filter support scales with the downsampling
factor; neither ``cv2.resize`` nor ``F.interpolate(antialias=True)`` matches it. So the resize
stays on PIL, and that is why Pillow is a dependency of this package rather than an accident.

**Upstream defines ``rgb_loader_refiner`` twice** -- at lines 1161 and 1350 -- and Python takes
the second. This module reproduces the second. They differ in two ways that matter: the later one
converts to RGB *before* the resize rather than after (different pixels for an image with alpha),
and it computes ``ImageOps.exif_transpose(...)`` into a local that the very next line overwrites,
so the EXIF correction is dead. An upright-looking photo with an orientation tag is matted
sideways. That is upstream's behaviour, it is what the published weights were exercised with, and
it is reproduced rather than fixed -- ``PROVENANCE.md`` records it as a divergence from the
obvious reading.
"""

from __future__ import annotations

__all__ = ["ALPHA_EPSILON", "postprocess", "preprocess", "refine_foreground"]

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import INPUT, MEAN, STD

#: Below this, the matte is treated as flat and the contrast stretch is skipped rather than
#: dividing by it. Upstream divides unconditionally, so a frame the model reads as uniform --
#: an empty sky, a blank scan, a 1x1 image -- produces ``inf``/``nan`` cast to uint8, which is
#: silent garbage rather than an error. See :func:`postprocess`.
ALPHA_EPSILON: float = 1e-8


def preprocess(rgb: np.ndarray, device: str | torch.device = "cpu",
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``(H, W, 3)`` uint8 RGB -> ``(1, 3, 1024, 1024)`` normalised.

    Equivalent to upstream's ``rgb_loader_refiner`` followed by ``img_transform32``, with the
    dtype made an argument instead of being chosen by ``torch.cuda.is_available()``. Upstream
    picks float16 on any machine with a CUDA device and float32 everywhere else, which makes the
    published model two models selected by a global property; here the caller says which.

    Args:
        rgb: ``(H, W, 3)`` uint8. Already RGB -- the conversion happens before this package, which
            matches upstream's order of convert-then-resize.
        device: Where to put the tensor.
        dtype: Compute dtype. ``float32`` is the only one this package's parity is claimed for.

    Returns:
        torch.Tensor: ``(1, 3, 1024, 1024)``.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB array, got {rgb.shape} {rgb.dtype}")

    square = Image.fromarray(rgb).resize((INPUT, INPUT), resample=Image.LANCZOS)

    # torchvision's ToTensor, written out: permute, float, divide. Written out rather than
    # imported so this package does not depend on torchvision for three lines.
    # np.array, not np.asarray: PIL hands back a read-only view, and torch.from_numpy on one
    # warns that writing through the tensor is undefined. A copy of a 1024x1024x3 uint8 buffer
    # costs 3 MB against ~5 s of inference.
    tensor = torch.from_numpy(np.array(square)).permute(2, 0, 1).contiguous().to(torch.float32).div(255)
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
    tensor = tensor.sub(mean).div(std)

    return tensor.unsqueeze(0).to(device=device, dtype=dtype)


def postprocess(matte: torch.Tensor, size: tuple[int, int], *, stretch: bool = True) -> np.ndarray:
    """``(1, 1, 1024, 1024)`` sigmoid -> ``(H, W)`` uint8 alpha.

    Args:
        matte: The network's output, still at 1024x1024.
        size: ``(height, width)`` to resize to -- the original image's, in numpy order.
        stretch: Reproduce upstream's per-image min-max normalisation.

    Returns:
        np.ndarray: ``(H, W)`` uint8.

    **What ``stretch`` does, and why it is not a detail.** Upstream ends with

    .. code-block:: python

        ma = torch.max(result); mi = torch.min(result)
        result = (result - mi) / (ma - mi)

    The network emits a sigmoid -- a per-pixel probability with a meaningful 0.5. This rescales it
    so that the most-foreground pixel in *this* image becomes 255 and the least becomes 0. The
    returned alpha is therefore a per-image contrast stretch of a probability: compare it within
    an image, never across two. An image whose most confident pixel scored 0.6 still comes back
    with pixels at 255.

    ``stretch=False`` returns the calibrated sigmoid, which is what you want for thresholding or
    for combining this model with another.

    The denominator is guarded, which upstream's is not. When the matte is flat to within
    :data:`ALPHA_EPSILON` the stretch is skipped and the constant is returned scaled, rather than
    dividing by zero and casting the resulting ``nan`` to uint8.
    """
    resized = torch.squeeze(F.interpolate(matte.float(), size=size, mode="bilinear"), 0)

    if stretch:
        mi, ma = torch.min(resized), torch.max(resized)
        if (ma - mi) > ALPHA_EPSILON:
            resized = (resized - mi) / (ma - mi)

    # Truncation, not rounding -- ``.astype(np.uint8)`` floors, and so does torchvision's
    # ToPILImage, which upstream's other path uses. Measured, not assumed: the two agree.
    array = (resized * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return np.squeeze(array, axis=2)


def _blur_fusion(image: np.ndarray, foreground: np.ndarray, background: np.ndarray,
                 alpha: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """One pass of Photoroom's blur-fusion foreground estimator. Float arrays in ``[0, 1]``."""
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]
    blurred_fa = cv2.blur(foreground * alpha, (r, r))
    blurred_f = blurred_fa / (blurred_alpha + 1e-5)
    blurred_b1a = cv2.blur(background * (1 - alpha), (r, r))
    blurred_b = blurred_b1a / ((1 - blurred_alpha) + 1e-5)
    estimated = blurred_f + alpha * (image - alpha * blurred_f - (1 - alpha) * blurred_b)
    return np.clip(estimated, 0, 1), blurred_b


def refine_foreground(rgb: np.ndarray, alpha: np.ndarray, r: int = 90) -> np.ndarray:
    """Estimate unmixed foreground colours. ``(H, W, 3)`` uint8 and ``(H, W)`` uint8 in.

    **This changes RGB, not alpha.** Along a soft edge every pixel is a mix of foreground and
    background; composited onto a new background the old one shows through as a fringe. This
    estimates what the foreground colour would have been on its own, so the fringe goes away.

    Not BEN2's own work: upstream credits Photoroom's fast-foreground-estimation and runs it
    twice, at ``r=90`` then ``r=6``. Both the credit and the two radii are reproduced.

    Two box blurs over the full-resolution image is not free, which is why this is an explicit
    argument upstream and an explicit argument here rather than a default.
    """
    image = rgb.astype(np.float64) / 255.0
    a = (alpha.astype(np.float64) / 255.0)[:, :, None]

    foreground, blurred_b = _blur_fusion(image, image, image, a, r)
    foreground, _ = _blur_fusion(image, foreground, blurred_b, a, 6)

    return (foreground * 255.0).astype(np.uint8)
