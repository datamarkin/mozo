# SPDX-License-Identifier: Apache-2.0
"""Preprocessing for SAM 3.

Images arrive already decoded, as RGB ``uint8`` arrays -- mozo decodes in one place
(:mod:`mozo.image`) so exactly one piece of code decides channel order.

Four things here are easy to get wrong quietly.

**SAM 3 squashes, it does not letterbox.** The image goes straight to 1008x1008, distorting the
aspect ratio, with no padding to undo afterwards. Boxes and masks scale back by independent x and
y factors.

**The resize happens in uint8, not in float.** Upstream's transform is ``ToDtype(uint8) ->
Resize -> ToDtype(float32, scale=True) -> Normalize``, so the interpolation result is rounded
back to 8 bits *before* being divided by 255. Resizing in float instead -- the obvious way, and
what SAM 2 does -- shifts roughly every second pixel by 1/255. Reproduced here exactly, verified
bit-for-bit against ``torchvision.transforms.v2.Resize``.

**The resize antialiases.** ``v2.Resize`` defaults to ``antialias=True`` on tensors;
``cv2.INTER_LINEAR`` does not. On this image the difference reaches 72 grey levels.

**Normalisation is to [-1, 1], not ImageNet.** Mean and standard deviation are 0.5 on every
channel. Reaching for the ImageNet constants every other family uses is wrong here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import SPEC, Spec

__all__ = ["preprocess", "to_original"]


def preprocess(image: np.ndarray, spec: Spec = SPEC) -> torch.Tensor:
    """Scale an RGB image to the encoder's square and normalise it.

    Args:
        image: ``HxWx3`` RGB ``uint8`` array, as :func:`mozo.image.load_image` returns.
        spec: Vision geometry, for the resolution and the normalisation constants.

    Returns:
        The ``(1, 3, size, size)`` normalised float batch.

    Raises:
        ValueError: If the array is not ``HxWx3``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB array, got shape {image.shape}")
    size = spec.trunk.image_size

    chw = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
    # Interpolate in float, then round back to uint8 -- see the module docstring. ``round`` and
    # not truncation: truncating moves 1.5 million pixels on a single 1920x1281 photograph.
    resized = F.interpolate(
        chw[None].float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )
    resized = resized.round_().clamp_(0, 255).to(torch.uint8)

    # Multiply by the reciprocal, do not divide by 255. ``torchvision``'s uint8-to-float
    # conversion is ``image.to(dtype).mul_(1.0 / 255)``, and 1/255 is not exactly representable,
    # so the two disagree in the last bit on 40 percent of pixels.
    batch = resized.float().mul_(1.0 / 255.0)
    mean = batch.new_tensor(spec.mean).view(1, 3, 1, 1)
    std = batch.new_tensor(spec.std).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std)



def to_original(masks: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Resize mask logits back to the source image.

    Args:
        masks: ``(B, C, h, w)`` logits.
        shape: The source image's ``(height, width)``.

    Returns:
        ``(B, C, height, width)`` logits. Thresholding stays the caller's decision.
    """
    return F.interpolate(masks.float(), shape, mode="bilinear", align_corners=False)
