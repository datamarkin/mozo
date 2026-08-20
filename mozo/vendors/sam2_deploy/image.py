# SPDX-License-Identifier: Apache-2.0
"""Preprocessing, prompt scaling, and the mapping of masks back to the source image.

Images arrive already decoded, as RGB ``uint8`` arrays. Nothing here reads a file: mozo decodes in
one place (:mod:`mozo.image`) so exactly one piece of code decides channel order.

Two things in here are easy to get wrong and expensive to get wrong quietly.

**SAM 2 squashes, it does not letterbox.** The image is resized straight to ``1024x1024``,
distorting the aspect ratio, and there is no padding to undo afterwards. Every other family mozo
serves letterboxes, so the instinct to subtract a pad is wrong here -- prompts and masks scale by
independent x and y factors instead.

**The resize antialiases.** Upstream resizes with ``torchvision.transforms.Resize``, which on a
tensor defaults to ``antialias=True``; ``cv2.INTER_LINEAR`` does not antialias. Downscaling a
photograph to 1024 through the wrong one shifts every feature slightly, which shows up as masks
that are close enough to look right and wrong enough to fail a parity check. This module calls the
same ``F.interpolate`` that torchvision delegates to, with the same flags.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

#: ImageNet statistics the trunk was trained under.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: Above this a mask logit is foreground. Upstream's ``SAM2ImagePredictor.mask_threshold``.
MASK_THRESHOLD = 0.0


def preprocess(image: np.ndarray, size: int) -> torch.Tensor:
    """Scale an RGB image to a ``size`` square and normalise it for the trunk.

    Args:
        image: ``HxWx3`` RGB ``uint8`` array, as :func:`mozo.image.load_image` returns.
        size: Square side the encoder runs at.

    Returns:
        The ``(1, 3, size, size)`` normalised float batch.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB array, got shape {image.shape}")
    chw = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.0)
    batch = F.interpolate(
        chw[None], size=(size, size), mode="bilinear", align_corners=False, antialias=True
    )
    mean = batch.new_tensor(MEAN).view(1, 3, 1, 1)
    std = batch.new_tensor(STD).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std)


def to_model_coords(coords: np.ndarray, shape: tuple[int, int], size: int) -> torch.Tensor:
    """Scale x, y prompt coordinates from source pixels into the encoder's square.

    Args:
        coords: ``(..., 2)`` x, y in the source image's pixels.
        shape: The source image's ``(height, width)``.
        size: Square side the encoder runs at.

    Returns:
        The same shape, as a float tensor in ``[0, size]``.
    """
    height, width = shape
    # ``torch.tensor`` copies, which is what keeps the in-place divides below off the caller's
    # array -- ``as_tensor`` would share memory with a float32 input and scale it under them.
    scaled = torch.tensor(np.asarray(coords, dtype=np.float32))
    # Normalise then scale, rather than multiplying by the combined ratio. The two differ in the
    # last bits of a float, which is enough to move a box corner across a pixel boundary and put
    # one pixel of the mask on the other side of the threshold.
    scaled[..., 0] /= width
    scaled[..., 1] /= height
    return scaled * size


def to_original(masks: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Resize low-resolution mask logits back to the source image.

    Args:
        masks: ``(B, C, h, w)`` logits, as the decoder returns them.
        shape: The source image's ``(height, width)``.

    Returns:
        ``(B, C, height, width)`` logits. Thresholding is the caller's decision, so that a caller
        who wants soft masks is not handed something already flattened to booleans.
    """
    return F.interpolate(masks.float(), shape, mode="bilinear", align_corners=False)
