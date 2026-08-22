# SPDX-License-Identifier: Apache-2.0
"""Grounding DINO's preprocessing: resize the short side to 800, cap the long side at 1333.

Unlike every other family in mozo, nothing here is letterboxed to a fixed square. The aspect
ratio is preserved, nothing is padded, and the tensor a photograph becomes therefore depends on
the photograph -- a 1920x1281 image runs at 1199x800 and a portrait one runs at 800x1199. That is
DETR's convention and Grounding DINO inherits it.

**The resize goes through PIL.** Upstream calls ``torchvision.transforms.functional.resize`` on a
``PIL.Image``, which is bilinear *with* antialiasing, and neither ``cv2.resize`` nor
``torch.nn.functional.interpolate`` reproduces it: PIL's filter has support that scales with the
downsampling factor, so the two disagree on almost every pixel of a 0.62x reduction. Measured on
the fixture photograph, swapping in ``interpolate(mode="bilinear", antialias=True)`` moves the
tensor by up to 0.07 and the final boxes by whole pixels, silently.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "preprocess", "resized_size"]

#: ImageNet statistics, in RGB order. mozo's image contract is already RGB, so nothing is swapped.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resized_size(height: int, width: int, short_side: int, max_side: int) -> tuple[int, int]:
    """Return the ``(height, width)`` the image is resized to.

    The short side goes to *short_side* unless that would push the long side past *max_side*, in
    which case the scale is reduced until it fits. Reproduces upstream's
    ``get_size_with_aspect_ratio`` exactly, including its early return when the image is already
    the right size -- which matters because the general branch would round it differently.
    """
    smallest = float(min(height, width))
    largest = float(max(height, width))

    if largest / smallest * short_side > max_side:
        short_side = int(round(max_side * smallest / largest))

    if (width <= height and width == short_side) or (height <= width and height == short_side):
        return height, width
    if width < height:
        return int(short_side * height / width), short_side
    return short_side, int(short_side * width / height)


def preprocess(image: np.ndarray, short_side: int, max_side: int) -> Tensor:
    """Turn an ``HxWx3`` RGB ``uint8`` array into the ``(3, h, w)`` tensor the model wants.

    Args:
        image: RGB ``uint8``, mozo's contract.
        short_side: Target for the shorter side. From :class:`~.config.Spec`, which owns it --
            defaulting it here would let a caller who forgot it silently preprocess at 800 for a
            variant published at something else, and a gate that forgot it would stay green.
        max_side: Ceiling for the longer side. Likewise.

    Returns:
        A normalised float32 tensor, no batch dimension.
    """
    height, width = image.shape[:2]
    target = resized_size(height, width, short_side, max_side)

    # Through PIL because upstream does; see the module docstring for what changes if it does not.
    resized = TF.resize(Image.fromarray(image), list(target))
    tensor = TF.to_tensor(resized)
    return TF.normalize(tensor, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
