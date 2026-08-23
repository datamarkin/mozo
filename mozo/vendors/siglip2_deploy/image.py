# SPDX-License-Identifier: Apache-2.0
"""SigLIP 2's image preprocessing: resize to a square, and normalise to roughly [-1, 1].

Shorter than CLIP's -- no aspect-preserving resize, no centre crop, no per-channel statistics --
and every step of it is a place a reasonable reading produces different pixels. The three
alternatives below were each measured against the reference on a real photograph; none is a
rounding difference you could dismiss.

============================================  ==================
instead of what upstream does                 max abs difference
============================================  ==================
``(x / 255 - 0.5) / 0.5``                     5.9e-08
``antialias=False``                           9.4e-01
resize the float tensor, not the uint8 one    3.9e-03
============================================  ==================

**The rescale is folded into the statistics, so nothing is ever divided by 255.**
``TorchvisionBackend._fuse_mean_std_and_rescale_factor`` multiplies mean and standard deviation by
``1 / rescale_factor`` and sets ``do_rescale = False``, so a 0.5 becomes 127.5 and one ``normalize``
does both jobs. Dividing by 255 first is the same arithmetic in a different order and it is not the
same float.

**The resize runs on uint8.** torchvision rounds and clamps back to eight bits before the
normalise, so resizing in floating point keeps precision the reference has already thrown away.

**Antialiasing is on.** ``TorchvisionBackend.resize`` defaults ``antialias=True``. This is the
difference that is not subtle: 0.94 on a tensor whose whole range is about two.

Upstream's own JAX preprocessing is ``resize({res})|value_range(-1, 1)`` through
``tf.image.resize(..., method="bilinear", antialias=False)`` with a truncating cast to uint8. That
is a third set of pixels again, and it is not what the published PyTorch checkpoint is paired with.
See ``PROVENANCE.md``.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF

__all__ = ["preprocess"]

#: Mean and standard deviation, already multiplied by 255 the way the reference fuses them.
#: The published ``preprocessor_config.json`` says 0.5 for both; ``rescale_factor`` is 1/255.
_MEAN = [127.5, 127.5, 127.5]
_STD = [127.5, 127.5, 127.5]


def preprocess(image: np.ndarray, resolution: int) -> torch.Tensor:
    """Turn one ``HxWx3`` RGB ``uint8`` array into the tensor the image tower expects.

    Args:
        image: An ``HxWx3`` RGB ``uint8`` array. mozo's ``load_image`` guarantees this shape and
            this colour order.
        resolution: The variant's square side.

    Returns:
        ``(3, resolution, resolution)`` float32, roughly in ``[-1, 1]``. No batch dimension --
        :class:`~mozo.vendors.siglip2_deploy.predictor.Encoder` stacks.

    Note:
        The reference would not convert a non-RGB input: ``do_convert_rgb`` is ``null`` in all
        fifteen of Google's published configs and it is read as false. mozo converts upstream of
        here, so the two agree on every input mozo can produce, and this function requires what
        it is given to already be RGB.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {image.dtype}")

    pixels = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    resized = tvF.resize(
        pixels, [resolution, resolution],
        interpolation=tvF.InterpolationMode.BILINEAR, antialias=True,
    )
    return tvF.normalize(resized.float(), _MEAN, _STD)
