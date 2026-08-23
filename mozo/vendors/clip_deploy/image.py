# SPDX-License-Identifier: Apache-2.0
"""CLIP's preprocessing: resize the short side, centre crop, normalise.

Three details here are each enough to move every embedding, and none of them raises.

**Bicubic, through PIL.** Upstream resizes a ``PIL.Image`` with ``InterpolationMode.BICUBIC``.
PIL's filter has support that scales with the downsampling factor; neither ``cv2.resize`` nor
``F.interpolate`` reproduces it, with or without ``antialias``. mozo's contract hands vendors a
numpy array, so the conversion to PIL and back is mandatory rather than stylistic -- the same fact
:mod:`mozo.vendors.grounding_dino_deploy.image` records for its own resize.

**The constants are CLIP's, not ImageNet's.** They are close enough to look like a typo when they
differ and far enough apart to shift a similarity.

**Resize takes a single int.** ``Resize(224)`` scales the *short* side to 224 and keeps the aspect
ratio; the centre crop then takes a square out of the middle. Passing ``(224, 224)`` instead would
squash the image to a square and is a different picture.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as TF

__all__ = ["CLIP_MEAN", "CLIP_STD", "preprocess"]

#: CLIP's own channel statistics, in RGB order. Not ImageNet's
#: ``(0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)``.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess(image: np.ndarray, resolution: int) -> Tensor:
    """Turn an ``HxWx3`` RGB ``uint8`` array into the ``(3, n, n)`` tensor the tower wants.

    Args:
        image: RGB ``uint8``, mozo's contract.
        resolution: The variant's square side, from :class:`~mozo.vendors.clip_deploy.config.Spec`.
            Passed rather than defaulted, so a variant published at another resolution cannot
            silently be preprocessed at 224.

    Returns:
        Normalised float32, no batch dimension.
    """
    resized = TF.resize(
        Image.fromarray(image), resolution, interpolation=TF.InterpolationMode.BICUBIC
    )
    cropped = TF.center_crop(resized, resolution)
    return TF.normalize(TF.to_tensor(cropped), mean=list(CLIP_MEAN), std=list(CLIP_STD))
