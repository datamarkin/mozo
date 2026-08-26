# SPDX-License-Identifier: Apache-2.0
"""Pixels in, tensors out, and the composite that puts the answer back.

Small, and every line of it is a place a plausible-looking choice changes the picture.

**The image lives in ``[-1, 1]`` and the hole is filled with zero**, which is mid-grey, not black.
Zeroing in ``[0, 1]`` would be black, would look equally reasonable in a debugger, and would hand
the model a conditioning signal it was never trained on.

**The mask is binarised twice**, once on the way in and once after any resize, because a resize
with a smooth filter turns a hard edge into a ramp and the model's mask channel is meant to be
``{0, 1}``.

**The composite is the contract, not a convenience.** Upstream defaults to returning the decoder's
reconstruction of the *whole* frame -- every pixel changed, including the ones nobody selected.
mozo returns the caller's own array byte for byte everywhere the feathered mask does not reach, and
takes only the hole from the model. The feather is upstream's -- a 3-pixel Gaussian -- and it is
worth being exact about what it costs: the blend spreads about 8 px *past* the selection, so
"outside the mask" and "untouched" are the same region only at ``feather=0``.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFilter

__all__ = ["MASK_FEATHER", "as_tensor", "binarise", "composite", "dilate", "to_pixels"]

#: Radius of the Gaussian applied to the mask before compositing. Upstream hardcodes 3.
MASK_FEATHER = 3


def binarise(mask: np.ndarray) -> np.ndarray:
    """``(H, W)`` of anything to ``(H, W)`` uint8 of ``{0, 1}``, splitting at half.

    Accepts ``bool``, ``uint8`` in ``0..255``, ``uint8`` in ``{0, 1}``, or floats in ``0..1``. The
    midpoint is chosen from the dtype **and the values**, not the dtype alone: a float mask
    thresholded at 127 is empty, and so is a ``{0, 1}`` uint8 one, and neither raises.
    """
    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)
    if not mask.size:
        return mask.astype(np.uint8)

    # By content, not by dtype alone. A uint8 mask holding {0, 1} is exactly what a segmenter's
    # boolean masks become, and thresholding *that* at 127.5 empties it -- silently, and then the
    # empty-mask shortcut returns the original image and reports success. Found by running it.
    ceiling = float(mask.max())
    midpoint = 127.5 if (mask.dtype == np.uint8 and ceiling > 1.0) else 0.5
    return (mask > midpoint).astype(np.uint8)


def dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Grow the mask by *pixels* in every direction. ``0`` returns it untouched.

    A removal that stops exactly at the object's edge leaves its shadow and its antialiased rim
    behind, so callers usually want a little more than they selected. Upstream defaults this to
    zero; mozo keeps that default so the extraction is comparable, and exposes it.
    """
    if pixels <= 0:
        return mask
    import cv2

    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def as_tensor(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, ...]:
    """``(image, mask, masked_image)`` as the denoiser wants them.

    *image* is ``(H, W, 3)`` uint8 RGB and *mask* is ``(H, W)`` binary. Returns the image in
    ``[-1, 1]`` as ``(1, 3, H, W)``, the mask as ``(1, 1, H, W)``, and the image with the masked
    region set to zero -- which in ``[-1, 1]`` is mid-grey.
    """
    # ``dtype=`` forces a writable copy. ``from_numpy`` on a read-only array -- which is what
    # ``np.asarray(Image.open(...))`` hands back -- warns, and a library has no business printing
    # a warning about an array the caller never asked it to write to.
    pixels = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    pixels = pixels.div(255.0).mul(2.0).sub(1.0).permute(2, 0, 1).unsqueeze(0)
    binary = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None]
    return pixels, binary, pixels * (1.0 - binary)


def to_pixels(decoded: torch.Tensor) -> np.ndarray:
    """The decoder's ``[-1, 1]`` output as ``(H, W, 3)`` uint8.

    Shifted to ``[0, 1]`` *then* clamped, in that order -- clamping first would fold the tails in
    at the wrong place and shift every value that survived.
    """
    scaled = (decoded[0] + 1.0) / 2.0
    scaled = scaled.permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0)
    return scaled.round().to(torch.uint8).cpu().numpy()


def composite(original: np.ndarray, generated: np.ndarray, mask: np.ndarray,
              feather: int = MASK_FEATHER) -> np.ndarray:
    """Take the hole from *generated* and everything else from *original*.

    The mask is blurred first so the seam is a gradient rather than a staircase. Where the blurred
    mask is exactly zero the result is *original*'s own bytes, unchanged -- which is the property
    that makes this safe to call a photo editor.
    """
    if generated.shape != original.shape:
        raise ValueError(
            f"generated {generated.shape} does not match original {original.shape}; the composite "
            "has to happen at the caller's resolution")

    soft = Image.fromarray((mask * 255).astype(np.uint8), "L")
    if feather > 0:
        soft = soft.filter(ImageFilter.GaussianBlur(radius=feather))
    alpha = (np.asarray(soft, dtype=np.float32) / 255.0)[:, :, None]

    blended = generated.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)
    out = np.rint(blended).astype(np.uint8)
    # Untouched means untouched: where the feathered mask is zero, hand back the original bytes
    # rather than a value that rounded back to them.
    return np.where(alpha > 0.0, out, original)
