"""mozo's image contract: RGB, ``uint8``, ``HxWx3``.

One function, because there is one decision to make and it must be made once. Channel order is
created when bytes are decoded and invisible afterwards, so a second decoder somewhere else is
not a duplicate -- it is a second answer, and the wrong one fails silently. A channel swap cost
Depth Anything V2 0.166 m of mean error and 1.84 m at worst, with nothing raised.

**The decoder is PixelFlow's.** That is the same argument aimed one layer down rather than a
retreat from it: ``pf.read_image`` decodes for every consumer of the library whose stated contract
is that images are RGB, so mozo having its own would be the second answer this module exists to
prevent -- one that agreed today and could stop agreeing after any change to either side. What
stays here is the contract itself, and the name every adapter and both entry points call.

    >>> from mozo.image import load_image
    >>> load_image("photo.jpg").shape          # doctest: +SKIP
    (1281, 1920, 3)
"""

import os
from typing import Union

import numpy as np
import pixelflow as pf


def load_image(image: Union[str, os.PathLike, bytes, np.ndarray]) -> np.ndarray:
    """Decode *image* into mozo's canonical form: an ``HxWx3`` **RGB** ``uint8`` array.

    This is the one place a file, a byte stream, or an array becomes pixels mozo will act on.
    Both entry points go through it for that reason: the Python API when a caller passes a path,
    and the HTTP API when a request body arrives as bytes.

    RGB rather than BGR because that is what everything mozo composes with uses -- PixelFlow, PIL,
    torchvision, and RF-DETR's own documented contract. BGR is an OpenCV convention that leaks
    outward through ``cv2.imread``; it was never a decision here, just the default that came with
    the decoder.

    Vendors keep whatever their upstream requires -- RF-DETR wants RGB, Depth Anything V2 wants
    BGR -- and the adapters translate. Neither vendor can define this contract, because they
    disagree; the adapters must not, because then every caller would need to know which model it
    is calling before it could read a file.

    **A path and encoded bytes are decoded; an array is trusted.** For the first two this function
    knows the answer is RGB, because it watched the bytes become pixels. An array arrives already
    decoded and carries no colour metadata, so nothing in it can distinguish RGB from BGR -- that
    one input is an assertion by the caller, not a guarantee by this function. Everything about it
    that *can* be checked is: shape, dimensions and dtype are refused when wrong, so a grayscale,
    RGBA or float array stops here instead of reaching a model. Only the channel order is
    untestable, and it is untestable everywhere. If your array came from ``cv2.imread`` it is BGR;
    pass the path instead and let this decode it, which is what it is for.

    Args:
        image: A file path (``str`` or ``Path``), encoded image bytes, or an ``HxWx3`` RGB
            ``uint8`` array.

    Returns:
        np.ndarray: ``HxWx3`` RGB.

    Raises:
        FileNotFoundError: If a path does not exist.
        ValueError: If bytes cannot be decoded, or an array is not ``HxWx3`` ``uint8``.
        TypeError: If the input is not one of the three accepted kinds.

    Examples:
        >>> image = load_image('photo.jpg')          # doctest: +SKIP
        >>> image = load_image(request_body)         # doctest: +SKIP
        >>> image = load_image(existing_rgb_array)   # doctest: +SKIP
    """
    return pf.read_image(image)
