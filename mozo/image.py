"""mozo's image contract: RGB, ``uint8``, ``HxWx3``.

One decision, made once, and it is made at the codec boundary in both directions. Channel order is
created when bytes are decoded and destroyed when they are encoded, and invisible in between -- so
a second decoder or a second encoder somewhere else is not a duplicate, it is a second answer, and
the wrong one fails silently. A channel swap cost Depth Anything V2 0.166 m of mean error and
1.84 m at worst, with nothing raised.

Two functions rather than one because bytes cross the boundary both ways, not because there are two
decisions. :func:`load_image` is what a file or a request body becomes; :func:`encode_image` is what
goes back out. The encoder was private inside the workflow's HTTP layer until the model endpoints
started borrowing it by its private name, which is how a shared fact announces it is homed wrong.

**The encoder is PixelFlow's for three channels and mozo's for four.** A cut-out is an image and
leaves by this door like any other, but ``pf.encode_image`` refuses a fourth channel, so the RGBA
path is written out below. That is a widening of one function rather than a second encoder
elsewhere, which is the thing this module exists to prevent -- and it is safe for a reason specific
to alpha: the ambiguity a hand-written encoder forgets is RGB-versus-BGR, and there is exactly one
channel order for RGBA.

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

import cv2
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

    Note:
        An ``HxWx4`` array is refused here while the workflow quietly narrows one at a node
        boundary (:func:`as_rgb`). Not two answers to one question: a graph edge is machinery
        deciding what a node can be handed, and a hand-written call is the caller asserting what
        they have. Passing a cut-out to something that wants a photograph should say so.

    Examples:
        >>> image = load_image('photo.jpg')          # doctest: +SKIP
        >>> image = load_image(request_body)         # doctest: +SKIP
        >>> image = load_image(existing_rgb_array)   # doctest: +SKIP
    """
    return pf.read_image(image)


def encode_image(image: np.ndarray, extension: str = ".png") -> bytes:
    """Encode an RGB array as bytes -- the way back out, and the mirror of :func:`load_image`.

    PixelFlow's, for the reason the decoder is: ``cv2.imencode`` alone is only half the operation,
    and the half it leaves out is the RGB-to-BGR conversion that goes with it. Written by hand that
    step is remembered in most places and forgotten in one, which is exactly the failure this module
    exists to make unconstructible -- and forgetting it here produces a picture that looks plausible
    and has its channels swapped.

    **Four channels are accepted here and nowhere else in this module.** A cut-out is an image --
    every viewer on the receiving end already knows what its fourth channel means -- so it leaves
    through the same door rather than through a second encoder. That does not weaken the contract
    :func:`load_image` states, and the asymmetry is not an oversight: the reason this module
    exists is that a three-channel array carries no record of whether it is RGB or BGR, and
    *that* ambiguity is what a hand-written encoder forgets. Alpha has no such ambiguity. There is
    one channel order for RGBA and one for BGRA, and the swap between them is this function's job
    in both cases.

    PNG is what everything in mozo defaults to and the only format a cut-out reaches here through;
    :data:`ALPHA_FORMATS` is what is *accepted*. Asking for JPEG with an alpha channel present is
    refused rather than flattened, because flattening picks a background colour on the caller's
    behalf and that is a decision, not a conversion.

    Args:
        image: ``HxWx3`` RGB ``uint8`` as :func:`load_image` returns, or ``HxWx4`` RGBA where the
            fourth channel is opacity.
        extension: The format, with or without the dot. PNG by default because an annotated image
            is mostly thin lines and mask edges, which is what JPEG is worst at, and a result that
            has been quietly smeared is worse than a larger response. A caller that wants a cheap
            preview rather than a result asks for ``".jpg"`` and says so.

    Returns:
        bytes: The encoded image.

    Raises:
        ValueError: If *image* is not ``HxWx3`` or ``HxWx4`` ``uint8``, if a format that cannot
            carry alpha is asked for with a four-channel image, or if the encode fails.
    """
    if has_alpha(image):
        return _encode_rgba(image, extension)
    return pf.encode_image(image, extension)


#: Formats that can carry an alpha channel. Not a complete list of what OpenCV writes -- a list of
#: what mozo will hand a cut-out to. PNG is what everything here defaults to; the rest are accepted
#: because a caller who asks for them has not asked for anything lossy or lossless-but-flat.
ALPHA_FORMATS = frozenset({".png", ".webp", ".tiff", ".tif"})


def has_alpha(image: object) -> bool:
    """Is *image* a cut-out -- an ``HxWx4`` array with opacity in the fourth channel?

    One definition, because this module owns what mozo's canonical image is and three callers
    were each deciding it again. A copy that drops the ``isinstance`` guard raises
    ``AttributeError`` where this raises nothing, which is how the same rule comes to have two
    answers.
    """
    return isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 4


def _encode_rgba(image: np.ndarray, extension: str) -> bytes:
    """Encode an ``HxWx4`` RGBA array. The only place in mozo that encodes four channels."""
    if image.dtype != np.uint8:
        raise ValueError(f"expected an RGBA uint8 HxWx4 image, got dtype {image.dtype}")

    suffix = extension if extension.startswith(".") else f".{extension}"
    if suffix.lower() not in ALPHA_FORMATS:
        raise ValueError(
            f"{suffix} cannot carry an alpha channel, and this image has one. Ask for .png, or "
            f"drop the alpha yourself if you meant to composite it.")

    # The RGB-to-BGR swap this module exists to keep in one place, with alpha left where it is.
    # Through cv2 rather than ``image[:, :, [2, 1, 0, 3]]``: the fancy-index gather is bit-identical
    # and 17x slower -- 11.0 ms against 0.63 ms on a 4K frame -- because it reads the source with a
    # stride of four instead of letting OpenCV do a planar shuffle.
    ok, buffer = cv2.imencode(suffix, cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise ValueError(f"could not encode the image as {suffix}")
    return buffer.tobytes()


def as_rgb(image: np.ndarray) -> np.ndarray:
    """Drop an alpha channel if there is one. ``HxWx3`` in, ``HxWx3`` out, unchanged.

    What the workflow applies at a node boundary so that a node declaring ``Image`` is handed
    three channels whatever the wire carried. Kept here rather than in the workflow because it is
    the same decision this module owns everywhere else: what mozo's canonical image is.
    """
    if has_alpha(image):
        # ``np.ascontiguousarray(image[:, :, :3])`` is bit-identical and **75x slower**: 37.3 ms
        # against 0.50 ms on a 4K frame. This runs at every node boundary of every workflow, so
        # the difference is the whole cost of letting one image port carry both channel counts.
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image
