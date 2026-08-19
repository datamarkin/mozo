import os
from typing import Union

import numpy as np
import cv2


def load_image(image: Union[str, os.PathLike, bytes, np.ndarray]) -> np.ndarray:
    """Decode *image* into mozo's canonical form: an ``HxWx3`` **RGB** ``uint8`` array.

    This is the one place a file, a byte stream, or an array becomes pixels mozo will act on,
    and it exists because channel order is created at decode and invisible afterwards. A numpy
    array carries no colour metadata, so once bytes have become an ndarray nothing downstream
    can tell RGB from BGR -- it has to be declared here or guessed later, and guessing later is
    silent. Both entry points go through this function for that reason: the Python API when a
    caller passes a path, and the HTTP API when a request body arrives.

    RGB rather than BGR because that is what everything mozo composes with uses -- PixelFlow,
    PIL, torchvision, and RF-DETR's own documented contract. BGR is an OpenCV convention that
    leaks outward through ``cv2.imread``; it was never a decision here, just the default that
    came with the decoder.

    Vendors keep whatever their upstream requires -- RF-DETR wants RGB, Depth Anything V2 wants
    BGR -- and the adapters translate. Neither vendor can define this contract, because they
    disagree; the adapters must not, because then every caller would need to know which model
    it is calling before it could read a file.

    Args:
        image: A file path (``str`` or ``Path``), encoded image bytes, or an ``HxWx3`` RGB
            array (returned as-is -- an array is taken at its word, since there is nothing in
            it to check).

    Returns:
        np.ndarray: ``HxWx3`` RGB.

    Raises:
        FileNotFoundError: If a path does not exist or cannot be decoded.
        ValueError: If bytes cannot be decoded, or the input is of an unsupported type.

    Examples:
        >>> image = load_image('photo.jpg')          # doctest: +SKIP
        >>> image = load_image(request_body)         # doctest: +SKIP
        >>> image = load_image(existing_rgb_array)   # doctest: +SKIP
    """
    if isinstance(image, np.ndarray):
        return image

    if isinstance(image, (str, os.PathLike)):
        loaded = cv2.imread(os.fspath(image))
        if loaded is None:
            raise FileNotFoundError(f"Could not load image from path: '{image}'")
        # In place: this array was decoded a line ago and nothing else holds it, so the
        # conversion costs no allocation.
        return cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB, dst=loaded)

    if isinstance(image, (bytes, bytearray, memoryview)):
        loaded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if loaded is None:
            raise ValueError("Could not decode image from bytes.")
        # In place: this array was decoded a line ago and nothing else holds it, so the
        # conversion costs no allocation.
        return cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB, dst=loaded)

    raise ValueError(
        f"Expected image path (str or Path), encoded bytes, or numpy array, "
        f"got {type(image).__name__}"
    )
