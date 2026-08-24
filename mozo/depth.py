"""How a depth map travels: a 16-bit PNG, and the two numbers that make it a measurement again.

One home, because the recovery formula is a contract with whoever reads the bytes, and a contract
stated in two places is one that will eventually be stated two different ways. Both the ``/predict``
endpoint and the workflow runtime hand depth maps out; they encode them the same because they call
the same function, not because someone kept two copies in step.
"""

from __future__ import annotations

import cv2
import numpy as np


def encode(depth: np.ndarray) -> tuple[bytes, float, float]:
    """Encode a depth map as a 16-bit PNG, with the endpoints needed to read it back.

    An 8-bit PNG is the wrong answer here: six of the nine Depth Anything V2 variants predict
    metres, and metres are the entire point of choosing one. Quantising them to 256 levels and
    calling it an image discards the measurement.

    16-bit is lossless enough to be honest -- over an 80 m range one step is 1.2 mm -- and PNG stays
    viewable in any tool. The values are min-max normalised into the full 16-bit range, so a client
    recovers the original with::

        depth = low + png / 65535 * (high - low)

    Returns:
        The PNG bytes, and the ``(low, high)`` the normalisation used.

    Raises:
        ValueError: If OpenCV could not encode the map.
    """
    low, high = float(depth.min()), float(depth.max())

    # One pass into one buffer. The arithmetic spelling of this allocates a full-size float32
    # temporary per operator -- four of them, ~34 MB of churn on a 1920x1281 map -- for the same
    # rounding, and needs a special case for a flat map that NORM_MINMAX handles itself.
    scaled = cv2.normalize(depth, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

    success, encoded = cv2.imencode(".png", scaled)
    if not success:
        raise ValueError("could not encode the depth map")
    return encoded.tobytes(), low, high
