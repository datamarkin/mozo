import time
import uuid
from typing import Union

import numpy as np
import cv2


def load_image(image: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load an image from path or return as-is if already a numpy array.

    This utility enables predict() methods to accept either file paths
    or numpy arrays, providing a more flexible API.

    Args:
        image: Either a file path (str) or numpy array (BGR format)

    Returns:
        np.ndarray: Image in BGR format (OpenCV standard)

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If image cannot be loaded or invalid type

    Examples:
        >>> # From path
        >>> img = load_image('photo.jpg')
        >>>
        >>> # Already numpy array (passthrough)
        >>> img = load_image(existing_array)
    """
    if isinstance(image, np.ndarray):
        return image

    if isinstance(image, str):
        loaded = cv2.imread(image)
        if loaded is None:
            raise FileNotFoundError(f"Could not load image from path: '{image}'")
        return loaded

    raise ValueError(
        f"Expected image path (str) or numpy array, got {type(image).__name__}"
    )


def create_openai_response(model_name, text_content):
    """
    Formats a text response into an OpenAI Chat Completion compatible dictionary.
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text_content,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,  # Dummy value, as we are not tracking tokens
            "completion_tokens": 0, # Dummy value
            "total_tokens": 0, # Dummy value
        },
    }
