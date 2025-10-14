"""
Mozo - Universal Computer Vision Model Server

25+ pre-configured models ready to use. No deployment, no configuration.
Just `mozo start` and make HTTP requests.

Quick Start:
    >>> # From terminal:
    >>> mozo start
    >>>
    >>> # Then use any model via HTTP:
    >>> curl -X POST "http://localhost:8000/predict/detectron2/mask_rcnn_R_50_FPN_3x" \\
    >>>   -F "file=@image.jpg"

Advanced Usage (Python SDK):
    >>> from mozo import ModelManager
    >>> import cv2
    >>>
    >>> manager = ModelManager()
    >>> model = manager.get_model('detectron2', 'mask_rcnn_R_50_FPN_3x')
    >>> image = cv2.imread('example.jpg')
    >>> detections = model.predict(image)  # Returns PixelFlow Detections
    >>> print(f"Found {len(detections)} objects")

Features:
    - 25+ models from Detectron2, HuggingFace Transformers
    - Zero deployment - no Docker, Kubernetes, or cloud needed
    - Automatic memory management with lazy loading
    - PixelFlow integration for unified detection format
    - Thread-safe concurrent access

For more information, see:
    - Documentation: https://github.com/datamarkin/mozo
"""

__version__ = "0.2.0"

# Public API exports
from mozo.manager import ModelManager
from mozo.registry import (
    MODEL_REGISTRY,
    get_available_families,
    get_available_variants,
    get_model_info,
)

__all__ = [
    "ModelManager",
    "MODEL_REGISTRY",
    "get_available_families",
    "get_available_variants",
    "get_model_info",
    "__version__",
]