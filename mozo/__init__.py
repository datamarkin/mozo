"""
Mozo - Universal Computer Vision Model Server

35+ pre-configured models ready to use. No deployment, no configuration.
Just `mozo start` and make HTTP requests.

Quick Start (Server):
    >>> # From terminal:
    >>> mozo start
    >>>
    >>> # Then use any model via HTTP:
    >>> curl -X POST "http://localhost:8000/predict/detectron2/mask_rcnn_R_50_FPN_3x" \\
    >>>   -F "file=@image.jpg"

Quick Start (Python SDK):
    >>> from mozo import get_model
    >>>
    >>> # Load model with simple one-liner
    >>> model = get_model('detectron2/mask_rcnn_R_50_FPN_3x')
    >>>
    >>> # Run prediction - accepts file path or numpy array
    >>> detections = model.predict('image.jpg')
    >>> print(f"Found {len(detections)} objects")

Advanced Usage (with ModelManager):
    >>> from mozo import ModelManager
    >>>
    >>> manager = ModelManager()
    >>> model = manager.get_model('detectron2', 'mask_rcnn_R_50_FPN_3x')
    >>> detections = model.predict('image.jpg')
    >>>
    >>> # Advanced: cleanup inactive models
    >>> manager.cleanup_inactive_models(inactive_seconds=600)

Features:
    - 35+ models from Detectron2, HuggingFace Transformers, PaddleOCR, EasyOCR
    - Zero deployment - no Docker, Kubernetes, or cloud needed
    - Automatic memory management with lazy loading
    - PixelFlow integration for unified detection format
    - Thread-safe concurrent access
    - Path support: pass file paths directly to predict()

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

# Module-level singleton manager for convenience API
_default_manager = None


def get_model(identifier, variant=None, device=None):
    """
    Load a model without explicitly creating a ModelManager.

    This is a convenience function that uses a shared ModelManager instance.
    For advanced use cases (cleanup, unloading, multiple managers), use
    ModelManager directly.

    Args:
        identifier: Either "family/variant" string or just "family"
        variant: Variant name (optional if identifier contains "/")
        device: Compute device - 'cuda', 'mps', 'cpu', or None (auto-detect)
                If None, automatically selects best available device:
                CUDA GPU > Apple MPS > CPU

    Returns:
        Loaded model predictor instance

    Examples:
        >>> from mozo import get_model
        >>>
        >>> # Auto-selects best device (GPU if available)
        >>> model = get_model('detectron2/mask_rcnn_R_50_FPN_3x')
        >>>
        >>> # Force CPU (e.g., for memory reasons)
        >>> model = get_model('detectron2/mask_rcnn_R_50_FPN_3x', device='cpu')
        >>>
        >>> # Force specific GPU
        >>> model = get_model('detectron2/mask_rcnn_R_50_FPN_3x', device='cuda:1')
        >>>
        >>> # Run prediction
        >>> detections = model.predict('image.jpg')
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelManager()

    # Support "family/variant" format
    if variant is None and "/" in identifier:
        family, variant = identifier.split("/", 1)
    else:
        family = identifier

    return _default_manager.get_model(family, variant, device=device)


__all__ = [
    "ModelManager",
    "MODEL_REGISTRY",
    "get_available_families",
    "get_available_variants",
    "get_model_info",
    "get_model",
    "__version__",
]