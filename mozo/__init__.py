"""
Mozo - Universal Computer Vision Model Serving Library

Mozo provides a unified interface for deploying and managing computer vision models
from multiple ML frameworks (Detectron2, HuggingFace Transformers) with intelligent
lifecycle management, lazy loading, and automatic memory cleanup.

Key Features:
    - Multi-framework support (Detectron2, Transformers, custom adapters)
    - Lazy loading with thread-safe concurrent access
    - Automatic cleanup of inactive models
    - REST API server with FastAPI
    - Registry-Factory-Manager pattern for clean architecture

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