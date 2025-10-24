"""
Mozo Model Adapters

This module contains adapter classes for different ML frameworks.
Each adapter provides a unified interface for model loading and inference.

Note: Adapters are imported lazily. Import errors are deferred until
      the adapter is actually used, allowing mozo to work even if some
      dependencies are not installed.
"""

# Note: We don't pre-import adapters here to avoid import errors
# when optional dependencies (detectron2, paddleocr, etc.) are missing.
# The ModelFactory will import adapters dynamically when needed.

__all__ = [
    'Detectron2Predictor',
    'DepthAnythingPredictor',
    'Qwen2_5VLPredictor',
    'Qwen3VLPredictor',
    'PaddleOCRPredictor',
    'PPStructurePredictor',
    'EasyOCRPredictor',
    'StabilityInpaintingPredictor',
    'Florence2Predictor',
    'BlipVqaPredictor',
    'DatamarkinPredictor',
]
