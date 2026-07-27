"""
Mozo Model Adapters

This module contains adapter classes for different ML frameworks.
Each adapter provides a unified interface for model loading and inference.

Adapters are deliberately NOT imported here: ModelFactory imports them
dynamically from the module path recorded in MODEL_REGISTRY, so a missing
optional dependency (detectron2, paddleocr, rfdetr, ...) only fails when that
family is actually requested rather than at package import time.

Import an adapter directly if you need it without the factory:

    from mozo.adapters.rfdetr import RFDETRPredictor
"""
