"""Adapters: one per model family, each turning a published artifact into a predictor.

Nothing is imported here on purpose. :class:`mozo.manager.ModelManager` imports an adapter by
the module path recorded in ``MODEL_REGISTRY``, so a missing optional dependency -- onnxruntime,
coremltools -- fails when that family is asked for rather than when mozo is imported.

Import one directly when you do not want the cache:

    from mozo.adapters.rfdetr import RFDETRPredictor
"""
