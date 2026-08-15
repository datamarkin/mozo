"""Detectron2 — placeholder.

The previous implementation wrapped detectron2's `DefaultPredictor` directly,
which made the whole framework a runtime dependency and required a per-platform
source build. It has been removed pending reimplementation on exported
artifacts.

Nothing here loads, and nothing here predicts.
"""


class Detectron2Predictor:
    def __init__(self, variant=None, **kwargs):
        raise NotImplementedError(
            "detectron2 is not implemented. The adapter was removed and is "
            "pending reimplementation."
        )

    def predict(self, image):
        raise NotImplementedError
