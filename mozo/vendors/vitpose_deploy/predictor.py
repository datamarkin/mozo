# SPDX-License-Identifier: Apache-2.0
"""The deployable surface: a frame and some person boxes in, joints out.

**This model does not detect.** It is top-down: it is told where a person is and answers where
their joints are. So :meth:`Predictor.predict` takes boxes, and where those boxes came from is not
its business -- a detector, a tracker, or a hand-drawn rectangle all work the same way. Nothing
here filters them either. Hand it the box of a car and it will return seventeen confident joints on
a car; that is the caller's instruction being carried out, not a failure to notice.

**The expert is fixed at 0.** ViTPose++'s blocks select one of six dataset experts, and upstream
exposes ``dataset_index`` for it. Every published checkpoint's head is COCO's 17 keypoints, and
only expert 0 was trained against that head, so the other five are not alternatives -- they are
ways to get a wrong answer. An argument that can only be set incorrectly is not offered.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import torch

from .config import get_spec
from .image import preprocess
from .network import VitPose
from .postprocess import to_keypoints

__all__ = ["EXPERT", "Predictor"]

#: Which mixture-of-experts branch to run. COCO's, which is the one the published heads match.
EXPERT = 0


class Predictor:
    """Pose estimation for one ViTPose++ checkpoint.

    Args:
        checkpoint: Path to the published weights, which mozo republishes as ``torch-fp32.pth``.
            ``None`` builds no model, which is only useful with *forward*.
        variant: Which published geometry. See :data:`~.config.SPECS`.
        device: Where to run. mozo decides this; the default is only for direct use.
        forward: Run the graph some other way -- takes an ``(N, 3, H, W)`` tensor and returns
            ``(N, K, H, W)``. This is the seam an exported artifact plugs into: everything either
            side of the forward pass is the vendor's, so an ONNX graph cannot drift from the torch
            model by having its preprocessing or its decode reimplemented around it.

    Attributes:
        spec: The geometry in use.
        device: Where it runs.

    Examples:
        >>> model = Predictor("torch-fp32.pth", "base")          # doctest: +SKIP
        >>> joints = model.predict(frame, [[10, 20, 110, 300]])  # doctest: +SKIP
        >>> joints.shape                                         # doctest: +SKIP
        (1, 17, 3)
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike | None = None,
        variant: str = "base",
        device: str | torch.device = "cpu",
        *,
        forward: Callable[[torch.Tensor], np.ndarray] | None = None,
    ):
        if checkpoint is None and forward is None:
            raise ValueError("give a checkpoint to load, or a forward to run instead of one")

        self.spec = get_spec(variant)
        self.device = device
        self._forward = forward
        self.model = None
        if checkpoint is not None:
            # Built on the meta device: every parameter is about to be overwritten by the
            # checkpoint, so allocating and randomly initialising them first is work whose result
            # is discarded on the next line. ``assign=True`` adopts the loaded tensors rather than
            # copying into empty ones. Worth roughly 0.1 s on ``small`` and 2 s on ``huge``, per
            # load -- which the server pays every time it brings a variant into memory.
            with torch.device("meta"):
                self.model = VitPose(variant)
            state = torch.load(os.fspath(checkpoint), map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True, assign=True)
            self.model.eval().to(device)

    def predict(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Find the joints of every person named by *boxes*.

        One forward pass for the whole batch: N boxes become N crops that go through the graph
        together, which is why passing all of a frame's people at once costs far less than a call
        each.

        Args:
            image: ``HxWx3`` RGB ``uint8``, the **whole frame** -- not a crop. The model's own
                cropping reaches outside each box, so a tight crop has already thrown away pixels
                it wants. See :mod:`~.image`.
            boxes: ``(N, 4)`` xyxy in the frame's own pixels.

        Returns:
            ``(N, 17, 3)`` float32 as ``(x, y, confidence)``, in the frame's own pixels, in the
            order the boxes were given. Empty boxes give an empty ``(0, 17, 3)``: a frame with
            nobody in it is an answer, not an error.
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, self.spec.keypoints, 3), dtype=np.float32)

        batch, centers, scales = preprocess(image, boxes, self.spec.height, self.spec.width)
        heatmaps = self.heatmaps(batch)
        return to_keypoints(heatmaps, centers, scales)

    def heatmaps(self, batch: torch.Tensor) -> np.ndarray:
        """Run the graph over a preprocessed batch, returning ``(N, K, H, W)`` as numpy.

        Split out from :meth:`predict` so a runtime other than torch can be swapped in without
        reimplementing either half around it, and so parity can be measured on the raw output
        rather than through the postprocessing.
        """
        if self._forward is not None:
            return np.asarray(self._forward(batch), dtype=np.float32)
        with torch.inference_mode():
            heatmaps = self.model(batch.to(self.device), EXPERT)
        return heatmaps.float().cpu().numpy()
