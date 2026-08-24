# SPDX-License-Identifier: Apache-2.0
"""The public detector: load a checkpoint, run an image, get boxes in original image coordinates."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .build import load_network
from .image import letterbox, survivors, to_original
from .mask import assemble


def detect(
    image: np.ndarray,
    forward: Callable[[torch.Tensor], torch.Tensor],
    imgsz: int,
    conf: float = 0.25,
    iou: float = 0.7,
    max_det: int = 300,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Detect in one image, with *forward* supplying the middle step.

    Everything except the forward pass: letterbox the image, run whatever executes the graph,
    suppress the overlaps, map the survivors back to source pixels. A torch module and an ONNX
    session differ only in *forward*, so both go through one pre- and post-processing path and
    cannot drift apart -- which is the property mozo publishes two artifacts on.

    Args:
        image: ``HxWx3`` RGB ``uint8`` array.
        forward: Takes the ``(1, 3, imgsz, imgsz)`` batch and returns the raw head output --
            ``(1, 4 + classes, anchors)`` from a detection head, and from a segmentation head that
            tensor with ``nm`` coefficient rows appended, paired with the ``(1, nm, H/4, W/4)``
            prototypes those are coefficients *of*.
        imgsz: Square side to run at.
        conf: Minimum class score to keep.
        iou: Overlap above which two boxes of one class suppress each other.
        max_det: Most detections to return.

    Returns:
        Boxes in the source image's pixels, their scores, their class ids, and -- from a
        segmentation checkpoint -- one boolean mask per detection at the source image's
        resolution. A detection checkpoint returns ``None`` in that slot rather than an empty
        array, so the two are distinguishable without inspecting a length.
    """
    batch, gain, pad_x, pad_y = letterbox(image, imgsz)
    # Brought to the CPU before suppression rather than after, so the answer does not depend on
    # where the forward pass ran -- which is what lets a graph runtime and a torch module on any
    # device be compared for exact equality.
    answer = forward(batch)
    # Tested for explicitly rather than indexed. A segmentation head answers with a pair, and
    # ``answer[0]`` succeeds on a pair too -- handing back the rows with their batch axis still
    # attached, which is plausible, wrong, and silent.
    prediction, protos = answer if isinstance(answer, tuple) else (answer, None)
    prediction = prediction[0].float().cpu()
    coefficients = None
    if protos is not None:
        protos = protos[0].float().cpu()
        # How many trailing rows are mask coefficients is not a convention to carry: there is one
        # coefficient per prototype, so the prototype stack states it. Split off before the
        # suppression, which is the only step that has to know the difference -- everything below
        # is written once and runs the same for both kinds of head.
        prediction, coefficients = prediction.split(
            (prediction.shape[0] - protos.shape[0], protos.shape[0]))

    boxes, scores, class_ids, anchors = survivors(prediction, conf, iou, max_det)
    boxes = to_original(boxes, gain, pad_x, pad_y, image.shape[:2])
    if coefficients is None:
        return boxes, scores, class_ids, None

    masks = assemble(protos, coefficients.T[anchors], boxes, image.shape[:2])
    # Upstream drops any detection whose mask came out empty, and so does this. A box with nothing
    # under it is one the mask branch did not agree with, and keeping it would put a ``None`` where
    # every other row has a mask.
    alive = masks.amax((-2, -1)) > 0
    if not bool(alive.all()):
        boxes, scores, class_ids, masks = boxes[alive], scores[alive], class_ids[alive], masks[alive]
    return boxes, scores, class_ids, masks


@dataclass
class Detections:
    """Detections for one image, in that image's own pixel coordinates."""

    boxes: np.ndarray  # (n, 4) x1, y1, x2, y2
    scores: np.ndarray  # (n,)
    class_ids: np.ndarray  # (n,) int
    names: list[str]  # (n,) the name of each detected class
    masks: np.ndarray | None = None  # (n, h, w) bool, or None from a detection checkpoint

    def __len__(self) -> int:
        return len(self.scores)


class Detector:
    """Inference for a checkpoint, built entirely from what that checkpoint records.

    Args:
        checkpoint: Path to a ``.pt`` file recording a detection or segmentation model. Which of
            the two it is decides whether :attr:`Detections.masks` is filled in; nothing here has
            to be told.
        imgsz: Square side the network is run at. Must be a positive multiple of the coarsest
            stride the checkpoint records.
        device: Where to run. mozo decides this; the default is only for direct use.
        fuse_norm: Fold each batch norm into the convolution before it.

    Attributes:
        names: ``{class id: name}`` as recorded in the checkpoint. mozo never invents these.
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike,
        imgsz: int = 640,
        device: str | torch.device = "cpu",
        fuse_norm: bool = True,
    ):
        self.network = load_network(os.fspath(checkpoint), fuse=fuse_norm)
        self.network.to(device)
        self.device = device
        self.names = self.network.names
        step = int(max(self.network.strides))
        if imgsz <= 0 or imgsz % step:
            raise ValueError(f"imgsz={imgsz} must be a positive multiple of the coarsest stride {step}")
        self.imgsz = imgsz

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Run one letterboxed batch. This is what :func:`detect` plugs a runtime into."""
        return self.network(batch.to(self.device))

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
    ) -> Detections:
        """Detect objects in one image, given as an ``HxWx3`` RGB ``uint8`` array."""
        boxes, scores, class_ids, masks = detect(
            image, self.forward, self.imgsz, conf, iou, max_det)
        ids = class_ids.numpy()
        return Detections(boxes.numpy(), scores.numpy(), ids, [self.names[int(i)] for i in ids],
                          None if masks is None else masks.numpy())
