# SPDX-License-Identifier: Apache-2.0
"""The public detector: load a checkpoint, run an image, get boxes in original image coordinates."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .image import letterbox, to_original
from .network import build_detector, check_imgsz


def detect(
    image: np.ndarray,
    forward: Callable[[torch.Tensor], torch.Tensor],
    imgsz: int,
    conf: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect in one image, with *forward* supplying the middle step.

    Everything except the forward pass: letterbox the image, run whatever executes the graph, keep
    what clears the threshold, map it back to source pixels. A torch module and an ONNX session
    differ only in *forward*, so both go through one pre- and post-processing path and cannot drift
    apart -- which is the property mozo publishes two artifacts on.

    Shorter than its siblings' by exactly one step, and that is the architecture rather than an
    omission. The head fires once per object, so the network returns a ranked detection list and
    there is no suppression to do; ``iou`` and ``max_det`` are not parameters here because there is
    nothing to overlap and the list length is fixed by the graph.

    Args:
        image: ``HxWx3`` RGB ``uint8`` array.
        forward: Takes the ``(1, 3, imgsz, imgsz)`` batch, returns ``(1, max_det, 6)`` rows of
            ``x1, y1, x2, y2, score, class`` in the letterboxed image's coordinates.
        imgsz: Square side to run at.
        conf: Minimum score to keep.

    Returns:
        Boxes in the source image's pixels, their scores, and their class ids.
    """
    batch, gain, pad_x, pad_y = letterbox(image, imgsz)
    # Brought to the CPU before the threshold rather than after, so the answer does not depend on
    # where the forward pass ran -- which is what lets a graph runtime and a torch module on any
    # device be compared for exact equality.
    rows = forward(batch)[0].float().cpu()
    kept = rows[rows[:, 4] > conf]
    boxes = to_original(kept[:, :4], gain, pad_x, pad_y, image.shape[:2])
    return boxes, kept[:, 4], kept[:, 5].to(torch.int64)


@dataclass
class Detections:
    """Detections for one image, in that image's own pixel coordinates."""

    boxes: np.ndarray  # (n, 4) x1, y1, x2, y2
    scores: np.ndarray  # (n,)
    class_ids: np.ndarray  # (n,) int
    names: list[str]  # (n,) the name of each detected class

    def __len__(self) -> int:
        return len(self.scores)


class Detector:
    """Detection inference for a checkpoint, built entirely from what that checkpoint records.

    Args:
        checkpoint: Path to a ``.pt`` file recording a detection model.
        imgsz: Square side the network is run at. Must be a positive multiple of the coarsest
            stride the checkpoint records. Defaults to the size the checkpoint itself names.
        device: Where to run. mozo decides this; the default is only for direct use.
        fuse_norm: Fold each batch norm into the convolution before it.

    Attributes:
        names: ``{class id: name}`` as recorded in the checkpoint. mozo never invents these.
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike,
        imgsz: int | None = None,
        device: str | torch.device = "cpu",
        fuse_norm: bool = True,
    ):
        self.network = build_detector(os.fspath(checkpoint), fuse=fuse_norm)
        self.network.eval().to(device)
        self.device = device
        self.names = self.network.names
        self.imgsz = (self.network.imgsz if imgsz is None
                      else check_imgsz(int(imgsz), self.network.strides))

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Run one letterboxed batch. This is what :func:`detect` plugs a runtime into."""
        return self.network(batch.to(self.device))

    def predict(self, image: np.ndarray, conf: float = 0.25) -> Detections:
        """Detect objects in one image, given as an ``HxWx3`` RGB ``uint8`` array."""
        boxes, scores, class_ids = detect(image, self.forward, self.imgsz, conf)
        ids = class_ids.numpy()
        return Detections(boxes.numpy(), scores.numpy(), ids, [self.names[int(i)] for i in ids])
