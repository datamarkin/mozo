# SPDX-License-Identifier: Apache-2.0
"""The public detector: load a checkpoint, run an image, get boxes in original image coordinates."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .build import build_network, fuse
from .image import letterbox, suppress, to_original
from .reader import load_checkpoint


def detect(
    image: np.ndarray,
    forward: Callable[[torch.Tensor], torch.Tensor],
    imgsz: int,
    conf: float = 0.25,
    iou: float = 0.7,
    max_det: int = 300,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """Detect in one image, with *forward* supplying the middle step.

    Everything except the forward pass: letterbox the image, run whatever executes the graph,
    suppress the overlaps, map the survivors back to source pixels. A torch module and an ONNX
    session differ only in *forward*, so both go through one pre- and post-processing path and
    cannot drift apart -- which is the property mozo publishes two artifacts on.

    Args:
        image: ``HxWx3`` RGB ``uint8`` array.
        forward: Takes the ``(1, 3, imgsz, imgsz)`` batch, returns the raw head output.
        imgsz: Square side to run at.
        conf: Minimum class score to keep.
        iou: Overlap above which two boxes of one class suppress each other.
        max_det: Most detections to return.

    Returns:
        Boxes in the source image's pixels, their scores, their class ids, and ``None`` where a
        family with a mask branch would return one per detection. This family has no such branch,
        so the fourth value is always ``None`` -- it is present because ``mozo.adapters._yolo``
        serves four families through one call, and a seam that changes width between them would
        have to be sniffed at the call site.
    """
    batch, gain, pad_x, pad_y = letterbox(image, imgsz)
    # Brought to the CPU before suppression rather than after, so the answer does not depend on
    # where the forward pass ran. Costs ~0.06 ms and is what lets a graph runtime and a torch
    # module on any device be compared for exact equality.
    prediction = forward(batch)[0].float().cpu()
    boxes, scores, class_ids = suppress(prediction, conf, iou, max_det)
    return to_original(boxes, gain, pad_x, pad_y, image.shape[:2]), scores, class_ids, None


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
        record = load_checkpoint(os.fspath(checkpoint))
        weights = record["ema"] if record.get("ema") is not None else record["model"]
        self.network = build_network(weights)
        if fuse_norm:
            fuse(self.network)
        self.network.eval().to(device)
        self.device = device
        self.names = self.network.names
        step = int(self.network.stride.max())
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
        # The fourth value is the mask slot every vendor's ``detect`` now carries; this
        # family has no mask branch, so it is always None and nothing reads it.
        boxes, scores, class_ids, _ = detect(
            image, self.forward, self.imgsz, conf, iou, max_det)
        ids = class_ids.numpy()
        return Detections(boxes.numpy(), scores.numpy(), ids, [self.names[int(i)] for i in ids])
