# ------------------------------------------------------------------------
# Depth Anything V2
# Copyright (c) 2024 TikTok / The University of Hong Kong. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pre-processing, forward pass, and post-processing for Depth Anything V2.

Upstream keeps these on the model itself, as ``DepthAnythingV2.infer_image`` and
``.image2tensor``. They live here instead for one reason: ``image2tensor`` probes the host for a
device and moves the tensor there itself, which makes the model's own placement unobservable and
a CPU-vs-MPS comparison impossible to set up. Everything else is upstream's, step for step --
the same ``Resize``/``NormalizeImage``/``PrepareForNet`` chain from :mod:`.util.transform`, the
same ``cv2.INTER_CUBIC``, the same bilinear resize back to the input resolution.

That chain preserves the image's aspect ratio: the shorter side becomes ``input_size`` and the
longer one follows, each rounded to a multiple of 14. The tensor reaching the model is therefore
rarely square and differs in shape from image to image, which is exactly what upstream does and
what the published numbers were measured with.
"""

from __future__ import annotations

__all__ = ["Predictor"]

import os
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.transforms import Compose

from .config import ModelSpec, get_spec
from .dpt import DepthAnythingV2
from .util.transform import NormalizeImage, PrepareForNet, Resize

# ImageNet statistics, as upstream applies them.
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class Predictor:
    """One loaded Depth Anything V2 variant, ready to run.

    Attributes:
        spec: The variant's architecture specification.
        model: The underlying :class:`~.dpt.DepthAnythingV2` module, in eval mode.
    """

    def __init__(self, model: DepthAnythingV2, spec: ModelSpec) -> None:
        self.model = model.eval()
        self.spec = spec
        # Built once: every argument is either a constant or ``spec.input_size``, which is frozen,
        # and the transforms keep no per-image state -- ``Resize`` derives the output size from
        # each sample. Upstream rebuilds this per call; that is the one thing not worth copying.
        self._transform = Compose([
            Resize(
                width=spec.input_size,
                height=spec.input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=_MEAN, std=_STD),
            PrepareForNet(),
        ])

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        weights: str | os.PathLike[str],
        device: str | torch.device = "cpu",
    ) -> Predictor:
        """Build variant *name*, load its checkpoint, and return a ready predictor.

        Args:
            name: A released variant, e.g. ``"large"`` or ``"indoor-small"``.
            weights: Path to that variant's checkpoint.
            device: Where to place the model.

        Returns:
            A predictor with every parameter populated from the checkpoint.

        Raises:
            KeyError: If *name* is not a released variant.
            RuntimeError: If the checkpoint does not match the variant's architecture.
        """
        spec = get_spec(name)
        model = DepthAnythingV2(
            encoder=spec.encoder,
            features=spec.features,
            out_channels=list(spec.out_channels),
            max_depth=spec.max_depth,
        )
        state = torch.load(weights, map_location="cpu", weights_only=True)
        # strict=True on purpose: a checkpoint that leaves any parameter at its random
        # initialization is a bug, and one that silently half-loads is a worse one.
        model.load_state_dict(state, strict=True)
        return cls(model.to(device), spec)

    @property
    def device(self) -> torch.device:
        """Where the model's parameters live."""
        return next(self.model.parameters()).device

    def preprocess(self, image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Turn one BGR image into a batched tensor, and report its original size.

        Args:
            image: An ``HxWx3`` BGR array, as ``cv2.imread`` returns.

        Returns:
            The ``1x3xH'xW'`` tensor on this predictor's device, and the original ``(h, w)``.
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        tensor = torch.from_numpy(self._transform({"image": rgb})["image"]).unsqueeze(0)
        return tensor.to(self.device), (h, w)

    def postprocess(self, depth: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
        """Resize a raw depth map back to the original image resolution.

        Args:
            depth: The model's ``1xH'xW'`` output.
            size: The original ``(h, w)``.

        Returns:
            An ``HxW`` float32 array. Metres for the metric variants; unitless inverse depth,
            larger meaning nearer, for the relative ones -- see :attr:`ModelSpec.unit`.
        """
        resized = F.interpolate(depth[:, None], size, mode="bilinear", align_corners=True)
        return resized[0, 0].float().cpu().numpy()

    @torch.inference_mode()
    def predict(self, image: Any) -> np.ndarray:
        """Estimate depth for one image.

        Args:
            image: An ``HxWx3`` BGR array.

        Returns:
            An ``HxW`` float32 depth map at the input's resolution.
        """
        tensor, size = self.preprocess(image)
        return self.postprocess(self.model(tensor), size)
