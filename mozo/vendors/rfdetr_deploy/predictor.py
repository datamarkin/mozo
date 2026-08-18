# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The deployment entry point: build a variant, load its weights, run images through it.

Outputs are the raw per-image dictionaries that :class:`~rfdetr_deploy.models.PostProcess` produces — tensors of
``scores`` / ``labels`` / ``boxes``, plus ``masks`` or keypoint fields for the variants that have them.  Wrapping those
in a richer result type is deliberately left to the caller, so this package stays free of any output-format dependency.
"""

from __future__ import annotations

__all__ = ["Predictor"]

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor

from .config import ModelSpec, get_spec
from .models import PostProcess, build_model
from .utilities.logger import get_logger
from .weights import load_checkpoint, load_state_dict_into

logger = get_logger()

# ImageNet statistics; RF-DETR trains and exports against these.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

ImageInput = "str | os.PathLike[str] | Image.Image | np.ndarray | Tensor"


class Predictor:
    """A loaded RF-DETR model ready for inference.

    Args:
        spec: The variant's architecture spec, already reconciled with the checkpoint.
        model: The built and weight-loaded ``LWDETR``.
        postprocess: The matching post-processor.
        class_names: Class names recorded in the checkpoint, empty when absent.

    Examples:
        >>> predictor = Predictor.from_pretrained("rfdetr-small", weights=path)  # doctest: +SKIP
        >>> results = predictor.predict("image.jpg", threshold=0.5)  # doctest: +SKIP
        >>> sorted(results[0])  # doctest: +SKIP
        ['boxes', 'labels', 'scores']
    """

    def __init__(
        self,
        spec: ModelSpec,
        model: Any,
        postprocess: PostProcess,
        class_names: list[str],
    ) -> None:
        self.spec = spec
        self.model = model
        self.postprocess = postprocess
        self.class_names = class_names

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        *,
        weights: str | os.PathLike[str],
        device: str | torch.device = "cpu",
        resolution: int | None = None,
    ) -> Predictor:
        """Build variant *name*, load its checkpoint, and return a ready predictor.

        The model is built to match the checkpoint, not the other way round: the class count comes from
        ``class_embed.bias`` and the keypoint schema from ``_kp_active_mask``, so a fine-tuned checkpoint loads without
        the caller having to restate its shape.  Loading rejects any missing or unexpected key — see
        :func:`~rfdetr_deploy.weights.load_state_dict_into`.

        Args:
            name: Variant identifier, e.g. ``"rfdetr-small"``.
            weights: Path to the checkpoint to load.
            device: Device to place the model on.
            resolution: Square input side to run at. Defaults to the variant's training resolution; a different value
                triggers positional-embedding interpolation at load time.

        Returns:
            A ready :class:`Predictor` in eval mode.

        Raises:
            ValueError: If *resolution* is not a positive multiple of ``patch_size * num_windows``.

        Examples:
            >>> Predictor.from_pretrained("rfdetr-nano", weights="rf-detr-nano.pth")  # doctest: +SKIP
        """
        spec = get_spec(name)
        if resolution is not None:
            block = spec.patch_size * spec.num_windows
            if resolution <= 0 or resolution % block != 0:
                raise ValueError(
                    f"resolution must be a positive multiple of patch_size * num_windows ({block}); got {resolution}."
                )
            spec = spec.with_overrides(resolution=resolution, positional_encoding_size=resolution // spec.patch_size)

        checkpoint_path = Path(weights).expanduser()
        checkpoint = load_checkpoint(checkpoint_path)
        spec = _reconcile_spec_with_checkpoint(spec, checkpoint["model"])

        model = build_model(spec)
        class_names = load_state_dict_into(model, checkpoint, positional_encoding_size=spec.positional_encoding_size)
        model.eval().to(device)

        postprocess = PostProcess(
            num_select=spec.num_select,
            num_keypoints_per_class=list(spec.num_keypoints_per_class),
            trace_alpha=spec.postprocess_trace_alpha,
        )
        return cls(spec=spec, model=model, postprocess=postprocess, class_names=class_names)

    @property
    def device(self) -> torch.device:
        """Device the model's parameters live on."""
        return next(self.model.parameters()).device

    def preprocess(self, images: list[Any]) -> tuple[Tensor, list[tuple[int, int]]]:
        """Convert *images* into a normalized batch tensor and record their original sizes.

        Accepts file paths, PIL images, HWC ``uint8`` / float arrays, and CHW tensors already scaled to ``[0, 1]``.
        Resizing uses ``antialias=False`` to match the antialias-free bilinear resize RF-DETR trains under; enabling
        antialias here silently costs accuracy rather than raising.

        Args:
            images: Images in any accepted form.

        Returns:
            A ``(batch, original_sizes)`` pair, where ``original_sizes`` holds ``(height, width)`` per image.

        Raises:
            ValueError: If a tensor input is not CHW with the expected channel count, or is outside ``[0, 1]``.

        Examples:
            >>> predictor = Predictor.from_pretrained("rfdetr-nano")  # doctest: +SKIP
            >>> batch, sizes = predictor.preprocess([torch.rand(3, 64, 64)])  # doctest: +SKIP
        """
        side = self.spec.resolution
        tensors: list[Tensor] = []
        sizes: list[tuple[int, int]] = []

        for image in images:
            tensor = _to_chw_float_tensor(image, expected_channels=self.spec.num_channels)
            sizes.append((int(tensor.shape[1]), int(tensor.shape[2])))
            tensors.append(F.resize(tensor, [side, side], antialias=False))

        batch = torch.stack(tensors).to(self.device)
        return F.normalize(batch, list(_MEAN[: self.spec.num_channels]), list(_STD[: self.spec.num_channels])), sizes

    @torch.inference_mode()
    def predict(
        self,
        images: Any,
        *,
        threshold: float = 0.5,
    ) -> list[dict[str, Tensor]]:
        """Run inference and return one post-processed result dict per image.

        Args:
            images: A single image or a list of images, in any form :meth:`preprocess` accepts.
            threshold: Minimum score for a detection to be kept.

        Returns:
            One dict per input image with ``scores``, ``labels``, and ``boxes`` in source-image pixel coordinates, plus
            ``masks`` for segmentation variants and ``keypoints`` / ``keypoint_confidence`` for keypoint variants.

        Examples:
            >>> predictor = Predictor.from_pretrained("rfdetr-small")  # doctest: +SKIP
            >>> predictor.predict("photo.jpg")[0]["boxes"].shape[-1]  # doctest: +SKIP
            4
        """
        batch_input = images if isinstance(images, (list, tuple)) else [images]
        batch, sizes = self.preprocess(list(batch_input))

        outputs = self.model(batch)
        target_sizes = torch.tensor(sizes, device=batch.device)
        results = self.postprocess(outputs, target_sizes=target_sizes, score_threshold=threshold)

        kept: list[dict[str, Tensor]] = []
        for result in results:
            keep = result["scores"] > threshold
            kept.append({key: value[keep] for key, value in result.items()})
        return kept


def _reconcile_spec_with_checkpoint(spec: ModelSpec, state: dict[str, Any]) -> ModelSpec:
    """Return *spec* adjusted so the model it builds matches *state* exactly.

    Only shapes the checkpoint states unambiguously are taken from it: the detection head's class count, and the
    keypoint schema recorded in the ``_kp_active_mask`` buffer.

    Args:
        spec: The variant's published spec.
        state: The checkpoint's model state dict.

    Returns:
        A spec whose ``num_classes`` and ``num_keypoints_per_class`` agree with the checkpoint.

    Examples:
        >>> spec = get_spec("rfdetr-nano")
        >>> _reconcile_spec_with_checkpoint(spec, {"class_embed.bias": torch.zeros(4)}).num_classes
        3
    """
    overrides: dict[str, Any] = {}

    class_bias = state.get("class_embed.bias")
    if isinstance(class_bias, Tensor):
        # The head carries one slot per class plus background.
        checkpoint_classes = int(class_bias.shape[0]) - 1
        if checkpoint_classes != spec.num_classes:
            logger.info("Aligning num_classes %d -> %d from checkpoint.", spec.num_classes, checkpoint_classes)
            overrides["num_classes"] = checkpoint_classes

    kp_mask = state.get("_kp_active_mask")
    if spec.use_grouppose_keypoints and isinstance(kp_mask, Tensor) and kp_mask.ndim == 2:
        schema = [int(count) for count in kp_mask.sum(dim=1).tolist()]
        if any(count > 0 for count in schema) and schema != list(spec.num_keypoints_per_class):
            logger.info("Aligning keypoint schema %s -> %s from checkpoint.", spec.num_keypoints_per_class, schema)
            overrides["num_keypoints_per_class"] = schema

    return spec.with_overrides(**overrides) if overrides else spec


def _to_chw_float_tensor(image: Any, *, expected_channels: int) -> Tensor:
    """Normalize one input image to a CHW float tensor scaled to ``[0, 1]``.

    Args:
        image: A path, PIL image, HWC array, or CHW tensor.
        expected_channels: Channel count the model was built for.

    Returns:
        A CHW float tensor in ``[0, 1]``.

    Raises:
        ValueError: If a tensor input is not 3-D, has the wrong channel count, or falls outside ``[0, 1]``.

    Examples:
        >>> tuple(_to_chw_float_tensor(np.zeros((4, 5, 3), dtype=np.uint8), expected_channels=3).shape)
        (3, 4, 5)
    """
    if isinstance(image, (str, os.PathLike, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        # Any PIL mode (L, LA, RGBA, P, ...) converts to RGB so callers need not pre-convert.
        if image.mode != "RGB":
            image = image.convert("RGB")
        return F.to_tensor(image)
    if isinstance(image, np.ndarray):
        return F.to_tensor(image)
    if isinstance(image, Tensor):
        if image.dim() != 3 or image.shape[0] != expected_channels:
            raise ValueError(
                f"tensor images must be (C, H, W) with C == {expected_channels}; got shape {tuple(image.shape)}."
            )
        if bool((image > 1).any()) or bool((image < 0).any()):
            raise ValueError("tensor images must already be scaled to [0, 1].")
        return image.float()
    raise TypeError(f"unsupported image type {type(image).__name__}")
