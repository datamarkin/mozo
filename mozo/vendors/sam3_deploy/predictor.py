# SPDX-License-Identifier: Apache-2.0
"""Turning what the model returns into instances a caller can use.

This is deliberately outside :mod:`.grounding`: thresholding, gating by presence and mapping
masks back to source pixels are things a *caller* does with the model's output, not part of the
model. ``sam2_deploy`` draws the same line, keeping this in its predictor rather than its network.

It is the first piece of the ``Segmenter`` that will eventually hold the image and prompt caches.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .grounding.boxes import box_cxcywh_to_xyxy
from .image import to_original

__all__ = ["instances"]


def instances(
    result: dict[str, Tensor], shape: tuple[int, int], threshold: float = 0.5
) -> list[dict[str, Tensor]]:
    """Reduce a forward pass to the instances that survive ``threshold``.

    Args:
        result: What :meth:`~.grounding.concept.ConceptHead.forward` returned.
        shape: The source image's ``(height, width)``.
        threshold: Minimum score.

    Returns:
        One entry per image in the batch, each with ``masks`` ``(N, height, width)`` bool,
        ``boxes`` ``(N, 4)`` in source pixels as xyxy, and ``scores`` ``(N,)``.
    """
    # A query's score is its own confidence gated by whether the concept is in the picture at
    # all. Without the presence term, "cow" on a picture of an office still returns the 200
    # queries' best guesses.
    scores = result["logits"].sigmoid() * result["presence"].sigmoid()
    height, width = shape
    scale = result["boxes"].new_tensor([width, height, width, height])

    found = []
    for image in range(scores.shape[0]):
        keep = scores[image] > threshold
        selected = result["masks"][image : image + 1, keep]
        # A prompt that finds nothing is a normal answer, not an error -- ask a picture of an
        # office for "cow" and every query should fall below the threshold. Resizing would raise
        # on an empty batch, so the empty case is built directly.
        masks = (
            to_original(selected, shape)[0].sigmoid() > 0.5
            if selected.shape[1]
            else torch.zeros(0, height, width, dtype=torch.bool, device=selected.device)
        )
        found.append({
            "masks": masks,
            "boxes": box_cxcywh_to_xyxy(result["boxes"][image][keep]) * scale,
            "scores": scores[image][keep],
        })
    return found
