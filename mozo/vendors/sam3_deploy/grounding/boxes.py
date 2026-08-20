# SPDX-License-Identifier: Apache-2.0
"""Box arithmetic shared across the grounding stage.

Both of these are used by more than one module -- the geometry encoder pools exemplar boxes, and
the decoder both biases attention by its reference boxes and refines them -- so neither belongs
inside either one.

Boxes are ``(cx, cy, w, h)`` normalised to ``[0, 1]`` everywhere in this package, and are only
converted to corners where a consumer needs them that way.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["box_cxcywh_to_xyxy", "inverse_sigmoid"]


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert ``(cx, cy, w, h)`` boxes to ``(x1, y1, x2, y2)``."""
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def inverse_sigmoid(x: Tensor, eps: float = 1e-3) -> Tensor:
    """Map ``[0, 1]`` back to logits, clamped so the boundaries stay finite.

    The decoder refines boxes additively in logit space, so it needs to get back there from a
    sigmoid output without ``log(0)``.
    """
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))
