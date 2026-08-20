# SPDX-License-Identifier: Apache-2.0
"""The feed-forward block shared by the two transformer towers.

The ViT trunk and the CLIP text tower use the same shape -- widen, GELU, narrow -- and the same
``fc1``/``fc2`` key names. It lives at the package root rather than under ``vision/`` or ``text/``
because both need it and neither owns it, the same reason :mod:`.position` sits here.

This is *not* the same block as :class:`.grounding.layers.Mlp`, which the grounding stage uses:
that one activates with ReLU and stores its layers under ``layers.N``. Two activations and two
key layouts, so two classes.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["FeedForward"]


class FeedForward(nn.Module):
    """Widen, GELU, narrow.

    Args:
        width: Input and output width.
        intermediate: The widened width.
    """

    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.fc1 = nn.Linear(width, intermediate)
        self.fc2 = nn.Linear(intermediate, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))
