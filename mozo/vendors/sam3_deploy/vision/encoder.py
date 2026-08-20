# SPDX-License-Identifier: Apache-2.0
"""The vision half of SAM 3: trunk then dual neck, wrapped as one module.

This is the expensive part and the only part that depends solely on the image, so it is the unit
that gets cached and the unit that exports as its own graph. Everything downstream -- the concept
head, the click head -- consumes what :meth:`VisionEncoder.forward` returns and nothing else.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import SPEC, Spec
from .neck import Neck
from .vit import Trunk

__all__ = ["VisionEncoder"]


class VisionEncoder(nn.Module):
    """Image in, feature pyramids out.

    Args:
        spec: Vision geometry. Defaults to the published model.
    """

    def __init__(self, spec: Spec = SPEC):
        super().__init__()
        self.trunk = Trunk(spec.trunk)
        self.neck = Neck(spec)

    @torch.no_grad()
    def forward(
        self, batch: Tensor, stacks: tuple[str, ...] = Neck.STACKS
    ) -> dict[str, list[Tensor] | Tensor]:
        """Encode a preprocessed batch.

        Args:
            batch: ``(B, 3, image_size, image_size)``, normalised to [-1, 1].
            stacks: Which neck stacks to build -- see :meth:`~..vision.neck.Neck.forward`. A
                caller that will read only one should ask for only one.

        Returns:
            The requested pyramids, coarsest last, and ``positions``, the coarsest level's
            encoding. A pyramid is the whole of what a per-image cache needs to hold;
            ``positions`` depends only on the fixed input size, so it is memoised by shape
            inside :class:`~..position.SinePositionEmbedding` and shared across every image.
        """
        return self.neck(self.trunk(batch), stacks)
