# SPDX-License-Identifier: Apache-2.0
"""CLIP's image tower: a Vision Transformer that pools at its class token.

Pixels become patches by a strided convolution, a learned class token is prepended, positions are
added, and the stack runs. The class token's final row -- and only that row -- is normalised and
projected into the shared space. The patch rows are computed and discarded, which is what makes
this a whole-image representation rather than a spatial one, and why a CLIP embedding is 512
numbers instead of a grid.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..layers import LayerNorm, Transformer

__all__ = ["VisionTransformer"]


class VisionTransformer(nn.Module):
    """Image in, one vector out.

    Args:
        resolution: Square side the tower runs at.
        patch: Side of one patch.
        width: Transformer width.
        layers: Blocks in the stack.
        heads: Attention heads per block.
        embed_dim: Width of the shared space to project into.

    Examples:
        >>> VisionTransformer(224, 32, 768, 12, 12, 512)(pixels).shape   # doctest: +SKIP
        torch.Size([1, 512])
    """

    def __init__(
        self,
        resolution: int,
        patch: int,
        width: int,
        layers: int,
        heads: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        # No bias: the patch projection is a linear map over pixels and upstream leaves it
        # unbiased, so the checkpoint has no tensor for one.
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros((resolution // patch) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(torch.zeros(width, embed_dim))

    def forward(self, pixels: Tensor) -> Tensor:
        """Return ``(batch, embed_dim)``, unnormalised.

        Args:
            pixels: ``(batch, 3, resolution, resolution)``, already preprocessed.
        """
        x = self.conv1(pixels)                       # (batch, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)    # (batch, width, patches)
        x = x.permute(0, 2, 1)                       # (batch, patches, width)

        # The class token is a single learned vector broadcast across the batch. Written as an
        # addition to zeros rather than an expand, which is upstream's spelling and keeps the
        # dtype and device of ``x`` without a cast.
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)               # (batch, patches + 1, width)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # The stack is sequence-first, because nn.MultiheadAttention is.
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)

        # Row 0 is the class token. Everything else is dropped.
        return self.ln_post(x[:, 0, :]) @ self.proj
