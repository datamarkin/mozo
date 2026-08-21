# SPDX-License-Identifier: Apache-2.0
"""Turning clicks, boxes and a previous mask into the tokens the decoder attends over.

Derived from ``transformers/models/sam3_tracker`` (Apache-2.0). See :mod:`.layers` for why this
package derives from there rather than borrowing SAM 2's copy of the same architecture.

**One prompt structure, not several.** A point is an ``(x, y)`` with a label: ``1`` include,
``0`` exclude. A box is its two corners carrying the reserved labels ``2`` and ``3`` -- there is
no box input, and ``num_point_embeddings`` being 4 is what that means. A refinement adds a mask
channel. Which fields you fill decides what you get.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import CLICK, ClickSpec
from .layers import LayerNorm2d

__all__ = ["PromptEncoder"]


class PositionalEmbedding(nn.Module):
    """Random Fourier features over coordinates normalised to ``[0, 1]``.

    The Gaussian matrix is a buffer loaded from the checkpoint, not resampled -- it is as much a
    weight as anything with ``weight`` in its name, and drawing a new one would move every point.
    """

    def __init__(self, spec: ClickSpec = CLICK) -> None:
        super().__init__()
        self.register_buffer("positional_embedding", torch.zeros(2, spec.hidden // 2))

    def forward(self, coords: Tensor, side: int) -> Tensor:
        scaled = coords / side
        scaled = 2 * scaled - 1
        scaled = scaled.to(self.positional_embedding.dtype) @ self.positional_embedding
        scaled = 2 * math.pi * scaled
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class MaskEmbedding(nn.Module):
    """Downscale a previous call's mask to the feature grid, so it can be refined."""

    def __init__(self, spec: ClickSpec = CLICK) -> None:
        super().__init__()
        narrow = spec.mask_channels // 4
        self.conv1 = nn.Conv2d(1, narrow, kernel_size=2, stride=2)
        self.layer_norm1 = LayerNorm2d(narrow, eps=spec.layer_norm_eps)
        self.conv2 = nn.Conv2d(narrow, spec.mask_channels, kernel_size=2, stride=2)
        self.layer_norm2 = LayerNorm2d(spec.mask_channels, eps=spec.layer_norm_eps)
        self.conv3 = nn.Conv2d(spec.mask_channels, spec.hidden, kernel_size=1)

    def forward(self, masks: Tensor) -> Tensor:
        x = F.gelu(self.layer_norm1(self.conv1(masks)))
        x = F.gelu(self.layer_norm2(self.conv2(x)))
        return self.conv3(x)


class PromptEncoder(nn.Module):
    """Embed a prompt into sparse tokens and a dense feature-grid bias.

    Args:
        spec: Click geometry -- the square prompts are scaled into, and the grid they land on.

    Attributes:
        grid: Side of the feature map, in tokens.
        mask_input_size: Side a ``mask_input`` must have, which is four times the grid.
    """

    def __init__(self, spec: ClickSpec = CLICK) -> None:
        super().__init__()
        self.spec = spec
        self.grid = spec.grid
        self.mask_input_size = 4 * self.grid

        self.shared_embedding = PositionalEmbedding(spec)
        self.mask_embed = MaskEmbedding(spec)
        self.point_embed = nn.Embedding(spec.point_embeddings, spec.hidden)
        self.not_a_point_embed = nn.Embedding(1, spec.hidden)
        self.no_mask_embed = nn.Embedding(1, spec.hidden)

    def dense_positions(self) -> Tensor:
        """The position encoding of every cell of the feature grid, as the decoder wants it.

        A function of the grid alone, so it is the same tensor on every call.
        """
        grid = self.grid
        axis = torch.arange(grid, dtype=torch.float32, device=self.point_embed.weight.device)
        ys, xs = torch.meshgrid(axis, axis, indexing="ij")
        # The embedding only touches the last axis, so the grid goes in as it is.
        coords = torch.stack([xs + 0.5, ys + 0.5], dim=-1)
        return self.shared_embedding(coords, grid).permute(2, 0, 1)[None]

    def forward(
        self,
        points: Tensor | None,
        labels: Tensor | None,
        masks: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Embed one prompt.

        Args:
            points: ``(B, N, 2)`` x, y already scaled into the encoder's square.
            labels: ``(B, N)`` -- 1 include, 0 exclude, 2 and 3 a box's corners.
            masks: ``(B, 1, mask_input_size, mask_input_size)`` logits to refine.

        Returns:
            Sparse tokens ``(1, B, N, hidden)`` -- one image, ``B`` prompt sets -- and a dense
            ``(1, hidden, grid, grid)`` bias.
        """
        prompts = 1 if points is None else points.shape[0]

        if points is None:
            sparse = self.point_embed.weight.new_zeros((1, prompts, 0, self.spec.hidden))
        else:
            # One padding token, always. Upstream pads whenever no separate box input is given,
            # and this package never gives one -- a box arrives folded into the points as its two
            # corners carrying labels 2 and 3. So the encoder is always in the padded case, and
            # the extra not-a-point token is part of the sequence the decoder was trained on.
            shifted = F.pad(points + 0.5, (0, 0, 0, 1))  # and to the centre of the pixel
            # ``B`` prompt sets against one image, so the batch lands on the *prompt* axis and
            # the image axis stays 1. Putting it on the image axis instead makes a second prompt
            # set fail to broadcast against the single image it is asking about.
            marks = F.pad(labels, (0, 1), value=-1)[None]
            encoded = self.shared_embedding(shifted[None], self.spec.image_size)
            # ``where`` rather than indexing, so every label takes the same path: a padding
            # label of -1 becomes the not-a-point embedding and everything else adds the
            # embedding its label selects.
            sparse = torch.where(
                marks[..., None] == -1,
                self.not_a_point_embed.weight,
                encoded + self.point_embed(marks.clamp(min=0)),
            )

        if masks is None:
            dense = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, self.grid, self.grid
            )
        else:
            dense = self.mask_embed(masks)
        return sparse, dense
