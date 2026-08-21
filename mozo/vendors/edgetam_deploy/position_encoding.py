# SPDX-License-Identifier: Apache-2.0
"""The two position encodings the image path uses, and neither of the ones it does not.

Taken verbatim from ``sam2/modeling/position_encoding.py`` in ``facebookresearch/EdgeTAM``.

:class:`PositionEmbeddingSine` is what the neck adds to each feature map. Its ``encode_boxes`` and
``encode_points`` methods are left behind: both encode object pointers for the video tracker, and
nothing on the image path calls either.

:class:`PositionEmbeddingRandom` is what the prompt encoder positions clicks with. Its Gaussian
matrix is a buffer carried in the checkpoint rather than resampled, which is why the same click
lands in the same place on every machine.

Also left behind: the rotary machinery below them -- ``init_t_xy``, ``compute_axial_cis``,
``reshape_for_broadcast``, ``apply_rotary_enc`` and EdgeTAM's own ``apply_rotary_enc_v2``. Rotary
embeddings appear only in memory attention, which is the video path, and ``apply_rotary_enc_v2``
is EdgeTAM's addition for the 2-D spatial perceiver -- the paper's contribution, and entirely
about compressing a memory bank this package does not have.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn

__all__ = ["PositionEmbeddingRandom", "PositionEmbeddingSine"]


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position encoding over an image grid.

    Args:
        num_pos_feats: Output channels. Half go to the row encoding and half to the column one.
        temperature: Base of the frequency ladder.
        normalize: Scale coordinates to ``[0, scale]`` before encoding.
        scale: What normalised coordinates run to. Defaults to ``2*pi``.
    """

    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        # Keyed on spatial size, so the three feature maps of one image size fill it once and
        # every later image reuses them. A plain dict rather than a buffer because it holds no
        # learned state -- it is derived from the shape, and a checkpoint that carried it would
        # be carrying arithmetic.
        self.cache = {}

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Encode positions for a map the shape of *x*.

        Args:
            x: ``(B, C, H, W)``. Only its shape and device are read.

        Returns:
            ``(B, num_pos_feats, H, W)``.
        """
        cache_key = (x.shape[-2], x.shape[-1])
        if cache_key in self.cache:
            return self.cache[cache_key][None].repeat(x.shape[0], 1, 1, 1)
        y_embed = (
            torch.arange(1, x.shape[-2] + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .repeat(x.shape[0], 1, x.shape[-1])
        )
        x_embed = (
            torch.arange(1, x.shape[-1] + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .repeat(x.shape[0], x.shape[-2], 1)
        )

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(nn.Module):
    """Position encoding by projection onto random spatial frequencies.

    Args:
        num_pos_feats: Half the output width -- a sine and a cosine are emitted per frequency.
        scale: Standard deviation of the Gaussian matrix. Non-positive means one.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> Tensor:
        """Encode a whole grid, which is what the decoder adds to the image embedding.

        Args:
            size: ``(H, W)`` of the grid.

        Returns:
            ``(num_pos_feats * 2, H, W)``.
        """
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, coords_input: Tensor, image_size: tuple[int, int]) -> Tensor:
        """Encode points given in pixels rather than in ``[0, 1]``.

        Args:
            coords_input: ``(B, N, 2)`` x, y in the encoder's square.
            image_size: ``(H, W)`` of that square.

        Returns:
            ``(B, N, num_pos_feats * 2)``.
        """
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
