# SPDX-License-Identifier: Apache-2.0
"""Sinusoidal position encoding, in the three shapes SAM 3 asks for.

Derived from ``transformers/models/sam3/modeling_sam3.py`` (Apache-2.0).

This lives at the package root rather than under ``vision/`` because all three of its users sit in
different subpackages: the FPN needs whole grids (:meth:`SinePositionEmbedding.forward`), the
geometry encoder needs box centres (:meth:`~SinePositionEmbedding.encode_positions`), and the DETR
decoder needs whole boxes (:meth:`~SinePositionEmbedding.encode_boxes`). ``sam2_deploy`` puts its
equivalent at the root for the same reason.

All three share one frequency table and one sin/cos interleave; they differ only in what they
encode and whether the coordinates are normalised first.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["SinePositionEmbedding"]


class SinePositionEmbedding(nn.Module):
    """Sine/cosine position encoding.

    Args:
        features: Half the output width for grids and points -- 128 gives 256 channels, since x
            and y are concatenated. Boxes concatenate four, giving ``features * 4``.
        temperature: Frequency base.
    """

    def __init__(self, features: int = 128, temperature: int = 10000):
        super().__init__()
        self.features = features
        self.temperature = temperature
        self.scale = 2 * math.pi
        # Grids depend only on their shape, never on the image. Caching them here is what keeps
        # them out of any per-image cache downstream -- see the note in ``vision/encoder.py``.
        self._grids: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}

    def bands(self, device: torch.device | str) -> Tensor:
        """The shared frequency table: ``temperature ** (2 * (i // 2) / features)``."""
        index = torch.arange(self.features, dtype=torch.float32, device=device)
        return self.temperature ** (2 * (index // 2) / self.features)

    @staticmethod
    def interleave(scaled: Tensor, dim: int) -> Tensor:
        """Interleave ``sin`` of the even channels with ``cos`` of the odd ones."""
        return torch.stack(
            (scaled[..., 0::2].sin(), scaled[..., 1::2].cos()), dim=dim
        ).flatten(dim - 1)

    @torch.no_grad()
    def encode_positions(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Encode 1-D coordinate pairs -- box centres, rather than a whole grid.

        These arrive already normalised to ``[0, 1]``, so unlike :meth:`forward` there is nothing
        to divide by; only the scale is applied.

        Args:
            x: ``(N,)`` x coordinates in ``[0, 1]``.
            y: ``(N,)`` y coordinates in ``[0, 1]``.

        Returns:
            Two ``(N, features)`` encodings.
        """
        bands = self.bands(x.device)
        return (
            self.interleave((x * self.scale)[:, None] / bands, 2),
            self.interleave((y * self.scale)[:, None] / bands, 2),
        )

    @torch.no_grad()
    def encode_boxes(self, boxes: Tensor) -> Tensor:
        """Encode whole boxes -- centre *and* extent.

        The decoder conditions each object query on its current reference box, so all four numbers
        are encoded and concatenated, giving ``features * 4`` channels.

        Args:
            boxes: ``(B, Q, 4)`` as ``(cx, cy, w, h)``, normalised to ``[0, 1]``.

        Returns:
            ``(B, Q, features * 4)``.
        """
        bands = self.bands(boxes.device)
        # Order is y, x, w, h -- y before x, matching :meth:`forward`.
        return torch.cat(
            [
                self.interleave((boxes[:, :, index] * self.scale)[:, :, None] / bands, 3)
                for index in (1, 0, 2, 3)
            ],
            dim=2,
        )

    @torch.no_grad()
    def forward(self, height: int, width: int, device, dtype) -> Tensor:
        """Return the ``(1, features * 2, height, width)`` encoding for a map of this size.

        The result depends only on the arguments, so it is computed once per distinct shape and
        reused. At SAM 3's fixed resolution that saves about 42 ms and 110 MB on every image.

        The returned tensor is shared between callers and **must not be modified in place**.
        """
        key = (height, width, torch.device(device), dtype)
        cached = self._grids.get(key)
        if cached is not None:
            return cached

        # Positions are 1-based and normalised by the last one, which is what makes this
        # resolution-independent -- the same encoding stretches over any grid.
        y = torch.arange(1, height + 1, dtype=torch.float32, device=device).view(1, -1, 1)
        y = y.repeat(1, 1, width)
        x = torch.arange(1, width + 1, dtype=torch.float32, device=device).view(1, 1, -1)
        x = x.repeat(1, height, 1)
        eps = 1e-6
        y = y / (y[:, -1:, :] + eps) * self.scale
        x = x / (x[:, :, -1:] + eps) * self.scale

        bands = self.bands(device)
        pos_x = self.interleave(x[:, :, :, None] / bands, 4)
        pos_y = self.interleave(y[:, :, :, None] / bands, 4)
        grid = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).to(dtype)

        self._grids[key] = grid
        return grid
