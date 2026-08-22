# SPDX-License-Identifier: Apache-2.0
"""Sine position encodings, in the three shapes Grounding DINO needs.

All three are the same idea -- a coordinate becomes interleaved sines and cosines over a
geometric series of wavelengths -- and they differ in what they encode and how the halves are
ordered. They are written out separately rather than folded together because the orderings are
not interchangeable and folding them would need a flag whose wrong value still runs.

**The temperature is 20, not 10000**, for the image grid. Upstream sets ``pe_temperatureH`` and
``pe_temperatureW`` to 20 in both published configs. Every other DETR in this tree uses 10000, so
a copied position encoding brings the wrong constant with it and moves every box.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["image_position", "query_sine_embed", "token_position"]


def image_position(
    mask: Tensor,
    num_pos_feats: int = 128,
    temperature_h: int = 20,
    temperature_w: int = 20,
) -> Tensor:
    """Encode each pixel of a feature map by its normalised position.

    Args:
        mask: ``(batch, height, width)``, True where the image is padding. The position is
            counted over the *unpadded* region, so a padded batch still encodes each image's own
            geometry.
        num_pos_feats: Channels per axis. The result carries twice this.
        temperature_h: Wavelength base for the vertical axis.
        temperature_w: Wavelength base for the horizontal axis.

    Returns:
        ``(batch, 2 * num_pos_feats, height, width)``, y-encoding first.
    """
    scale = 2 * math.pi
    eps = 1e-6

    valid = ~mask
    y_embed = valid.cumsum(1, dtype=torch.float32)
    x_embed = valid.cumsum(2, dtype=torch.float32)
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    steps = torch.arange(num_pos_feats, dtype=torch.float32, device=mask.device)
    dim_x = temperature_w ** (2 * torch.div(steps, 2, rounding_mode="floor") / num_pos_feats)
    dim_y = temperature_h ** (2 * torch.div(steps, 2, rounding_mode="floor") / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_x
    pos_y = y_embed[:, :, :, None] / dim_y
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def query_sine_embed(pos_tensor: Tensor) -> Tensor:
    """Encode a reference point or box for the decoder's conditional query.

    Args:
        pos_tensor: ``(queries, batch, 2)`` as ``(x, y)`` or ``(queries, batch, 4)`` as
            ``(x, y, w, h)``, all normalised to [0, 1].

    Returns:
        ``(queries, batch, 256)`` for a point, ``(queries, batch, 512)`` for a box. The halves
        are ordered ``y`` then ``x`` -- the reverse of the input -- and then ``w``, ``h``.
    """
    scale = 2 * math.pi
    steps = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * torch.div(steps, 2, rounding_mode="floor") / 128)

    def encode(values: Tensor) -> Tensor:
        scaled = values[:, :, None] * scale / dim_t
        return torch.stack((scaled[:, :, 0::2].sin(), scaled[:, :, 1::2].cos()), dim=3).flatten(2)

    pos_x = encode(pos_tensor[:, :, 0])
    pos_y = encode(pos_tensor[:, :, 1])
    if pos_tensor.size(-1) == 2:
        return torch.cat((pos_y, pos_x), dim=2)
    if pos_tensor.size(-1) == 4:
        return torch.cat(
            (pos_y, pos_x, encode(pos_tensor[:, :, 2]), encode(pos_tensor[:, :, 3])), dim=2
        )
    raise ValueError(f"reference must have 2 or 4 coordinates, got {pos_tensor.size(-1)}")


def token_position(
    pos_tensor: Tensor, num_pos_feats: int = 256, temperature: int = 10000
) -> Tensor:
    """Encode a text token's position within its own phrase.

    The positions handed here restart at zero for every phrase in the caption, which is what
    makes ``"a person"`` and ``"a mug"`` occupy the same positional space rather than being
    token 1-2 and token 4-5 of one sentence.

    Args:
        pos_tensor: ``(batch, tokens, 1)``.
        num_pos_feats: Channels produced per input coordinate.
        temperature: Wavelength base.

    Returns:
        ``(batch, tokens, num_pos_feats)``. The x/y swap :func:`query_sine_embed` performs is
        deliberately absent -- there is one axis here, and nothing to exchange it with.
    """
    scale = 2 * math.pi
    steps = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(steps, 2, rounding_mode="floor") / num_pos_feats)

    def encode(values: Tensor) -> Tensor:
        scaled = values * scale / dim_t
        return torch.stack((scaled[..., 0::2].sin(), scaled[..., 1::2].cos()), dim=3).flatten(2)

    return torch.cat(
        [encode(value) for value in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)], dim=-1
    )
