# SPDX-License-Identifier: Apache-2.0
"""Multi-scale deformable attention, in plain PyTorch.

Upstream ships this twice: a CUDA/C++ extension (``groundingdino._C``) built by ``setup.py``, and
a ``grid_sample`` fallback for when the extension is absent. Only the fallback is carried, for
three reasons: it needs no compiler, it is the path that runs on every device mozo targets, and
it is the one whose numbers can be compared against the reference on this machine. The extension
is a faster route to the same arithmetic, not a different model.

This is a second copy of an operator :mod:`mozo.vendors.rfdetr_deploy` also carries. That is
deliberate and is the rule for this tree: a vendor may not import another, because then one
family's refactor could move another family's boxes.
"""

from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["MultiScaleDeformableAttention", "deformable_attention"]


def deformable_attention(
    value: Tensor,
    spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Sample *value* at *sampling_locations* and combine by *attention_weights*.

    Args:
        value: ``(batch, tokens, heads, channels)``, every level concatenated.
        spatial_shapes: ``(levels, 2)`` of ``(height, width)``.
        sampling_locations: ``(batch, queries, heads, levels, points, 2)``, normalised to [0, 1].
        attention_weights: ``(batch, queries, heads, levels, points)``, already softmaxed.

    Returns:
        ``(batch, queries, heads * channels)``.
    """
    batch, _, heads, channels = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape

    # Read off the device once, not per level. `spatial_shapes` lives wherever the model does, so
    # every `int(...)` on it is a device-to-host sync -- and this function runs twelve times per
    # image. Measured on MPS, eight scalar extractions cost 2.94 ms against 0.40 ms for one
    # `tolist()`, which is about 30 ms an image spent waiting rather than computing.
    shapes = spatial_shapes.tolist()

    per_level = value.split([height * width for height, width in shapes], dim=1)
    # grid_sample wants [-1, 1]; the reference points are [0, 1].
    grids = 2 * sampling_locations - 1

    sampled = []
    for level, (height, width) in enumerate(shapes):
        # (batch, hw, heads, channels) -> (batch * heads, channels, height, width)
        source = (
            per_level[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, channels, height, width)
        )
        # (batch, queries, heads, points, 2) -> (batch * heads, queries, points, 2)
        grid = grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(
            F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        )

    weights = attention_weights.transpose(1, 2).reshape(batch * heads, 1, queries, levels * points)
    output = (
        (torch.stack(sampled, dim=-2).flatten(-2) * weights)
        .sum(-1)
        .view(batch, heads * channels, queries)
    )
    return output.transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """Attend to a few learned points per level instead of to every token.

    Args:
        embed_dim: Model width.
        num_heads: Attention heads.
        num_levels: Feature levels attended over.
        num_points: Sampling points per head per level.

    Examples:
        >>> attn = MultiScaleDeformableAttention(256, 8, 4, 4)     # doctest: +SKIP
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Run one deformable attention step, batch-first throughout.

        Args:
            query: ``(batch, queries, embed_dim)``.
            reference_points: ``(batch, queries, levels, 2)`` or ``(..., 4)``. Two numbers is a
                point; four is a box, and the sampling window then scales with its size.
            value: ``(batch, tokens, embed_dim)``.
            spatial_shapes: ``(levels, 2)``.
            key_padding_mask: ``(batch, tokens)``, True where padded.
        """
        batch, queries, _ = query.shape
        _, tokens, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(batch, tokens, self.num_heads, -1)

        offsets = self.sampling_offsets(query).view(
            batch, queries, self.num_heads, self.num_levels, self.num_points, 2
        )
        weights = self.attention_weights(query).view(
            batch, queries, self.num_heads, self.num_levels * self.num_points
        )
        weights = weights.softmax(-1).view(
            batch, queries, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            # Offsets are in feature-map cells, so they are divided by the level's own size.
            normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            locations = (
                reference_points[:, :, None, :, None, :]
                + offsets / normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            # A box reference: the sampling window is half the box, spread over the points.
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"reference points must have 2 or 4 coordinates, got {reference_points.shape[-1]}"
            )

        return self.output_proj(
            deformable_attention(value, spatial_shapes, locations, weights)
        )
