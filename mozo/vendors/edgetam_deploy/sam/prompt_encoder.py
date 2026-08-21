# SPDX-License-Identifier: Apache-2.0
"""Turning clicks, boxes and a previous mask into the tokens the decoder attends to.

Taken from ``sam2/modeling/sam/prompt_encoder.py`` in ``facebookresearch/EdgeTAM``, with one
deliberate change.

**The label embeddings are selected with** :func:`torch.where` **rather than by indexed
assignment.** EdgeTAM writes::

    point_embedding[labels == -1] = 0.0
    point_embedding[labels == -1] += self.not_a_point_embed.weight
    point_embedding[labels == 0] += self.point_embeddings[0].weight

which is what SAM 2 used to do and what SAM 2 no longer does -- upstream replaced it with
``torch.where`` because a boolean mask index is data-dependent and does not trace, so an exported
graph either bakes in one prompt's labels or refuses to convert. EdgeTAM forked before that
change. The two forms are arithmetically identical on every input: each row is either overwritten
by ``not_a_point_embed`` or has exactly one label embedding added, and rows carrying a label
outside ``{-1, 0, 1, 2, 3}`` are left as the bare positional encoding under both. This package
carries the traceable one, and the gate compares against the other.

There is no separate box input on the path this package runs. A box is spelled as its two corners
with labels 2 and 3, folded into the point list ahead of any clicks, which is exactly what
upstream's own ``SAM2ImagePredictor._predict`` does before calling this module with ``boxes=None``.
``_embed_boxes`` is therefore unreachable here; it is kept rather than deleted so the module still
diffs cleanly against upstream, and because dropping it while keeping the parameter that selects
it would make the signature lie.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..layers import LayerNorm2d
from ..position_encoding import PositionEmbeddingRandom

__all__ = ["PromptEncoder"]


class PromptEncoder(nn.Module):
    """Encode prompts into the sparse and dense embeddings the mask decoder takes.

    Args:
        embed_dim: Token width.
        image_embedding_size: ``(H, W)`` of the image embedding the dense output must match.
        input_image_size: ``(H, W)`` of the encoder's square, which is what point coordinates
            are normalised against.
        mask_in_chans: Hidden width of the mask-downscaling stack.
        activation: Activation inside that stack.
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: tuple[int, int],
        input_image_size: tuple[int, int],
        mask_in_chans: int,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> Tensor:
        """The positional encoding of the image grid, as the decoder wants it.

        Returns:
            ``(1, embed_dim, H, W)`` for the image embedding's ``(H, W)``.
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: Tensor, labels: Tensor, pad: bool) -> Tensor:
        """Embed point prompts: a positional encoding plus a per-label token."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # See the module docstring: upstream's indexed assignment, written so it traces. The
        # first branch replaces the encoding rather than adding to it, which is why it starts
        # from zeros -- a padding slot carries no position.
        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        for label in range(self.num_point_embeddings):
            point_embedding = torch.where(
                (labels == label).unsqueeze(-1),
                point_embedding + self.point_embeddings[label].weight,
                point_embedding,
            )
        return point_embedding

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        """Embed box prompts. Unreachable from this package -- see the module docstring."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: Tensor) -> Tensor:
        """Embed a low-resolution mask from a previous call."""
        return self.mask_downscaling(masks)

    def _get_batch_size(
        self,
        points: Optional[tuple[Tensor, Tensor]],
        boxes: Optional[Tensor],
        masks: Optional[Tensor],
    ) -> int:
        """How many prompts are being encoded, from whichever input carries a batch."""
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[tuple[Tensor, Tensor]],
        boxes: Optional[Tensor],
        masks: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Encode whatever prompt was given.

        Args:
            points: ``(coords, labels)``, each ``(B, N, ...)``. Coordinates are in the encoder's
                square. Labels are 1 to include, 0 to exclude, 2 and 3 for a box's corners.
            boxes: ``(B, 4)`` in the encoder's square. Not used by this package.
            masks: ``(B, 1, 4H, 4W)`` logits from an earlier call.

        Returns:
            Sparse embeddings ``(B, N, embed_dim)`` for the points and boxes, and dense
            embeddings ``(B, embed_dim, H, W)`` for the mask -- or a learned "no mask" constant
            broadcast to that shape when none was given.
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
