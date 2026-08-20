# SPDX-License-Identifier: Apache-2.0
"""Turning exemplar boxes -- "find more things like this one" -- into prompt tokens.

Derived from ``transformers/models/sam3/modeling_sam3.py`` (Apache-2.0).

Two things about this module are easy to get wrong, and both were found by comparing behaviour
against the published model rather than by reading.

**It runs on every prompt, including text-only ones.** With no boxes it still emits a CLS token
that has cross-attended to the image through three blocks, and that token is concatenated onto
the text tokens before fusion -- measured as a prompt of 33 tokens where the text alone is 32.
``transformers`` skips this module when ``input_boxes`` is empty, which drops the token the
weights were trained with. Here it always runs.

**Padding masks are ``True`` for padding.** ``transformers`` uses ``True`` for *valid* and sums
masks to recover sequence lengths. Both conventions appear in the same problem, and mixing them
attends to precisely the tokens meant to be ignored.

``roi_align`` is the one thing here that comes from ``torchvision``. It is a declared dependency
of mozo, as it already was in practice -- ``rfdetr_deploy`` and ``depth_anything_v2_deploy`` both
import it at module scope -- so it is imported normally rather than defended against.
"""

from __future__ import annotations

import torch
import torchvision
from torch import Tensor, nn

from ..config import GEOMETRY, GeometrySpec
from ..position import SinePositionEmbedding
from .boxes import box_cxcywh_to_xyxy
from .layers import FusionLayer

__all__ = ["GeometryEncoder"]


class GeometryEncoder(nn.Module):
    """Exemplar boxes plus a learned CLS token, contextualised against the image.

    Args:
        spec: Block widths and the ROI pooling size.
    """

    def __init__(self, spec: GeometrySpec = GEOMETRY):
        super().__init__()
        self.spec = spec
        self.position_encoding = SinePositionEmbedding(features=spec.hidden // 2)

        #: 0 negative, 1 positive. A box's label is added to its embedding, not concatenated.
        self.label_embed = nn.Embedding(2, spec.hidden)
        #: Always emitted, always valid -- see the module docstring.
        self.cls_embed = nn.Embedding(1, spec.hidden)

        # Three independent views of a box, summed: its raw coordinates, the image features it
        # encloses, and a sinusoidal encoding of where it sits.
        self.boxes_direct_project = nn.Linear(4, spec.hidden)
        self.boxes_pool_project = nn.Conv2d(spec.hidden, spec.hidden, spec.roi_size)
        self.boxes_pos_enc_project = nn.Linear(spec.hidden + 2, spec.hidden)

        self.img_pre_norm = nn.LayerNorm(spec.hidden)
        self.final_proj = nn.Linear(spec.hidden, spec.hidden)
        self.norm = nn.LayerNorm(spec.hidden)
        # Sequence-first, which is how these weights were run. See ``layers.py``.
        self.encode = nn.ModuleList(
            FusionLayer(spec, batch_first=False) for _ in range(spec.layers)
        )
        self.encode_norm = nn.LayerNorm(spec.hidden)

    def _encode_boxes(self, boxes: Tensor, labels: Tensor, features: Tensor) -> Tensor:
        """Embed ``(B, N, 4)`` normalised cxcywh boxes against the image features."""
        batch, count = boxes.shape[:2]
        embedded = self.boxes_direct_project(boxes)

        if count:
            # Normalising the feature map feeds pooling and nothing else, so it stays inside this
            # branch -- on the text-only path it would be a LayerNorm over 1.3M elements thrown
            # away.
            features = self.img_pre_norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            map_height, map_width = features.shape[-2:]
            corners = box_cxcywh_to_xyxy(boxes)
            scale = corners.new_tensor(
                [map_width, map_height, map_width, map_height]
            ).view(1, 1, 4)
            pooled = torchvision.ops.roi_align(
                features, (corners * scale).float().unbind(0), self.spec.roi_size
            )
            embedded = embedded + self.boxes_pool_project(pooled).view(batch, count, -1)

        center_x, center_y, box_width, box_height = boxes.unbind(-1)
        pos_x, pos_y = self.position_encoding.encode_positions(
            center_x.flatten(), center_y.flatten()
        )
        # Width is spelled out rather than inferred: with no boxes there are no elements to
        # infer from, and ``-1`` raises instead of yielding an empty tensor of the right shape.
        width = 2 * self.position_encoding.features + 2
        encoded = torch.cat(
            (pos_y, pos_x, box_height.reshape(-1, 1), box_width.reshape(-1, 1)), dim=1
        ).view(batch, count, width)
        embedded = embedded + self.boxes_pos_enc_project(encoded)

        return self.label_embed(labels.long()) + embedded

    @torch.no_grad()
    def forward(
        self,
        boxes: Tensor,
        labels: Tensor,
        features: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode a geometric prompt.

        Args:
            boxes: ``(B, N, 4)`` normalised cxcywh. ``N`` may be 0.
            labels: ``(B, N)`` 1 positive, 0 negative.
            features: ``(B, hidden, H, W)`` the coarsest surviving FPN level.
            positions: ``(B, hidden, H, W)`` its position encoding.

        Returns:
            ``(N + 1, B, hidden)`` prompt tokens -- sequence-first, as the fusion stage and the
            published model both expect -- and their ``(B, N + 1)`` padding mask. The extra slot
            is the always-valid CLS token.
        """
        batch = boxes.shape[0]

        # Cross-attention reads the raw feature map; pooling normalises its own copy.
        flat = features.flatten(2).transpose(1, 2)
        flat_positions = positions.flatten(2).transpose(1, 2)

        embedded = self._encode_boxes(boxes, labels, features)

        cls = self.cls_embed.weight.view(1, 1, -1).expand(batch, -1, -1)
        tokens = torch.cat([embedded, cls], dim=1)
        # Every slot is real: a caller passes the boxes it has, and the CLS token is always
        # valid. The mask exists because the fusion stage concatenates this onto padded text.
        mask = torch.zeros(batch, tokens.shape[1], dtype=torch.bool, device=tokens.device)

        tokens = self.norm(self.final_proj(tokens))

        # Everything from here runs sequence-first. The cross-attention key is the image plus
        # its position encoding; both are loop-invariant, so it is built once rather than in
        # each of the three layers.
        tokens = tokens.transpose(0, 1)
        flat = flat.transpose(0, 1)
        key = flat + flat_positions.transpose(0, 1)
        for layer in self.encode:
            tokens = layer(tokens, flat, key, target_padding=mask)
        return self.encode_norm(tokens), mask
