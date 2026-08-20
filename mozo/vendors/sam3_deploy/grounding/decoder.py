# SPDX-License-Identifier: Apache-2.0
"""The DETR decoder: 200 object queries refined into boxes, six layers deep.

Derived from ``transformers/models/sam3/modeling_sam3.py`` (Apache-2.0).

Each layer attends three ways -- queries to each other, queries to the prompt, queries to the
image -- and then refines every query's *reference box*. The refinement is additive in logit
space: the current box is pushed through :func:`inverse_sigmoid`, a predicted delta is added, and
the result is squashed back. Six rounds of that is how a query converges on an object.

Four things here differ from the encoders and are easy to get backwards.

**These layers run sequence-first and post-norm.** Sequence-first because that is the layout the
weights were run under -- see :mod:`.layers` for why that changes the numbers. Post-norm because
the normalisation comes *after* the residual add, the opposite of the encoders. Same weights
either way; different numbers.

**The presence token rides at position 0.** It is not an object query -- it answers "is this
concept in the picture at all", and its logit gates every score downstream. It is prepended to
the queries, excluded from box refinement, given a zero row in the position encoding, and given a
zero row in the box bias so it attends to the whole image evenly.

**The normalisations are numbered in a different order than they run.** ``norm2`` follows
self-attention, ``norm1`` follows cross-attention to the image, ``catext_norm`` follows
cross-attention to the prompt, and ``norm3`` follows the feed-forward. Pairing them by number
puts two of them on the wrong residual, which is a large error, not a subtle one.

**Box-relative position bias is an additive bias, not a boolean mask.** For each query it
scores every image location by how that location sits relative to the query's current box, on a
log scale, per attention head. It is what makes the queries spatially selective before they have
converged.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import DECODER, DecoderSpec
from ..position import SinePositionEmbedding
from .boxes import box_cxcywh_to_xyxy, inverse_sigmoid
from .layers import Mlp

__all__ = ["Decoder"]

#: Base of the log compression applied to box-relative distances. Written once because it appears
#: on both sides of the same expression.
RPB_LOG_BASE = 8



class DecoderLayer(nn.Module):
    """Self-attention, then cross-attention to the prompt, then to the image. Post-norm."""

    def __init__(self, spec: DecoderSpec):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(spec.hidden, spec.heads)
        self.norm1 = nn.LayerNorm(spec.hidden)
        self.ca_text = nn.MultiheadAttention(spec.hidden, spec.heads)
        self.catext_norm = nn.LayerNorm(spec.hidden)
        self.cross_attn = nn.MultiheadAttention(spec.hidden, spec.heads)
        self.norm2 = nn.LayerNorm(spec.hidden)
        self.linear1 = nn.Linear(spec.hidden, spec.intermediate)
        self.linear2 = nn.Linear(spec.intermediate, spec.hidden)
        self.norm3 = nn.LayerNorm(spec.hidden)

    def forward(
        self,
        hidden: Tensor,
        query_pos: Tensor,
        prompt: Tensor,
        prompt_padding: Tensor | None,
        image: Tensor,
        image_key: Tensor,
        box_bias: Tensor | None,
    ) -> Tensor:
        """Run one decoder layer over ``(1 + queries, B, hidden)`` states, sequence-first."""
        # The presence token gets no positional offset, so prepend a zero row.
        query_pos = torch.cat([query_pos.new_zeros(1, *query_pos.shape[1:]), query_pos], dim=0)

        positioned = hidden + query_pos
        # ``need_weights`` decides which path ``nn.MultiheadAttention`` takes internally, and the
        # two accumulate differently. Which one reproduces upstream is not uniform: the two
        # unbiased attentions match the default, and the box-biased one below matches the fused
        # path. Measured per call rather than assumed -- see ``PROVENANCE.md``.
        attended, _ = self.self_attn(positioned, positioned, hidden)
        # ``norm2`` after self-attention and ``norm1`` after cross-attention -- the checkpoint's
        # numbering is not the order they run in, and reading it the natural way puts each
        # normalisation on the wrong residual. Established by tracing the published model.
        hidden = self.norm2(hidden + attended)

        positioned = hidden + query_pos
        attended, _ = self.ca_text(positioned, prompt, prompt, key_padding_mask=prompt_padding)
        hidden = self.catext_norm(hidden + attended)

        positioned = hidden + query_pos
        attended, _ = self.cross_attn(
            positioned, image_key, image, need_weights=False, attn_mask=box_bias
        )
        hidden = self.norm1(hidden + attended)

        return self.norm3(hidden + self.linear2(F.relu(self.linear1(hidden))))


class Decoder(nn.Module):
    """Object queries in, per-layer query features, reference boxes and presence logits out.

    Args:
        spec: Decoder geometry.
    """

    def __init__(self, spec: DecoderSpec = DECODER):
        super().__init__()
        self.spec = spec
        self.layers = nn.ModuleList(DecoderLayer(spec) for _ in range(spec.layers))
        self.norm = nn.LayerNorm(spec.hidden)

        self.query_embed = nn.Embedding(spec.queries, spec.hidden)
        #: Learned starting boxes, one per query, in logit space until sigmoided.
        self.reference_points = nn.Embedding(spec.queries, 4)
        self.bbox_embed = Mlp((spec.hidden, spec.hidden, spec.hidden, 4))

        self.presence_token = nn.Embedding(1, spec.hidden)
        self.presence_token_head = Mlp((spec.hidden, spec.hidden, spec.hidden, 1))
        self.presence_token_out_norm = nn.LayerNorm(spec.hidden)

        self.ref_point_head = Mlp((2 * spec.hidden, spec.hidden, spec.hidden))
        self.boxRPB_embed_x = Mlp((2, spec.hidden, spec.heads))
        self.boxRPB_embed_y = Mlp((2, spec.hidden, spec.heads))

        self.position_encoding = SinePositionEmbedding(features=spec.hidden // 2)

    def refine(self, queries: Tensor, boxes: Tensor) -> Tensor:
        """Push boxes one refinement step, additively in logit space.

        Args:
            queries: Normalised query features, any leading batch shape.
            boxes: The boxes those queries started from, matching shape.

        Returns:
            The refined boxes, back in ``[0, 1]``.
        """
        return (self.bbox_embed(queries) + inverse_sigmoid(boxes)).sigmoid()

    def _box_bias(self, boxes: Tensor, height: int, width: int) -> Tensor:
        """Score every image location against every query's current box, per head.

        Args:
            boxes: ``(B, Q, 4)`` reference boxes as ``(cx, cy, w, h)`` in ``[0, 1]``.
            height: Feature map height.
            width: Feature map width.

        Returns:
            ``(B, heads, Q + 1, height * width)`` additive attention bias, the leading row being
            the presence token's.
        """
        corners = box_cxcywh_to_xyxy(boxes)
        batch, queries, _ = corners.shape
        rows = torch.arange(0, height, device=boxes.device, dtype=boxes.dtype) / height
        columns = torch.arange(0, width, device=boxes.device, dtype=boxes.dtype) / width

        # Signed distance from each coordinate to the box's two edges on that axis.
        flat_corners = corners.reshape(-1, 1, 4)
        deltas_y = (rows.view(1, -1, 1) - flat_corners[:, :, 1:4:2])
        deltas_y = deltas_y.view(batch, queries, -1, 2)
        deltas_x = (columns.view(1, -1, 1) - flat_corners[:, :, 0:3:2])
        deltas_x = deltas_x.view(batch, queries, -1, 2)

        # Log scale, so nearby locations are separated finely and distant ones compressed.
        def compress(deltas: Tensor) -> Tensor:
            scaled = deltas * RPB_LOG_BASE
            return (
                torch.sign(scaled) * torch.log2(scaled.abs() + 1.0) / math.log2(RPB_LOG_BASE)
            )

        embedded_x = self.boxRPB_embed_x(compress(deltas_x))
        embedded_y = self.boxRPB_embed_y(compress(deltas_y))

        # The presence token needs a zero row so it attends to the whole image evenly, so the
        # result is written straight into a padded buffer rather than being built and then
        # copied. The outer sum broadcasts into that buffer's view in one pass, where
        # ``unsqueeze -> flatten -> permute -> pad`` materialised a 33 MB intermediate and then
        # a strided copy of it, six times per prompt.
        heads = embedded_x.shape[-1]
        bias = embedded_x.new_zeros(batch, heads, queries + 1, height * width)
        torch.add(
            embedded_y.permute(0, 3, 1, 2).unsqueeze(-1),
            embedded_x.permute(0, 3, 1, 2).unsqueeze(-2),
            out=bias[:, :, 1:].view(batch, heads, queries, height, width),
        )
        return bias

    def forward(
        self,
        image: Tensor,
        image_pos: Tensor,
        prompt: Tensor,
        prompt_padding: Tensor,
        height: int,
        width: int,
    ) -> dict[str, Tensor]:
        """Run all six layers.

        Args:
            image: ``(B, H*W, hidden)`` fused image tokens, batch-first as the fusion stage
                returns them; transposed internally.
            image_pos: ``(B, H*W, hidden)`` their position encoding.
            prompt: ``(P, B, hidden)`` sequence-first prompt tokens.
            prompt_padding: ``(B, P)`` True where the prompt slot is padding.
            height: Feature map height, for the box bias.
            width: Feature map width.

        Returns:
            ``queries`` ``(L, B, Q, hidden)`` normalised query features per layer, ``boxes``
            ``(L, B, Q, 4)`` the reference box each layer *started* from, ``final`` ``(B, Q, 4)``
            the last layer's refined box, and ``presence`` ``(L, B, 1)`` logits.
        """
        batch = image.shape[0]
        # Everything below is sequence-first.
        image = image.transpose(0, 1)
        image_pos = image_pos.transpose(0, 1)

        # Both operands are loop-invariant, so the cross-attention key is built once rather than
        # once per layer.
        image_key = image + image_pos

        queries = self.query_embed.weight.unsqueeze(1).expand(-1, batch, -1)
        boxes = self.reference_points.weight.unsqueeze(1).expand(-1, batch, -1).sigmoid()
        presence = self.presence_token.weight.unsqueeze(1).expand(-1, batch, -1)
        hidden = torch.cat([presence, queries], dim=0)

        per_layer_queries: list[Tensor] = []
        per_layer_boxes: list[Tensor] = [boxes.transpose(0, 1)]
        per_layer_presence: list[Tensor] = []

        for layer in self.layers:
            # Every matmul below runs sequence-first, because that is the layout these weights
            # were run under and ``nn.Linear`` accumulates differently on a transposed view.
            # Only ``_box_bias`` takes batch-first, which is the layout it was verified in.
            sine = self.position_encoding.encode_boxes(boxes.transpose(0, 1)).transpose(0, 1)
            query_pos = self.ref_point_head(sine)
            bias = self._box_bias(boxes.transpose(0, 1), height, width)
            # ``nn.MultiheadAttention`` wants an additive mask folded into the batch dimension.
            bias = bias.reshape(batch * self.spec.heads, bias.shape[2], bias.shape[3])

            hidden = layer(hidden, query_pos, prompt, prompt_padding, image, image_key, bias)

            normed = self.norm(hidden[1:])
            refined = self.refine(normed, boxes)
            per_layer_queries.append(normed.transpose(0, 1))
            per_layer_boxes.append(refined.transpose(0, 1))
            boxes = refined

            # Not clamped. Upstream carries a clamp setting and leaves the returned logit
            # untouched by it -- measured -10.719295 for a concept that is absent, where a clamp
            # at its own limit of 10 would have produced -10. ``transformers`` applies it here.
            logits = self.presence_token_head(
                self.presence_token_out_norm(hidden[:1])
            ).squeeze(-1)
            per_layer_presence.append(logits.transpose(0, 1))

        return {
            "queries": torch.stack(per_layer_queries),
            # Each layer's *input* box, so that pairing a layer's queries with its boxes and
            # re-applying :meth:`refine` reproduces that layer's prediction.
            "boxes": torch.stack(per_layer_boxes[:-1]),
            # The last layer's *output* box -- the actual prediction. Returned rather than left
            # for the caller to recompute from ``queries`` and ``boxes``.
            "final": per_layer_boxes[-1],
            "presence": torch.stack(per_layer_presence),
        }
