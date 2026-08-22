# SPDX-License-Identifier: Apache-2.0
"""The encoder, the query selection between them, and the cross-modality decoder.

The encoder runs three things per layer, in this order: fusion (image ↔ text), a text
self-attention over the phrase-isolating mask, and deformable self-attention over the image
pyramid. The decoder then runs 900 queries through self-attention, text cross-attention and
deformable image cross-attention, refining a box at every layer.

Between them sits language-guided query selection: every image token is scored against the text,
the best 900 are taken, and their positions become the decoder's initial boxes. That is what
``two_stage_type="standard"`` names upstream.

Not carried: denoising training (``dn_*``), the ``no`` two-stage branch, query patterns, layer
sharing, gradient checkpointing, and the auxiliary per-layer outputs. All are training or ablation
paths; none is reachable from a published checkpoint at inference.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .fuse import BiAttentionBlock
from .ops import MultiScaleDeformableAttention
from .position import query_sine_embed, token_position

__all__ = ["Transformer"]


def _clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList(copy.deepcopy(module) for _ in range(count))


def inverse_sigmoid(x: Tensor, eps: float = 1e-3) -> Tensor:
    """The inverse of ``sigmoid``, clamped so 0 and 1 do not become infinities."""
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class MLP(nn.Module):
    """A plain feed-forward stack with ReLU between layers and none at the end."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        widths = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + widths, widths + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class ContrastiveEmbed(nn.Module):
    """Score every query against every text token. No parameters -- it is a dot product.

    The result is padded out to ``max_text_len`` with ``-inf`` so the logit tensor has a fixed
    width whatever the prompt length. That is why the published output is ``(queries, 256)`` and
    not ``(queries, tokens)``, and why columns past the prompt are never selected.
    """

    def __init__(self, max_text_len: int = 256) -> None:
        super().__init__()
        self.max_text_len = max_text_len

    def forward(self, x: Tensor, text: Tensor, text_token_mask: Tensor) -> Tensor:
        result = x @ text.transpose(-1, -2)
        result = result.masked_fill(~text_token_mask[:, None, :], float("-inf"))

        padded = torch.full(
            (*result.shape[:-1], self.max_text_len), float("-inf"), device=result.device
        )
        padded[..., : result.shape[-1]] = result
        return padded


def encoder_output_proposals(
    memory: Tensor, padding_mask: Tensor, spatial_shapes: Tensor
) -> tuple[Tensor, Tensor]:
    """Turn every image token into a candidate box centred on itself.

    Each level proposes boxes of a fixed size -- 0.05 of the image, doubling per level -- centred
    on each cell. Tokens that are padding, or whose proposal escapes [0.01, 0.99], are zeroed in
    the memory and pushed to ``inf`` in the proposals so selection cannot pick them.
    """
    batch, _, _ = memory.shape
    proposals = []
    start = 0
    for level, (height, width) in enumerate(spatial_shapes):
        height, width = int(height), int(width)
        level_mask = padding_mask[:, start : start + height * width].view(batch, height, width, 1)
        valid_h = torch.sum(~level_mask[:, :, 0, 0], 1)
        valid_w = torch.sum(~level_mask[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, height - 1, height, dtype=torch.float32, device=memory.device),
            torch.linspace(0, width - 1, width, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).view(batch, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(batch, -1, -1, -1) + 0.5) / scale
        size = torch.ones_like(grid) * 0.05 * (2.0**level)
        proposals.append(torch.cat((grid, size), -1).view(batch, -1, 4))
        start += height * width

    output_proposals = torch.cat(proposals, 1)
    valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~valid, float("inf"))

    output_memory = memory.masked_fill(padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~valid, float(0))
    return output_memory, output_proposals


class _TextLayer(nn.Module):
    """Post-norm self-attention over text tokens, masked to keep phrases apart."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.nhead = nhead

    def forward(self, src: Tensor, src_mask: Tensor, pos: Tensor) -> Tensor:
        if src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            src_mask = src_mask.repeat(self.nhead, 1, 1)
        q = k = src + pos
        src = self.norm1(src + self.self_attn(q, k, value=src, attn_mask=src_mask)[0])
        return self.norm2(src + self.linear2(F.relu(self.linear1(src))))


class _DeformableEncoderLayer(nn.Module):
    """Deformable self-attention over the image pyramid, then a feed-forward."""

    def __init__(
        self, d_model: int, d_ffn: int, n_levels: int, n_heads: int, n_points: int
    ) -> None:
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        src = self.norm1(
            src
            + self.self_attn(
                query=src + pos,
                reference_points=reference_points,
                value=src,
                spatial_shapes=spatial_shapes,
                key_padding_mask=key_padding_mask,
            )
        )
        return self.norm2(src + self.linear2(F.relu(self.linear1(src))))


class _Encoder(nn.Module):
    """Six layers of fuse, then text self-attention, then deformable image attention."""

    def __init__(
        self,
        layer: _DeformableEncoderLayer,
        text_layer: _TextLayer,
        fusion_layer: BiAttentionBlock,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.text_layers = _clones(text_layer, num_layers)
        self.fusion_layers = _clones(fusion_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def reference_points(spatial_shapes: Tensor, valid_ratios: Tensor, device) -> Tensor:
        """A normalised grid of cell centres per level, scaled by each image's valid fraction."""
        points = []
        for level, (height, width) in enumerate(spatial_shapes):
            height, width = int(height), int(width)
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            points.append(torch.stack((ref_x, ref_y), -1))
        stacked = torch.cat(points, 1)
        return stacked[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        memory_text: Tensor,
        text_attention_mask: Tensor,
        position_ids: Tensor,
        text_self_attention_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        output = src
        reference = self.reference_points(spatial_shapes, valid_ratios, src.device)
        pos_text = token_position(position_ids[..., None], num_pos_feats=256)

        for index in range(self.num_layers):
            output, memory_text = self.fusion_layers[index](
                image=output,
                text=memory_text,
                attention_mask_v=key_padding_mask,
                attention_mask_l=text_attention_mask,
            )
            memory_text = self.text_layers[index](
                src=memory_text.transpose(0, 1),
                src_mask=~text_self_attention_masks,
                pos=pos_text.transpose(0, 1),
            ).transpose(0, 1)
            output = self.layers[index](
                src=output,
                pos=pos,
                reference_points=reference,
                spatial_shapes=spatial_shapes,
                key_padding_mask=key_padding_mask,
            )
        return output, memory_text


class _DecoderLayer(nn.Module):
    """Query self-attention, text cross-attention, then deformable image cross-attention."""

    def __init__(
        self, d_model: int, d_ffn: int, n_levels: int, n_heads: int, n_points: int
    ) -> None:
        super().__init__()
        self.cross_attn = MultiScaleDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.ca_text = nn.MultiheadAttention(d_model, n_heads)
        self.catext_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        tgt_query_pos: Tensor,
        tgt_reference_points: Tensor,
        memory_text: Tensor,
        text_attention_mask: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        memory_spatial_shapes: Tensor,
    ) -> Tensor:
        q = k = tgt + tgt_query_pos
        tgt = self.norm2(tgt + self.self_attn(q, k, tgt)[0])

        tgt = self.catext_norm(
            tgt
            + self.ca_text(
                tgt + tgt_query_pos,
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
        )

        tgt = self.norm1(
            tgt
            + self.cross_attn(
                query=(tgt + tgt_query_pos).transpose(0, 1),
                reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
                value=memory.transpose(0, 1),
                spatial_shapes=memory_spatial_shapes,
                key_padding_mask=memory_key_padding_mask,
            ).transpose(0, 1)
        )
        return self.norm3(tgt + self.linear2(F.relu(self.linear1(tgt))))


class _Decoder(nn.Module):
    """Six decoder layers, refining a box after each."""

    def __init__(self, layer: _DecoderLayer, num_layers: int, norm: nn.Module, d_model: int):
        super().__init__()
        self.layers = _clones(layer, num_layers)
        self.norm = norm
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.bbox_embed: nn.ModuleList | None = None

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        refpoints_unsigmoid: Tensor,
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        memory_text: Tensor,
        text_attention_mask: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        output = tgt
        reference_points = refpoints_unsigmoid.sigmoid()
        intermediate = []
        ref_points = [reference_points]

        for index, layer in enumerate(self.layers):
            reference_input = (
                reference_points[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )
            sine = query_sine_embed(reference_input[:, :, 0, :])
            query_pos = self.ref_point_head(sine)

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_reference_points=reference_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_spatial_shapes=spatial_shapes,
            )

            delta = self.bbox_embed[index](output)
            new_reference = (delta + inverse_sigmoid(reference_points)).sigmoid()
            reference_points = new_reference.detach()
            ref_points.append(new_reference)
            intermediate.append(self.norm(output))

        return (
            [item.transpose(0, 1) for item in intermediate],
            [item.transpose(0, 1) for item in ref_points],
        )


class Transformer(nn.Module):
    """Encoder, query selection, decoder.

    Args:
        d_model: Model width.
        nhead: Attention heads. The text and fusion paths use half this, which is upstream's
            choice and not a derived quantity.
        num_queries: Object queries.
        num_encoder_layers: Encoder depth.
        num_decoder_layers: Decoder depth.
        dim_feedforward: Image feed-forward width. Text and fusion use half.
        num_feature_levels: Pyramid levels.
        enc_n_points: Deformable sampling points per head per level, encoder.
        dec_n_points: The same, decoder.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_queries: int = 900,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        num_feature_levels: int = 4,
        enc_n_points: int = 4,
        dec_n_points: int = 4,
        max_text_len: int = 256,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries

        self.encoder = _Encoder(
            layer=_DeformableEncoderLayer(
                d_model, dim_feedforward, num_feature_levels, nhead, enc_n_points
            ),
            text_layer=_TextLayer(d_model, nhead // 2, dim_feedforward // 2),
            fusion_layer=BiAttentionBlock(d_model, d_model, dim_feedforward // 2, nhead // 2),
            num_layers=num_encoder_layers,
        )
        self.decoder = _Decoder(
            layer=_DecoderLayer(
                d_model, dim_feedforward, num_feature_levels, nhead, dec_n_points
            ),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
            d_model=d_model,
        )

        self.level_embed = nn.Parameter(torch.zeros(num_feature_levels, d_model))
        self.tgt_embed = nn.Embedding(num_queries, d_model)
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        self.enc_out_bbox_embed: MLP | None = None
        self.enc_out_class_embed: ContrastiveEmbed | None = None

    @staticmethod
    def valid_ratio(mask: Tensor) -> Tensor:
        """What fraction of each axis is real image rather than padding."""
        _, height, width = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1).float() / height
        valid_w = torch.sum(~mask[:, 0, :], 1).float() / width
        return torch.stack([valid_w, valid_h], -1)

    def forward(
        self,
        srcs: list[Tensor],
        masks: list[Tensor],
        pos_embeds: list[Tensor],
        encoded_text: Tensor,
        text_token_mask: Tensor,
        position_ids: Tensor,
        text_self_attention_masks: Tensor,
    ) -> tuple[list[Tensor], list[Tensor], Tensor]:
        """Return ``(per-layer hidden states, per-layer reference boxes, fused text)``."""
        src_flatten, mask_flatten, pos_flatten, spatial_shapes = [], [], [], []
        for level, (src, mask, pos) in enumerate(zip(srcs, masks, pos_embeds)):
            batch, channels, height, width = src.shape
            spatial_shapes.append((height, width))
            src_flatten.append(src.flatten(2).transpose(1, 2))
            mask_flatten.append(mask.flatten(1))
            pos_flatten.append(
                pos.flatten(2).transpose(1, 2) + self.level_embed[level].view(1, 1, -1)
            )

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        pos_flatten = torch.cat(pos_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        valid_ratios = torch.stack([self.valid_ratio(m) for m in masks], 1)

        memory, memory_text = self.encoder(
            src_flatten,
            pos=pos_flatten,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=encoded_text,
            text_attention_mask=~text_token_mask,
            position_ids=position_ids,
            text_self_attention_masks=text_self_attention_masks,
        )

        # Language-guided query selection: score every image token against the prompt, keep the
        # best `num_queries`, and use their proposals as the decoder's starting boxes.
        output_memory, output_proposals = encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        scores = self.enc_out_class_embed(output_memory, memory_text, text_token_mask)
        top = torch.topk(scores.max(-1)[0], self.num_queries, dim=1)[1]

        coords = self.enc_out_bbox_embed(output_memory) + output_proposals
        refpoint = torch.gather(coords, 1, top.unsqueeze(-1).repeat(1, 1, 4)).detach()
        # `embed_init_tgt` is True in both published configs, so the queries are the learned
        # embedding rather than the gathered memory. The gather still happens upstream and its
        # result is discarded; it is not computed here.
        tgt = self.tgt_embed.weight[:, None, :].repeat(1, batch, 1).transpose(0, 1)

        hidden, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            refpoints_unsigmoid=refpoint.transpose(0, 1),
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        return hidden, references, memory_text
