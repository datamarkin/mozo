# SPDX-License-Identifier: Apache-2.0
"""The whole model: backbone, neck, text tower, transformer, heads.

    image ──► Swin ──► input_proj ──┐
                                    ├──► encoder ──► query selection ──► decoder ──► logits, boxes
    prompt ─► BERT ──► feat_map ────┘

The two towers meet in the encoder's fusion layers and again in the decoder's text
cross-attention. The output is 900 queries, each carrying a box and a similarity against every
text token -- there is no class head, because there is no class list.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import Spec
from .position import image_position
from .text.bert import BertEncoder
from .transformer import MLP, ContrastiveEmbed, Transformer, inverse_sigmoid
from .vision.swin import SwinTransformer

__all__ = ["GroundingDino", "phrase_masks"]


def phrase_masks(
    input_ids: Tensor, separators: Tensor
) -> tuple[Tensor, Tensor, list[Tensor]]:
    """Build the phrase-isolating attention mask, the per-phrase positions, and the phrase map.

    A caption is several prompts joined by ``.``, and the separators are what tell them apart.
    Between two separators lies one phrase; the mask lets its tokens attend to each other and to
    nothing else, and the position ids restart at zero inside it. Without this a two-prompt
    caption would let ``"a mug"`` condition ``"a person"``, which is not what the weights were
    trained to do.

    Args:
        input_ids: ``(batch, tokens)``.
        separators: Token ids that end a phrase -- ``[CLS]``, ``[SEP]``, ``.`` and ``?``.

    Returns:
        ``(attention_mask, position_ids, per_row_phrase_masks)``. The attention mask is
        ``(batch, tokens, tokens)`` and True where attention is allowed. The phrase map is one
        ``(phrases, tokens)`` boolean tensor per row, which is what lets a detection be traced
        back to the prompt that produced it.
    """
    batch, tokens = input_ids.shape
    device = input_ids.device

    special = torch.zeros((batch, tokens), device=device, dtype=torch.bool)
    for token in separators.tolist():
        special |= input_ids == token

    attention = torch.eye(tokens, device=device, dtype=torch.bool)[None].repeat(batch, 1, 1)
    position_ids = torch.zeros((batch, tokens), device=device, dtype=torch.long)
    phrases: list[list[Tensor]] = [[] for _ in range(batch)]

    for row in range(batch):
        # Restarted per row. Upstream carries `previous_col` across rows of a batch, so a second
        # caption inherits the first one's last separator position; mozo runs one caption per
        # image, where the two are identical. See PROVENANCE.
        previous = 0
        for col in torch.nonzero(special[row]).flatten().tolist():
            if col == 0 or col == tokens - 1:
                attention[row, col, col] = True
                position_ids[row, col] = 0
            else:
                attention[row, previous + 1 : col + 1, previous + 1 : col + 1] = True
                position_ids[row, previous + 1 : col + 1] = torch.arange(
                    0, col - previous, device=device
                )
                belongs = torch.zeros(tokens, device=device, dtype=torch.bool)
                belongs[previous + 1 : col] = True
                phrases[row].append(belongs)
            previous = col

    return attention, position_ids, [torch.stack(row) for row in phrases]


class GroundingDino(nn.Module):
    """One Grounding DINO variant, assembled and ready to run.

    Args:
        spec: The variant's geometry.

    Examples:
        >>> model = GroundingDino(SPECS["tiny"])            # doctest: +SKIP
        >>> logits, boxes = model(image, ids, types, mask)  # doctest: +SKIP
    """

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.spec = spec
        swin = spec.swin

        self.backbone = SwinTransformer(
            embed_dim=swin.embed_dim,
            depths=swin.depths,
            num_heads=swin.num_heads,
            window_size=swin.window_size,
            out_indices=spec.return_levels,
        )
        self.bert = BertEncoder(hidden=spec.text_hidden_dim)
        self.feat_map = nn.Linear(spec.text_hidden_dim, spec.hidden_dim, bias=True)

        # One 1x1 projection per backbone level, then a stride-2 convolution that invents the
        # fourth, coarsest level from the last backbone output.
        channels = spec.backbone_channels
        projections = [
            nn.Sequential(
                nn.Conv2d(channel, spec.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, spec.hidden_dim),
            )
            for channel in channels
        ]
        projections.append(
            nn.Sequential(
                nn.Conv2d(channels[-1], spec.hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, spec.hidden_dim),
            )
        )
        self.input_proj = nn.ModuleList(projections)

        self.transformer = Transformer(
            d_model=spec.hidden_dim,
            nhead=spec.nheads,
            num_queries=spec.num_queries,
            num_encoder_layers=spec.enc_layers,
            num_decoder_layers=spec.dec_layers,
            dim_feedforward=spec.dim_feedforward,
            num_feature_levels=spec.num_feature_levels,
            enc_n_points=spec.enc_n_points,
            dec_n_points=spec.dec_n_points,
            max_text_len=spec.max_text_len,
        )

        # ``dec_pred_bbox_embed_share`` is True, so every decoder layer uses one box head and the
        # checkpoint stores six identical copies of it (verified, not assumed). Referencing the
        # same list from the decoder reproduces that layout.
        self.bbox_embed = nn.ModuleList(
            MLP(spec.hidden_dim, spec.hidden_dim, 4, 3) for _ in range(spec.dec_layers)
        )
        self.class_embed = ContrastiveEmbed(spec.max_text_len)
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # ``two_stage_bbox_embed_share`` is False, so the head that scores query selection is a
        # *deepcopy* with its own trained weights -- not the decoder's. Aliasing the two loads
        # one set of tensors over the other, and because both keys still match, a strict load
        # reports nothing. It cost 12.9 of divergence in the initial reference boxes and moved
        # every prediction. The class head has no parameters, so its own deepcopy is a no-op.
        self.transformer.enc_out_bbox_embed = MLP(spec.hidden_dim, spec.hidden_dim, 4, 3)
        self.transformer.enc_out_class_embed = self.class_embed

    def forward(
        self,
        image: Tensor,
        input_ids: Tensor,
        token_type_ids: Tensor,
        token_mask: Tensor,
        self_attention_mask: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run one image against one caption.

        Args:
            image: ``(batch, 3, height, width)``, already normalised.
            input_ids: ``(batch, tokens)``.
            token_type_ids: ``(batch, tokens)``.
            token_mask: ``(batch, tokens)``, True for real tokens.
            self_attention_mask: ``(batch, tokens, tokens)`` from :func:`phrase_masks`.
            position_ids: ``(batch, tokens)`` from :func:`phrase_masks`.

        Returns:
            ``(logits, boxes)`` from the last decoder layer -- ``(batch, queries, max_text_len)``
            of raw similarities, and ``(batch, queries, 4)`` of ``cxcywh`` normalised to the
            resized image.
        """
        encoded_text = self.feat_map(
            self.bert(input_ids, token_type_ids, position_ids, self_attention_mask)
        )

        features = self.backbone(image)
        # One projection per backbone level, then the coarsest level from a stride-2 convolution
        # over the last of them.
        srcs = [self.input_proj[level](feature) for level, feature in enumerate(features)]
        srcs.append(self.input_proj[-1](features[-1]))

        # No padding is ever added -- one image per call, resized rather than letterboxed -- so
        # every mask is all-False. They are still carried because the encoder's valid-ratio and
        # proposal maths read them, and inventing a padding-free shortcut there would be a
        # second implementation of the same arithmetic. Derived from ``srcs`` rather than built
        # alongside them, so the two cannot fall out of step.
        masks = [
            torch.zeros((src.shape[0], *src.shape[2:]), dtype=torch.bool, device=src.device)
            for src in srcs
        ]
        positions = [
            image_position(
                mask,
                num_pos_feats=self.spec.hidden_dim // 2,
                temperature_h=self.spec.pe_temperature_h,
                temperature_w=self.spec.pe_temperature_w,
            ).to(image.dtype)
            for mask in masks
        ]

        hidden, references, memory_text = self.transformer(
            srcs, masks, positions, encoded_text, token_mask, position_ids, self_attention_mask
        )

        boxes = (self.bbox_embed[-1](hidden[-1]) + inverse_sigmoid(references[-2])).sigmoid()
        logits = self.class_embed(hidden[-1], memory_text, token_mask)
        return logits, boxes
