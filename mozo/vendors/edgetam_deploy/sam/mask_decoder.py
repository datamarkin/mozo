# SPDX-License-Identifier: Apache-2.0
"""Prompt tokens and image features in, mask logits and their predicted IoU out.

Taken verbatim from ``sam2/modeling/sam/mask_decoder.py`` in ``facebookresearch/EdgeTAM``, which
is byte-identical to SAM 2's own file. Nothing is dropped: every branch here is reachable from a
single image, including the two that look like tracker concerns.

``pred_obj_scores`` and ``use_multimask_token_for_obj_ptr`` decide how many tokens the decoder
allocates and therefore the shape of the published weights, so they have to match the checkpoint
even though this package reads neither of their outputs.

``dynamic_multimask_via_stability`` is the one setting here that carries no weights at all, which
means a strict load cannot catch it being wrong. It is on for EdgeTAM -- upstream's
``build_sam2`` turns it on through ``apply_postprocessing``, which defaults to true -- and it
changes the answer only when ``multimask_output`` is false. A single-mask prompt is therefore the
only one that can tell whether it is set, which is why the gate runs one.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..layers import MLP, LayerNorm2d

__all__ = ["MaskDecoder"]


class MaskDecoder(nn.Module):
    """The mask head: a two-way transformer, a hypernetwork, and an IoU predictor.

    Args:
        transformer_dim: Channel width of the transformer.
        transformer: The two-way transformer to run tokens and image through.
        num_multimask_outputs: Candidate masks to predict when disambiguating.
        activation: Activation used while upscaling.
        iou_head_depth: Layers in the IoU MLP.
        iou_head_hidden_dim: Hidden width of the IoU MLP.
        use_high_res_features: Take the two finer feature maps as skip connections during
            upscaling, rather than upscaling from the stride-16 embedding alone.
        iou_prediction_use_sigmoid: Squash predicted IoU into ``(0, 1)``.
        dynamic_multimask_via_stability: With ``multimask_output=False``, fall back to the best
            of the three multimask candidates when the single-mask output is unstable.
        dynamic_multimask_stability_delta: Logit band the stability score is measured across.
        dynamic_multimask_stability_thresh: Stability below which the fallback fires.
        pred_obj_scores: Predict whether an object is present at all.
        pred_obj_scores_mlp: Use an MLP rather than a linear layer for that prediction.
        use_multimask_token_for_obj_ptr: Which mask token feeds the object pointer.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[list[Tensor]] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict masks, then select which of the four the caller asked for.

        Args:
            image_embeddings: ``(B, C, H, W)`` from the image encoder.
            image_pe: Positional encoding the shape of *image_embeddings*, batch size 1.
            sparse_prompt_embeddings: ``(B, N, C)`` point and box tokens.
            dense_prompt_embeddings: ``(B, C, H, W)`` mask tokens.
            multimask_output: Return the three disambiguating masks rather than the single one.
            repeat_image: Broadcast one image embedding across a batch of prompts.
            high_res_features: The two finer maps, when ``use_high_res_features`` is set.

        Returns:
            The selected masks, their predicted IoU, the token that would feed an object pointer,
            and the object-presence logits. This package reads the first two.
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        repeat_image: bool,
        high_res_features: Optional[list[Tensor]] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict all four masks, before any selection. See :meth:`forward`."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1, "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: list[Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits: Tensor) -> Tensor:
        """How much of the mask survives raising the threshold by the stability delta."""
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(
        self, all_mask_logits: Tensor, all_iou_scores: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Swap an unstable single mask for the best of the three multimask candidates.

        Fires only when the single-mask output's stability falls below the threshold, and only
        when ``multimask_output`` was false -- so a caller asking for one mask still gets a
        usable one rather than a mask that would vanish under a slightly different threshold.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
