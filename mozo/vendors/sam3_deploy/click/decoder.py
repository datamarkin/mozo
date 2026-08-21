# SPDX-License-Identifier: Apache-2.0
"""The two-way transformer and the mask decoder that turn prompt tokens into masks.

Derived from ``transformers/models/sam3_tracker`` (Apache-2.0). See :mod:`.layers` for why this
package derives from there rather than from another vendor's copy of the same architecture.

The transformer is "two-way" because attention runs in both directions each block: the prompt
tokens attend to the image, and then the image attends back to the tokens. Both directions are
kept, which is why ``keys`` is threaded through and returned rather than discarded -- the
upscaling path reads it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import CLICK, ClickSpec
from ..grounding.layers import Mlp
from .layers import Attention, LayerNorm2d

__all__ = ["MaskDecoder"]


class TwoWayAttentionBlock(nn.Module):
    """Self-attention over the tokens, then cross-attention each way, with an MLP between.

    Args:
        spec: Click geometry.
        skip_first_layer_pe: On the first block the tokens *are* the position encoding, so
            adding it to itself would double it. Only the first block skips.
    """

    def __init__(self, spec: ClickSpec = CLICK, skip_first_layer_pe: bool = False) -> None:
        super().__init__()
        self.self_attn = Attention(spec.hidden, spec.heads, downsample=1)
        self.layer_norm1 = nn.LayerNorm(spec.hidden)
        self.cross_attn_token_to_image = Attention(spec.hidden, spec.heads, spec.downsample)
        self.layer_norm2 = nn.LayerNorm(spec.hidden)
        # Two linears, which is what the checkpoint has. Upstream writes the count as
        # ``num_hidden_layers`` and the two numbers happen to coincide at 2; spelling the sizes
        # out keeps a change to the block count from silently reshaping the feed-forward.
        self.mlp = Mlp((spec.hidden, spec.intermediate, spec.hidden))
        self.layer_norm3 = nn.LayerNorm(spec.hidden)
        self.layer_norm4 = nn.LayerNorm(spec.hidden)
        self.cross_attn_image_to_token = Attention(spec.hidden, spec.heads, spec.downsample)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q, q, queries)
        queries = self.layer_norm1(queries)

        # ``keys`` is unchanged until the last line, so its position encoding is added once and
        # reused; ``queries`` changes under the MLP and has to be re-added.
        k = keys + key_pe
        queries = queries + self.cross_attn_token_to_image(queries + query_pe, k, keys)
        queries = self.layer_norm2(queries)

        queries = queries + self.mlp(queries)
        queries = self.layer_norm3(queries)

        keys = keys + self.cross_attn_image_to_token(k, queries + query_pe, queries)
        return queries, self.layer_norm4(keys)


class TwoWayTransformer(nn.Module):
    """``spec.layers`` two-way blocks, then one last pass from the tokens to the image."""

    def __init__(self, spec: ClickSpec = CLICK) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(spec, skip_first_layer_pe=(index == 0))
            for index in range(spec.layers)
        )
        self.final_attn_token_to_image = Attention(spec.hidden, spec.heads, spec.downsample)
        self.layer_norm_final_attn = nn.LayerNorm(spec.hidden)

    def forward(self, tokens: Tensor, image: Tensor, image_pe: Tensor) -> tuple[Tensor, Tensor]:
        # The image arrives as a feature map and attends as a sequence.
        keys = image.flatten(2).permute(0, 2, 1).unsqueeze(1)
        key_pe = image_pe.flatten(2).permute(0, 2, 1).unsqueeze(1)

        queries = tokens
        for layer in self.layers:
            queries, keys = layer(queries, keys, tokens, key_pe)

        queries = queries + self.final_attn_token_to_image(queries + tokens, keys + key_pe, keys)
        return self.layer_norm_final_attn(queries), keys


class MaskDecoder(nn.Module):
    """Prompt tokens and image features in, mask logits and a predicted IoU out.

    Args:
        spec: Click geometry.

    Attributes:
        num_mask_tokens: One single-mask token plus ``spec.multimask_outputs`` candidates.
    """

    def __init__(self, spec: ClickSpec = CLICK) -> None:
        super().__init__()
        self.spec = spec
        self.num_mask_tokens = spec.multimask_outputs + 1

        self.obj_score_token = nn.Embedding(1, spec.hidden)
        self.iou_token = nn.Embedding(1, spec.hidden)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, spec.hidden)

        self.transformer = TwoWayTransformer(spec)

        self.upscale_conv1 = nn.ConvTranspose2d(spec.hidden, spec.hidden // 4, 2, stride=2)
        self.upscale_layer_norm = LayerNorm2d(spec.hidden // 4, eps=spec.layer_norm_eps)
        self.upscale_conv2 = nn.ConvTranspose2d(spec.hidden // 4, spec.hidden // 8, 2, stride=2)

        self.output_hypernetworks_mlps = nn.ModuleList(
            Mlp((spec.hidden,) * spec.iou_head_depth + (spec.hidden // 8,))
            for _ in range(self.num_mask_tokens)
        )
        self.iou_prediction_head = Mlp(
            (spec.hidden,) * spec.iou_head_depth + (self.num_mask_tokens,)
        )
        # Built and loaded but never run: the object-score token rides through attention and is
        # read as a token, while this head that turns it into a logit has no caller here. Kept so
        # a strict load has somewhere to put it, as ``MaskSpec.upsampling_stages`` is.
        self.pred_obj_score_head = Mlp((spec.hidden, spec.hidden, spec.hidden, 1))

        # The two finest click levels arrive at the FPN's width and are narrowed to what the
        # upscaling path adds them to.
        self.conv_s0 = nn.Conv2d(spec.hidden, spec.hidden // 8, kernel_size=1)
        self.conv_s1 = nn.Conv2d(spec.hidden, spec.hidden // 4, kernel_size=1)

    def forward(
        self,
        image: Tensor,
        image_pe: Tensor,
        sparse: Tensor,
        dense: Tensor,
        fine: Tensor,
        middle: Tensor,
        multimask_output: bool,
    ) -> tuple[Tensor, Tensor]:
        """Decode one prompt against one image.

        Returns:
            ``(B, P, C, 4*grid, 4*grid)`` mask logits and their ``(B, P, C)`` predicted IoU,
            where ``C`` is three candidates when ``multimask_output`` is set and one otherwise.
        """
        batch, channels, height, width = image.shape
        prompts = sparse.shape[1]

        heads = torch.cat(
            [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0
        ).repeat(batch, prompts, 1, 1)
        tokens = torch.cat((heads, sparse), dim=2) if sparse.shape[2] else heads

        # Repeated only when there is something to repeat. ``transformers`` calls
        # ``repeat_interleave`` unconditionally, and repeating once is not free: it materialises
        # a contiguous copy, which changes which attention kernel torch picks and moves the
        # result by 8e-06 -- enough to shift mask pixels. Upstream guards the same way.
        image = image + dense
        positions = image_pe
        if prompts > 1:
            image = image.repeat_interleave(prompts, dim=0)
            positions = positions.repeat_interleave(prompts, dim=0)
        tokens, keys = self.transformer(tokens, image, positions)

        iou_token_out = tokens[:, :, 1, :]
        mask_tokens_out = tokens[:, :, 2 : 2 + self.num_mask_tokens, :]

        keys = keys.transpose(2, 3).view(batch * prompts, channels, height, width)
        # Projected here rather than beside the cached pyramid: these are this module's layers,
        # and running them in the caller would put a head inside the encoder's cache. They depend
        # only on the image, so caching them would work and would shrink a click cache entry from
        # 111 MB to 21 MB -- measured at 0.6 ms of a 6 ms MPS decode, so nothing waits on it,
        # and the boundary is worth more than the millisecond.
        fine, middle = self.conv_s0(fine), self.conv_s1(middle)
        if prompts > 1:
            fine = fine.repeat_interleave(prompts, dim=0)
            middle = middle.repeat_interleave(prompts, dim=0)
        upscaled = F.gelu(self.upscale_layer_norm(self.upscale_conv1(keys) + middle))
        upscaled = F.gelu(self.upscale_conv2(upscaled) + fine)

        weights = torch.stack(
            [mlp(mask_tokens_out[:, :, i, :])
             for i, mlp in enumerate(self.output_hypernetworks_mlps)],
            dim=2,
        )
        _, channels, height, width = upscaled.shape
        flat = upscaled.view(batch, prompts, channels, height * width)
        masks = (weights @ flat).view(batch, prompts, -1, height, width)
        # Sigmoid here rather than inside the block: it is the only one of the four that
        # squashes, and a flag on a shared class to say so would be a knob with one setting.
        iou = torch.sigmoid(self.iou_prediction_head(iou_token_out))

        if multimask_output:
            return masks[:, :, 1:], iou[:, :, 1:]
        # Never a bare ``masks[:, :, :1]``: asking for one mask does not mean taking token 0
        # whatever it looks like. ``transformers`` makes that a config flag and SAM 3's tracker
        # sets it; it is spelled out here because it is always on and a flag with one setting
        # reads as a choice. It survived 23 of 24 parity prompts before one image caught it.
        return self._stable(masks, iou)

    def _steadiness(self, logits: Tensor) -> Tensor:
        """How much of the mask survives moving the threshold either way.

        A mask whose area barely changes between ``+delta`` and ``-delta`` is one the model is
        confident about; one that collapses is not.
        """
        logits = logits.flatten(-2)
        delta = self.spec.stability_delta
        above = torch.sum(logits > delta, dim=-1).float()
        below = torch.sum(logits > -delta, dim=-1).float()
        return torch.where(below > 0, above / below, 1.0)

    def _stable(self, masks: Tensor, iou: Tensor) -> tuple[Tensor, Tensor]:
        """Return the single-mask token unless it is unstable, then the best candidate.

        Asking for one mask should not mean taking token 0 whatever it looks like: on an
        ambiguous prompt that token can collapse, and one of the three candidates will be a
        mask the model actually believes in.
        """
        candidates, candidate_iou = masks[:, :, 1:], iou[:, :, 1:]
        best = torch.argmax(candidate_iou, dim=-1, keepdim=True)
        best_masks = torch.take_along_dim(candidates, best[..., None, None], dim=2)
        best_iou = torch.take_along_dim(candidate_iou, best, dim=2)

        single, single_iou = masks[:, :, :1], iou[:, :, :1]
        steady = self._steadiness(single) >= self.spec.stability_thresh
        return (
            torch.where(steady[..., None, None], single, best_masks),
            torch.where(steady, single_iou, best_iou),
        )
