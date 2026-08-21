# SPDX-License-Identifier: Apache-2.0
"""The image-mode EdgeTAM network: an image encoder, a prompt encoder and a mask decoder.

Upstream's ``SAM2Base`` carries the video tracker as well, and EdgeTAM adds its own to it -- a
memory attention stack, a memory encoder, object pointers, the 2-D spatial perceiver that is the
paper's contribution, and the frame bookkeeping that drives all of them. None of it runs when you
segment a single image, and it is most of the class. So this module builds the three sub-networks
the image path actually uses and wires them together. See ``PROVENANCE.md`` for what that drops.

The split into :meth:`EdgeTam.encode` and :meth:`EdgeTam.decode` is the seam mozo builds on. The
encoder is the expensive half and depends only on the image; the decoder is cheap and depends on
the prompt. Keeping them apart is what lets one encode serve many prompts, and it is also what
would let each half be exported to a graph runtime on its own.
"""

from __future__ import annotations

import torch
from torch import nn

from .backbones.image_encoder import FpnNeck, ImageEncoder
from .backbones.repvit import RepViT
from .config import SETTINGS, STABILITY
from .position_encoding import PositionEmbeddingSine
from .sam.mask_decoder import MaskDecoder
from .sam.prompt_encoder import PromptEncoder
from .sam.transformer import TwoWayTransformer

__all__ = ["EdgeTam"]


class EdgeTam(nn.Module):
    """EdgeTAM restricted to single images.

    Attributes:
        image_size: Square side the encoder runs at. Prompts are scaled into this space.
    """

    def __init__(self) -> None:
        super().__init__()
        self.image_size = SETTINGS["image_size"]
        self.num_feature_levels = SETTINGS["num_feature_levels"]
        hidden = SETTINGS["hidden_dim"]

        self.image_encoder = ImageEncoder(
            trunk=RepViT(),
            neck=FpnNeck(
                position_encoding=PositionEmbeddingSine(
                    num_pos_feats=hidden, normalize=True, scale=None, temperature=10000
                ),
                d_model=hidden,
                backbone_channel_list=list(SETTINGS["backbone_channel_list"]),
                fpn_top_down_levels=list(SETTINGS["fpn_top_down_levels"]),
                fpn_interp_model=SETTINGS["fpn_interp_model"],
            ),
            scalp=SETTINGS["scalp"],
        )

        # Built with the arguments upstream's ``_build_sam_heads`` passes, plus the stability
        # settings ``build_sam2`` appends. The flags that look like tracker concerns -- object
        # scores, the multimask output token -- still decide how many tokens the decoder
        # allocates, so they change the shape of the weights and have to match the checkpoint
        # even though nothing here reads their outputs.
        embedding = self.image_size // SETTINGS["backbone_stride"]
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=hidden,
            image_embedding_size=(embedding, embedding),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(depth=2, embedding_dim=hidden, mlp_dim=2048, num_heads=8),
            transformer_dim=hidden,
            iou_head_depth=3,
            iou_head_hidden_dim=hidden,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            **STABILITY,
        )

        # Trained as "there is no memory to attend to", which is exactly the situation a single
        # image is in -- so despite belonging to the tracker by name, this is added on the image
        # path as well. Upstream's config says so directly: ``directly_add_no_mem_embed: true``,
        # and its own image predictor adds it in ``set_image`` before caching the features.
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, hidden))

        # The decoder's positional encoding of the image grid is constant once the checkpoint is
        # loaded, and rebuilding it costs 1.4 ms of every click -- a cumsum, a 4096x2 @ 2x128
        # matmul, a sine and a cosine, for a tensor that never changes. Held as a non-persistent
        # buffer so it moves with ``.to(device)`` and stays out of the state dict, and filled on
        # first use rather than here: ``get_dense_pe`` reads a checkpoint buffer, and this module
        # is constructed before the weights are loaded and before a device is chosen.
        self.register_buffer("_dense_pe", None, persistent=False)

    @torch.no_grad()
    def encode(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the expensive half: an image in, the features every prompt will reuse out.

        Args:
            batch: ``(1, 3, image_size, image_size)`` normalised float tensor.

        Returns:
            ``image_embed`` at stride 16 and the two ``high_res_feats`` the decoder skips into.
            This dict is the whole of what a cache needs to hold; nothing else survives the call.
        """
        out = self.image_encoder(batch)
        # Upstream projects these two inside ``forward_image`` rather than in the decoder, so
        # that a second click on the same image does not repeat them. Same reason they are
        # cached here.
        levels = out["backbone_fpn"]
        levels[0] = self.sam_mask_decoder.conv_s0(levels[0])
        levels[1] = self.sam_mask_decoder.conv_s1(levels[1])
        maps = levels[-self.num_feature_levels :]
        # Upstream adds this in the flattened HWxNxC layout, where it broadcasts over positions;
        # here the maps stay NCHW, so the same bias is applied per channel instead.
        embed = maps[-1] + self.no_mem_embed.view(1, -1, 1, 1)
        return {"image_embed": embed, "high_res_feats": maps[:-1]}

    @torch.no_grad()
    def decode(
        self,
        features: dict[str, torch.Tensor],
        point_coords: torch.Tensor | None,
        point_labels: torch.Tensor | None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the cheap half: cached features plus a prompt in, low-resolution masks out.

        Args:
            features: What :meth:`encode` returned for this image.
            point_coords: ``(B, N, 2)`` x, y already scaled into ``image_size`` space.
            point_labels: ``(B, N)``. 1 positive, 0 negative, 2 and 3 the corners of a box.
            mask_input: ``(B, 1, 256, 256)`` logits from an earlier call, to refine. One channel,
                not the three a multimask call returns -- pick the candidate first.
            multimask_output: Return three candidate masks rather than one.

        Returns:
            ``(B, C, 256, 256)`` mask logits and their ``(B, C)`` predicted IoU.
        """
        if self._dense_pe is None:
            self._dense_pe = self.sam_prompt_encoder.get_dense_pe()
        points = None if point_coords is None else (point_coords, point_labels)
        sparse, dense = self.sam_prompt_encoder(points=points, boxes=None, masks=mask_input)
        # One encode serves a batch of prompts, so the single image embedding is repeated across
        # them inside the decoder rather than being encoded once per prompt. The count is read
        # off the prompt encoder's own output, because a mask-only prompt carries a batch too and
        # deriving it from the points alone would miss that.
        batched = sparse.shape[0] != features["image_embed"].shape[0]
        low_res, iou, _, _ = self.sam_mask_decoder(
            image_embeddings=features["image_embed"],
            image_pe=self._dense_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
            repeat_image=batched,
            high_res_features=list(features["high_res_feats"]),
        )
        return low_res, iou
