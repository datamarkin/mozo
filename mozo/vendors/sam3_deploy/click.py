# SPDX-License-Identifier: Apache-2.0
"""SAM 3's click path: point at something, get that one thing back.

The concept path answers "where is every taxi". This one answers "what is *this*". They are not
alternatives -- one checkpoint and one process serve both -- but they are not one forward pass
either. :class:`~.vision.neck.Neck` builds two FPN stacks, and this head consumes the second;
what it cannot do is consume the *same* trunk output, because the two heads preprocess the image
differently. :meth:`~.predictor.Segmenter.encode_click` is where that is measured and explained.

**The architecture is SAM 2's, at SAM 3's numbers.** Not "similar to": the checkpoint stores these
weights under ``tracker.sam_prompt_encoder`` and ``tracker.sam_mask_decoder``, with the module
names, tensor shapes and token counts SAM 2 uses, and Meta names the neck that feeds them
``sam2_convs``. So this module builds no new layers. It imports the three that
:mod:`mozo.vendors.sam2_deploy` already carries and configures them for a 1008 pixel square over
a 72x72 grid instead of 1024 over 64x64.

That import is the one place mozo lets two vendor trees touch, and it is deliberate. The
alternative is a second copy of roughly 600 lines that would have to stay bit-identical to the
first forever, which is a worse failure mode than the coupling: a divergence between the copies
would show up as wrong masks, not as an import error.

**One prompt structure, not four modes.** A point is an ``(x, y)`` with a label. A box is two
points with labels 2 and 3 -- ``sam2_deploy/sam/prompt_encoder.py`` says so in the source, at
``num_point_embeddings: int = 4  # pos/neg point + 2 box corners``. A refinement adds a mask
channel. There is no per-combination code path here because the model does not have one.
"""

from __future__ import annotations

import torch
from torch import nn
from torch import Tensor

from ..sam2_deploy.sam.mask_decoder import MaskDecoder
from ..sam2_deploy.sam.prompt_encoder import PromptEncoder
from ..sam2_deploy.sam.transformer import TwoWayTransformer
from .config import SPEC, Spec

__all__ = ["ClickHead"]


class ClickHead(nn.Module):
    """The prompt encoder and mask decoder that turn clicks into one instance.

    Args:
        spec: Geometry, for the square the encoder runs at and the FPN's width.

    Attributes:
        image_size: The square prompts must be scaled into.
        grid: The coarsest click level's side, which is what the dense position encoding covers.
    """

    def __init__(self, spec: Spec = SPEC):
        super().__init__()
        hidden = spec.fpn_hidden
        self.image_size = spec.trunk.image_size
        self.grid = spec.trunk.image_size // spec.trunk.patch

        # Built with the arguments the checkpoint's shapes imply, every one of them checked by
        # loading strict: mask_tokens is (4, 256) so num_multimask_outputs is 3; obj_score_token
        # exists and pred_obj_score_head has three layers, so both object-score flags are on;
        # conv_s0 and conv_s1 exist, so high-res features are in use.
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=hidden,
            image_embedding_size=(self.grid, self.grid),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=hidden, mlp_dim=2048, num_heads=8
            ),
            transformer_dim=hidden,
            iou_head_depth=3,
            iou_head_hidden_dim=hidden,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            # Carries no weights, so the strict load cannot catch it being wrong -- and it only
            # changes an answer when the single-mask token is unstable, which is why it survived
            # 23 of 24 parity prompts before one image caught it. With multimask_output=False the
            # decoder falls back to whichever multimask token is stable rather than returning
            # token 0 blindly.
            dynamic_multimask_via_stability=True,
        )

        # Trained as "there is no memory to attend to", which is exactly the situation a single
        # image is in. SAM 2 needs it on the image path despite the tracker owning it by name --
        # leaving it out there cost 4.5e-02 on the embedding, close enough to look correct -- so
        # it is here for the same reason, and pinned by the same kind of comparison.
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, hidden))

        # The dense position encoding covers a fixed 72x72 grid and is built from a loaded
        # buffer, so it is the same tensor on every call -- but the prompt encoder recomputes it
        # each time, at 0.3 ms and 5.3 MB a click. Held here instead, non-persistent so it stays
        # out of the state dict and the strict load.
        self.register_buffer("dense_pe", torch.zeros(1, hidden, self.grid, self.grid),
                             persistent=False)
        self._dense_pe_ready = False

    @torch.no_grad()
    def forward(
        self,
        click: list[Tensor],
        point_coords: Tensor | None,
        point_labels: Tensor | None,
        mask_input: Tensor | None = None,
        multimask_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Turn cached image features plus a prompt into mask logits.

        Args:
            click: The click pyramid from :class:`~.vision.neck.Neck`, coarsest last --
                ``288x288``, ``144x144``, ``72x72``, all at 256 channels.
            point_coords: ``(B, N, 2)`` x, y already scaled into ``image_size`` space by
                :func:`~.image.to_model_coords`.
            point_labels: ``(B, N)``. 1 positive, 0 negative, 2 and 3 a box's corners.
            mask_input: ``(B, 1, 288, 288)`` logits from an earlier call, to refine.
            multimask_output: Return three candidates rather than one. Worth it for a single
                point, where "this handle", "this door" and "this car" are all defensible. When
                off, the decoder still consults the three and returns the stable one.

        Returns:
            ``(B, C, 288, 288)`` mask logits and their ``(B, C)`` predicted IoU.
        """
        finest, middle, coarse = click

        # Projected per click rather than cached beside the pyramid. They depend only on the
        # image, so caching them in ``encode_click`` would work and would shrink a cache entry
        # from 111 MB to 21 MB; it is not done because it would put a head's layers inside the
        # encoder's cache. Measured at 0.5 ms of a 7.6 ms MPS decode, so nothing is waiting on it.
        high = [self.sam_mask_decoder.conv_s0(finest), self.sam_mask_decoder.conv_s1(middle)]
        # Upstream adds this in the flattened HWxNxC layout, where it broadcasts over positions;
        # the maps stay NCHW here, so the same bias is applied per channel instead.
        embed = coarse + self.no_mem_embed.view(1, -1, 1, 1)

        if not self._dense_pe_ready:
            # Filled on first use rather than in __init__, because the buffer it derives from is
            # random until the checkpoint is loaded.
            self.dense_pe.copy_(self.sam_prompt_encoder.get_dense_pe())
            self._dense_pe_ready = True

        points = None if point_coords is None else (point_coords, point_labels)
        sparse, dense = self.sam_prompt_encoder(points=points, boxes=None, masks=mask_input)
        # One image serves a batch of prompts, so the embedding is repeated inside the decoder
        # rather than encoded per prompt. The count comes off the prompt encoder's own output,
        # because a mask-only prompt carries a batch that the points would not show.
        low_res, iou, _, _ = self.sam_mask_decoder(
            image_embeddings=embed,
            image_pe=self.dense_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=multimask_output,
            repeat_image=sparse.shape[0] != embed.shape[0],
            high_res_features=high,
        )
        return low_res, iou
