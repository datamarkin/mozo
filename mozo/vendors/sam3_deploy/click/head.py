# SPDX-License-Identifier: Apache-2.0
"""SAM 3's click path: point at something, get that one thing back.

The concept path answers "where is every taxi". This one answers "what is *this*". They are not
alternatives -- one checkpoint and one process serve both -- but they are not one forward pass
either: the two heads preprocess an image differently, so each runs its own trunk pass and reads
its own neck stack. :meth:`~..predictor.Segmenter.encode_click` is where that is measured.

Derived from ``transformers/models/sam3_tracker`` (Apache-2.0), which is SAM 3's own tracker --
the same provenance as the rest of this package. Nothing here comes from another vendor.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import CLICK, ClickSpec
from .decoder import MaskDecoder
from .prompt import PromptEncoder

__all__ = ["ClickHead"]


class ClickHead(nn.Module):
    """The prompt encoder and mask decoder that turn clicks into one instance.

    Args:
        spec: Click geometry.

    Attributes:
        image_size: The square prompts must be scaled into.
        grid: The coarsest click level's side, which the dense position encoding covers.
    """

    def __init__(self, spec: ClickSpec = CLICK) -> None:
        super().__init__()
        self.image_size = spec.image_size
        self.grid = spec.grid

        self.prompt_encoder = PromptEncoder(spec)
        self.mask_decoder = MaskDecoder(spec)

        # Trained as "there is no memory to attend to", which is exactly the situation a single
        # image is in. It belongs to the video tracker by name and is needed here regardless:
        # leaving it out shifts the embedding, and the shift survives into the mask.
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, spec.hidden))

        # A function of the grid alone, so it is the same tensor on every call. Held rather than
        # rebuilt, and filled on first use because it derives from a buffer that is random until
        # the checkpoint arrives. Non-persistent, so it stays out of the state dict.
        self.register_buffer(
            "dense_pe", torch.zeros(1, spec.hidden, self.grid, self.grid), persistent=False
        )
        self._dense_pe_ready = False

    @torch.no_grad()
    def forward(
        self,
        click: list[Tensor],
        points: Tensor | None,
        labels: Tensor | None,
        mask_input: Tensor | None = None,
        multimask_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Turn cached image features plus a prompt into mask logits.

        Args:
            click: The click pyramid from :class:`~..vision.neck.Neck`, coarsest last --
                ``288x288``, ``144x144``, ``72x72``, all at 256 channels.
            points: ``(B, N, 2)`` x, y already scaled into ``image_size`` space by
                :func:`~..image.to_model_coords`.
            labels: ``(B, N)``. 1 include, 0 exclude, 2 and 3 a box's corners.
            mask_input: ``(B, 1, 288, 288)`` logits from an earlier call, to refine.
            multimask_output: Return three candidates rather than one. When off, the decoder
                still consults the three and returns the stable one.

        Returns:
            ``(B, C, 288, 288)`` mask logits and their ``(B, C)`` predicted IoU.
        """
        if not self._dense_pe_ready:
            self.dense_pe.copy_(self.prompt_encoder.dense_positions())
            self._dense_pe_ready = True

        fine, middle, coarse = click
        # Upstream adds this in the flattened HWxNxC layout, where it broadcasts over positions;
        # the maps stay NCHW here, so the same bias is applied per channel instead.
        image = coarse + self.no_mem_embed.view(1, -1, 1, 1)

        sparse, dense = self.prompt_encoder(points, labels, mask_input)
        masks, iou = self.mask_decoder(
            image, self.dense_pe, sparse, dense, fine, middle, multimask_output
        )
        # One image; the prompt axis is what the caller batched over, so it is the one kept.
        return masks[0], iou[0]
