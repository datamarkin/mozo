# SPDX-License-Identifier: Apache-2.0
"""The concept head: an encoded image and an encoded prompt in, instances out.

Derived from ``transformers/models/sam3/modeling_sam3.py`` (Apache-2.0).

This is the stage that answers *"where is every cow"*. It takes what the two encoders produced --
neither of which knows about the other -- and runs the five modules that bring them together:

1. :class:`~.geometry.GeometryEncoder` turns exemplar boxes into prompt tokens, and emits its CLS
   token even when there are none, so this runs on text-only prompts too.
2. :class:`~.fusion.FusionEncoder` conditions the image tokens on the prompt.
3. :class:`~.decoder.Decoder` refines 200 object queries into boxes.
4. :class:`~.scoring.DotProductScoring` scores each query against the prompt.
5. :class:`~.mask.MaskHead` turns the surviving queries into masks.

The decoder reports, per layer, the box each layer *started* from -- the convention upstream reads
its own decoder by -- plus the last layer's refined box, which is the actual prediction.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import (
    DECODER,
    FUSION,
    GEOMETRY,
    MASK,
    SCORING,
    BlockSpec,
    DecoderSpec,
    GeometrySpec,
    MaskSpec,
    ScoringSpec,
)
from .decoder import Decoder
from .fusion import FusionEncoder
from .geometry import GeometryEncoder
from .mask import MaskHead
from .scoring import DotProductScoring

__all__ = ["ConceptHead"]


class ConceptHead(nn.Module):
    """Text and exemplar-box prompting, end to end.

    Args:
        geometry: Geometry-encoder geometry.
        fusion: Fusion-encoder geometry.
        decoder: DETR decoder geometry.
        scoring: Scoring-head geometry.
        mask: Mask-head geometry.
    """

    def __init__(
        self,
        geometry: GeometrySpec = GEOMETRY,
        fusion: BlockSpec = FUSION,
        decoder: DecoderSpec = DECODER,
        scoring: ScoringSpec = SCORING,
        mask: MaskSpec = MASK,
    ):
        super().__init__()
        self.geometry = GeometryEncoder(geometry)
        self.fusion = FusionEncoder(fusion)
        self.decoder = Decoder(decoder)
        self.scoring = DotProductScoring(scoring)
        self.mask = MaskHead(mask)

    @torch.no_grad()
    def forward(
        self,
        levels: list[Tensor],
        positions: Tensor,
        text: Tensor,
        text_padding: Tensor,
        boxes: Tensor | None = None,
        box_labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Find every instance of the prompt.

        Args:
            levels: The concept pyramid, finest first, coarsest last.
            positions: The coarsest level's position encoding -- the only one anything reads.
            text: ``(T, B, hidden)`` sequence-first text features.
            text_padding: ``(B, T)`` True where the slot is padding.
            boxes: ``(B, N, 4)`` exemplar boxes as normalised cxcywh, or ``None`` for none.
            box_labels: ``(B, N)`` 1 positive, 0 negative. Required with ``boxes``.

        Returns:
            ``masks`` ``(B, Q, h, w)`` logits, ``boxes`` ``(B, Q, 4)`` as normalised cxcywh,
            ``logits`` ``(B, Q)``, ``presence`` ``(B, 1)``, and ``semantic`` ``(B, 1, h, w)`` --
            a whole-image foreground map the head produces alongside the per-query masks.
            Turning any of it into instances is :func:`~..predictor.instances`' job.

        Raises:
            ValueError: If ``boxes`` is given without ``box_labels``.
        """
        coarse, coarse_positions = levels[-1], positions
        batch = coarse.shape[0]

        if boxes is None:
            boxes = coarse.new_zeros(batch, 0, 4)
            box_labels = torch.zeros(batch, 0, dtype=torch.long, device=coarse.device)
        elif box_labels is None:
            # Guessing between a positive and a negative exemplar returns a confident answer to
            # the wrong question, so it raises rather than defaulting.
            raise ValueError("exemplar boxes need box_labels: 1 for positive, 0 for negative")

        geometry, geometry_padding = self.geometry(
            boxes, box_labels, coarse, coarse_positions
        )
        prompt = torch.cat([text, geometry], dim=0)
        prompt_padding = torch.cat([text_padding, geometry_padding], dim=1)

        # The fusion encoder already flattened the position encoding for its own use, and the
        # decoder needs exactly that tensor -- so it is passed along rather than rebuilt.
        fused, flat_positions = self.fusion(coarse, coarse_positions, prompt, prompt_padding)

        height, width = coarse.shape[-2:]
        decoded = self.decoder(fused, flat_positions, prompt, prompt_padding, height, width)

        logits = self.scoring(decoded["queries"], prompt, prompt_padding)
        predicted = self.mask(decoded["queries"][-1], levels, fused, prompt, prompt_padding)

        return {
            "masks": predicted["masks"],
            "semantic": predicted["semantic"],
            "boxes": decoded["final"],
            "logits": logits[-1].squeeze(-1),
            "presence": decoded["presence"][-1],
        }
