# SPDX-License-Identifier: Apache-2.0
"""Where the image meets the prompt.

Derived from ``transformers/models/sam3/modeling_sam3.py`` (Apache-2.0).

Six blocks in which the image tokens attend to themselves, then to the prompt. Only the coarsest
surviving pyramid level takes part -- 72x72 at stride 14, 5184 tokens -- because the model is
built with one feature level. The finer levels exist for the mask head, not for fusion.

The prompt arriving here is text *and* geometry: 32 text slots followed by the geometry encoder's
CLS token, 33 in total. Both halves are produced sequence-first; this stage runs batch-first,
which is the layout its weights were trained under. See :mod:`.layers`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import FUSION, BlockSpec
from .layers import FusionLayer

__all__ = ["FusionEncoder"]


class FusionEncoder(nn.Module):
    """Image tokens conditioned on the prompt.

    Args:
        spec: Block widths and depth.
    """

    def __init__(self, spec: BlockSpec = FUSION):
        super().__init__()
        self.layers = nn.ModuleList(
            FusionLayer(spec, batch_first=True) for _ in range(spec.layers)
        )

    @torch.no_grad()
    def forward(
        self, features: Tensor, positions: Tensor, prompt: Tensor, prompt_padding: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Fuse one pyramid level with the prompt.

        Args:
            features: ``(B, hidden, H, W)`` the coarsest surviving level.
            positions: ``(B, hidden, H, W)`` its position encoding.
            prompt: ``(P, B, hidden)`` sequence-first text-plus-geometry tokens.
            prompt_padding: ``(B, P)`` True where the prompt slot is padding.

        Returns:
            ``(B, H*W, hidden)`` fused image tokens and the ``(B, H*W, hidden)`` flattened
            position encoding, which the decoder needs as well.
        """
        flat = features.flatten(2).transpose(1, 2)
        flat_positions = positions.flatten(2).transpose(1, 2)
        memory = prompt.transpose(0, 1)

        hidden = flat
        for layer in self.layers:
            # The prompt carries no position encoding, so key and value are the same tensor.
            hidden = layer(
                hidden, memory, memory, memory_padding=prompt_padding, target_pos=flat_positions
            )
        return hidden, flat_positions
