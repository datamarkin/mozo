# SPDX-License-Identifier: Apache-2.0
"""The text tower: token ids in, one pooled vector out.

Every difference from CLIP's text tower is a difference that runs silently if you get it wrong.

**It is bidirectional.** No causal mask. Upstream leaves a comment saying so, because a reader
arriving from CLIP will assume the opposite and a causal mask changes every vector without raising.

**Nothing is masked out, including the padding.** The published processor returns ``input_ids``
only -- ``model_input_names`` has one entry -- so no attention mask exists and every pad token is
attended along with the prompt. Adding the obviously-correct mask is a change, not a fix.

**Pooling takes the last position, whatever is there.** Not the end-of-text marker, the way CLIP
finds its slot by ``argmax`` over token ids; simply ``hidden[:, -1]``. Upstream's own comment
notes it "may be padding", and for any prompt shorter than the context it is. This is coherent
only because every prompt is padded to a fixed 64 -- which is why the padding is not optional and
:class:`~mozo.vendors.siglip2_deploy.text.tokenizer.Tokenizer` never emits a shorter row.

**The head is a real projection.** ``Linear(width, projection)``, applied after pooling. There is
no separate ``text_projection`` matrix -- this is it -- and for ``giant-opt`` it projects *up*,
from a 1152-wide tower into the 1536-wide shared space the image tower defines.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import CONTEXT, LAYER_NORM_EPS, VOCAB, Spec
from ..layers import Encoder

__all__ = ["TextTower"]


class Embeddings(nn.Module):
    """Token lookup plus a learned position per slot."""

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB, spec.text_width)
        self.position_embedding = nn.Embedding(CONTEXT, spec.text_width)
        self.register_buffer(
            "position_ids", torch.arange(CONTEXT).expand((1, -1)), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = self.position_ids[:, : tokens.shape[-1]]
        return self.token_embedding(tokens) + self.position_embedding(positions)


class TextTower(nn.Module):
    """The whole text tower. Takes ``(N, 64)`` token ids, returns ``(N, projection)``."""

    def __init__(self, spec: Spec) -> None:
        super().__init__()
        self.embeddings = Embeddings(spec)
        self.encoder = Encoder(spec.text_width, spec.text_layers, spec.text_heads, spec.text_mlp)
        self.final_layer_norm = nn.LayerNorm(spec.text_width, eps=LAYER_NORM_EPS)
        self.head = nn.Linear(spec.text_width, spec.projection)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.final_layer_norm(self.encoder(self.embeddings(tokens)))
        return self.head(hidden[:, -1, :])
