# SPDX-License-Identifier: Apache-2.0
"""SAM 3's text tower: a CLIP-L text transformer, then a projection down to the fusion width.

Derived from ``transformers/models/sam3`` (Apache-2.0), which reaches CLIP for this half via
``CLIPTextModelWithProjection`` rather than defining it. The 24 blocks below are that model,
written out so nothing here imports ``transformers``.

This is the other expensive half of SAM 3 -- 354M parameters, 41 percent of the checkpoint -- and
it depends only on the prompt. A phrase encoded once stays valid for every image afterwards, which
is what makes annotating ten thousand pictures with one prompt cheap.

Two things are worth knowing before changing anything here.

**Attention is ``nn.MultiheadAttention``.** That is what upstream uses, and it is what the fused
``in_proj_weight`` in the checkpoint is shaped for. Reimplementing it as three projections is
algebraically identical and numerically not -- the same trap the trunk's rotary embedding set.

**The 512-wide ``text_projection`` is not built.** Upstream constructs it, applies it, and throws
the result away: with ``pool_type="none"`` the pooled branch is discarded by the caller, and only
the 1024-wide token sequence survives. Dropping it saves a matmul per prompt and 2MB, and the
parity gate covers the claim -- if it were live, the outputs would not match.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import TEXT, TextSpec
from ..layers import FeedForward

__all__ = ["TextEncoder", "TextTower"]


class Block(nn.Module):
    """One pre-norm residual attention block, in CLIP's layout."""

    def __init__(self, spec: TextSpec):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(spec.width, eps=spec.layer_norm_eps)
        self.attention = nn.MultiheadAttention(spec.width, spec.heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(spec.width, eps=spec.layer_norm_eps)
        self.mlp = FeedForward(spec.width, spec.intermediate)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        normed = self.layer_norm1(x)
        attended, _ = self.attention(
            normed, normed, normed, need_weights=False, attn_mask=attn_mask
        )
        x = x + attended
        return x + self.mlp(self.layer_norm2(x))


class TextTower(nn.Module):
    """Token ids in, contextual token features out. CLIP's text transformer."""

    def __init__(self, spec: TextSpec = TEXT):
        super().__init__()
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.width)
        self.position_embedding = nn.Parameter(torch.zeros(spec.context_length, spec.width))
        self.layers = nn.ModuleList(Block(spec) for _ in range(spec.layers))
        self.final_layer_norm = nn.LayerNorm(spec.width, eps=spec.layer_norm_eps)

        # Causal: each token attends only to itself and what came before. Additive rather than
        # boolean, because that is the form ``nn.MultiheadAttention`` was given upstream, and the
        # two take different code paths inside it. Not persistent -- it is derived, not trained.
        mask = torch.full((spec.context_length, spec.context_length), float("-inf"))
        self.register_buffer("attn_mask", mask.triu_(1), persistent=False)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Embed and contextualise a batch of token ids.

        Args:
            ids: ``(B, L)`` int64, right-padded with zeros.

        Returns:
            The raw ``(B, L, width)`` token embeddings and the contextual ``(B, L, width)``
            features. Both are returned because the grounding stage consumes each separately.
        """
        length = ids.shape[1]
        embedded = self.token_embedding(ids)
        x = embedded + self.position_embedding[:length]
        attn_mask = self.attn_mask[:length, :length]
        for layer in self.layers:
            x = layer(x, attn_mask)
        return embedded, self.final_layer_norm(x)


class TextEncoder(nn.Module):
    """The text tower plus the linear resize into the fusion width.

    Args:
        spec: Text geometry.
    """

    def __init__(self, spec: TextSpec = TEXT):
        super().__init__()
        self.tower = TextTower(spec)
        self.resizer = nn.Linear(spec.width, spec.fusion_width)

    @torch.no_grad()
    def forward(self, ids: Tensor) -> dict[str, Tensor]:
        """Encode tokenized prompts.

        Args:
            ids: ``(B, L)`` from :class:`~.tokenizer.Tokenizer`.

        Returns:
            ``mask`` ``(B, L)`` where **True marks padding** -- inverted relative to the obvious
            reading, because that is the convention PyTorch attention expects; ``features``
            ``(L, B, fusion_width)`` and ``embeddings`` ``(L, B, width)``, both sequence-first
            for the same reason.
        """
        embedded, contextual = self.tower(ids)
        return {
            "mask": ids == 0,
            "features": self.resizer(contextual.transpose(0, 1)),
            "embeddings": embedded.transpose(0, 1),
        }
