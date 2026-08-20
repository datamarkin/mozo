# SPDX-License-Identifier: Apache-2.0
"""How confident the model is that a query found the thing you asked for.

Derived from ``transformers/models/sam3/modeling_sam3.py`` (Apache-2.0).

Every object query is scored by projecting it and the pooled prompt into a shared space and
taking their dot product. There is no classifier and no label set: the prompt *is* the class, so
the score is a similarity, which is why one model can answer "cow" and "coffee mug" without
having been told either exists.

Pooling is masked. A prompt is 33 tokens wide but "person" occupies four of them, so averaging
across the padding would drag every score toward the embedding of nothing.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import SCORING, ScoringSpec
from .layers import Mlp

__all__ = ["DotProductScoring"]


class PromptMlp(Mlp):
    """A residual MLP over the prompt, normalised on the way out.

    Named for the checkpoint, which stores the normalisation inside the MLP even though it is
    applied after the residual add.
    """

    def __init__(self, spec: ScoringSpec):
        super().__init__((spec.hidden, spec.intermediate, spec.hidden))
        self.out_norm = nn.LayerNorm(spec.hidden)

    def forward(self, prompt: Tensor) -> Tensor:
        return self.out_norm(super().forward(prompt) + prompt)


class DotProductScoring(nn.Module):
    """Score each object query against the prompt it was asked to find.

    Args:
        spec: Scoring geometry.
    """

    def __init__(self, spec: ScoringSpec = SCORING):
        super().__init__()
        self.limit = spec.limit
        self.prompt_mlp = PromptMlp(spec)
        self.prompt_proj = nn.Linear(spec.hidden, spec.hidden)
        self.hs_proj = nn.Linear(spec.hidden, spec.hidden)
        self.scale = 1.0 / math.sqrt(spec.hidden)

    def forward(self, queries: Tensor, prompt: Tensor, prompt_padding: Tensor) -> Tensor:
        """Score queries against a prompt.

        Args:
            queries: ``(L, B, Q, hidden)`` per-layer query features from the decoder.
            prompt: ``(P, B, hidden)`` sequence-first prompt tokens.
            prompt_padding: ``(B, P)`` True where the slot is padding.

        Returns:
            ``(L, B, Q, 1)`` logits, clamped to +/- ``spec.limit``.
        """
        prompt = self.prompt_mlp(prompt.transpose(0, 1))

        # ``prompt_padding`` marks padding, so the weights are its complement.
        valid = (~prompt_padding).to(prompt.dtype).unsqueeze(-1)
        pooled = (prompt * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        scores = torch.matmul(
            self.hs_proj(queries), self.prompt_proj(pooled).unsqueeze(-1).unsqueeze(0)
        )
        return (scores * self.scale).clamp(-self.limit, self.limit)
