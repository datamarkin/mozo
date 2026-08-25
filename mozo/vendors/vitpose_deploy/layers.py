# SPDX-License-Identifier: Apache-2.0
"""The transformer block, and the mixture-of-experts MLP that makes it ViTPose++.

A plain ViT block with one substitution: the MLP writes part of its output from a per-dataset
expert instead of from a single shared projection. Everything else -- pre-norm attention, two
residuals, GELU -- is the ordinary arrangement.

Module names here are upstream's, not chosen. The published checkpoints follow
``transformers``' naming, and matching it exactly is what lets :class:`~.network.VitPose` load one
with ``strict=True`` and no translation table. A rename that reads better here would buy nothing
and cost the load.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import Spec

__all__ = ["Attention", "Layer", "MoeMLP", "SelfAttention", "SelfOutput"]


class SelfAttention(nn.Module):
    """Multi-head self-attention over the patch grid. No mask: every patch sees every patch."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.heads = spec.heads
        self.head_dim = spec.hidden // spec.heads
        self.query = nn.Linear(spec.hidden, spec.hidden)
        self.key = nn.Linear(spec.hidden, spec.hidden)
        self.value = nn.Linear(spec.hidden, spec.hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = hidden.shape
        shape = (batch, tokens, self.heads, self.head_dim)
        query = self.query(hidden).view(shape).transpose(1, 2)
        key = self.key(hidden).view(shape).transpose(1, 2)
        value = self.value(hidden).view(shape).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        return attended.transpose(1, 2).reshape(batch, tokens, -1)


class SelfOutput(nn.Module):
    """The projection after attention. The residual is added in :class:`Layer`, as upstream does."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.dense = nn.Linear(spec.hidden, spec.hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dense(hidden)


class Attention(nn.Module):
    """Attention and its output projection, under the two names the checkpoint uses."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.attention = SelfAttention(spec)
        self.output = SelfOutput(spec)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(hidden))


class MoeMLP(nn.Module):
    """The MLP whose last ``part_features`` columns come from a per-dataset expert.

    ``fc2`` produces ``hidden - part_features`` columns that every dataset shares; one of
    ``experts`` linear layers produces the remaining ``part_features``, and the two are
    concatenated. That is the whole of ViTPose++'s architectural change over ViTPose.

    Upstream selects the expert by running **all** of them and multiplying by a 0/1 mask, which is
    what a training-time implementation looks like when the batch mixes datasets. Inference here
    always asks for one expert for the whole batch, so this indexes it instead: same arithmetic,
    ``experts`` times less of it. The mask form is kept in the equivalence test rather than in the
    forward pass -- see ``tests/families/test_vitpose.py``.
    """

    def __init__(self, spec: Spec):
        super().__init__()
        self.part_features = spec.part_features
        intermediate = spec.hidden * spec.mlp_ratio
        self.fc1 = nn.Linear(spec.hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, spec.hidden - spec.part_features)
        self.experts = nn.ModuleList(
            nn.Linear(intermediate, spec.part_features) for _ in range(spec.experts)
        )

    def forward(self, hidden: torch.Tensor, expert: int) -> torch.Tensor:
        hidden = F.gelu(self.fc1(hidden))
        return torch.cat([self.fc2(hidden), self.experts[expert](hidden)], dim=-1)


class Layer(nn.Module):
    """One pre-norm block: norm, attend, add; norm, MLP, add."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.attention = Attention(spec)
        self.mlp = MoeMLP(spec)
        self.layernorm_before = nn.LayerNorm(spec.hidden, eps=spec.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(spec.hidden, eps=spec.layer_norm_eps)

    def forward(self, hidden: torch.Tensor, expert: int) -> torch.Tensor:
        hidden = self.attention(self.layernorm_before(hidden)) + hidden
        return self.mlp(self.layernorm_after(hidden), expert) + hidden
