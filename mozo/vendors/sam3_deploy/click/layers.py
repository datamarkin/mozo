# SPDX-License-Identifier: Apache-2.0
"""The pieces the click head is built out of.

Derived from ``transformers/models/sam3_tracker/modeling_sam3_tracker.py`` (Apache-2.0), which
is SAM 3's own tracker -- the same provenance as the rest of this package, and the reason none
of it comes from :mod:`mozo.vendors.sam2_deploy`. A vendor does not import another vendor: each
one is reproducible against its own upstream, and a shared substrate would let one family's
re-sync move another's masks.

Two divergences from ``transformers`` here, both to keep the runtime dependency surface at torch:

- The attention layer calls :func:`torch.nn.functional.scaled_dot_product_attention` directly
  rather than dispatching through ``ALL_ATTENTION_FUNCTIONS``. That table exists to let a
  caller pick flash or eager at runtime; there is one path here and it is the one the published
  weights were measured against.
- ``ACT2FN[...]`` lookups are replaced by the activation itself. The config only ever names
  ``relu`` for these blocks.

The feed-forward is :class:`~..grounding.layers.Mlp`, which this package already carries and
which the checkpoint's ``layers.N`` naming matches exactly. ``transformers`` names the first and
last projections instead, and following that would have cost six rename rules whose only job was
to undo the choice.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["Attention", "LayerNorm2d"]


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an ``NCHW`` tensor.

    Written out rather than permuting into :class:`torch.nn.LayerNorm`, which is what
    ``transformers`` does. The two are the same normalisation and disagree in the last bits, and
    the disagreement survives the upscaling path into the mask -- 7.6e-06 on a logit, which is
    two ulp and enough to move pixels across the threshold. The published weights were measured
    against this arrangement, so this is the one that reproduces them.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        normalised = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalised + self.bias[:, None, None]


class Attention(nn.Module):
    """Attention that projects down before attending and back up afterwards.

    Args:
        hidden: Width of the tokens arriving and leaving.
        heads: Attention heads.
        downsample: Divisor applied to ``hidden`` inside the attention. The self-attention
            block uses 1; every cross-attention uses 2.
    """

    def __init__(self, hidden: int, heads: int, downsample: int = 1) -> None:
        super().__init__()
        self.internal_dim = hidden // downsample
        self.heads = heads
        self.head_dim = self.internal_dim // heads
        self.q_proj = nn.Linear(hidden, self.internal_dim)
        self.k_proj = nn.Linear(hidden, self.internal_dim)
        self.v_proj = nn.Linear(hidden, self.internal_dim)
        self.o_proj = nn.Linear(self.internal_dim, hidden)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Attend, with ``(batch, prompts, tokens, channels)`` in and the same shape out."""
        batch, prompts = query.shape[:2]
        shape = (batch * prompts, -1, self.heads, self.head_dim)

        q = self.q_proj(query).view(*shape).transpose(1, 2)
        k = self.k_proj(key).view(*shape).transpose(1, 2)
        v = self.v_proj(value).view(*shape).transpose(1, 2)

        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(batch, prompts, -1, self.internal_dim)
        return self.o_proj(attended)
