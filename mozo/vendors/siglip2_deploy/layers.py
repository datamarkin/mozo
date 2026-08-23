# SPDX-License-Identifier: Apache-2.0
"""The transformer block both towers are built from.

Ordinary pre-norm blocks, with two details that are not ordinary.

**The activation is ``gelu_pytorch_tanh``.** That is ``F.gelu(x, approximate="tanh")``, the tanh
approximation -- not the exact erf GELU that ``nn.GELU()`` gives by default, and not CLIP's
``QuickGELU`` either. Three plausible activations, one right answer, and substituting either of the
others neither raises nor warns.

**The LayerNorm epsilon is 1e-6.** torch defaults to 1e-5. See ``config.LAYER_NORM_EPS``.

The attention reproduces upstream's ``eager_attention_forward`` rather than
``scaled_dot_product_attention``. Both are available upstream through ``ALL_ATTENTION_FUNCTIONS``
and they are the same arithmetic in a different order; the gate runs the reference with
``attn_implementation="eager"`` so that what is compared is one path against itself.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import LAYER_NORM_EPS

__all__ = ["Attention", "Encoder", "EncoderLayer", "MLP"]


class Attention(nn.Module):
    """Multi-head self-attention, unfused and unmasked.

    Unmasked is not an omission. SigLIP's text tower is bidirectional -- upstream flags it in a
    comment because a reader arriving from CLIP assumes otherwise -- and its processor returns no
    attention mask, so the padding is attended along with everything else. There is no mask to
    carry, in either tower.
    """

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        #: Applied to the *product*, after the matmul, which is where upstream applies it.
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = (*hidden.shape[:-1], -1, self.head_dim)
        queries = self.q_proj(hidden).view(shape).transpose(1, 2)
        keys = self.k_proj(hidden).view(shape).transpose(1, 2)
        values = self.v_proj(hidden).view(shape).transpose(1, 2)

        weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attended = torch.matmul(weights, values).transpose(1, 2).contiguous()
        return self.out_proj(attended.reshape(*hidden.shape[:-1], -1))


class MLP(nn.Module):
    """Two linears around the tanh-approximated GELU."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden), approximate="tanh"))


class EncoderLayer(nn.Module):
    """One pre-norm block: normalise, attend, add; normalise, project, add."""

    def __init__(self, width: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(width, eps=LAYER_NORM_EPS)
        self.self_attn = Attention(width, heads)
        self.layer_norm2 = nn.LayerNorm(width, eps=LAYER_NORM_EPS)
        self.mlp = MLP(width, hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.layer_norm1(hidden))
        return hidden + self.mlp(self.layer_norm2(hidden))


class Encoder(nn.Module):
    """A stack of blocks. Named ``layers`` because the checkpoint's keys say so."""

    def __init__(self, width: int, layers: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(width, heads, hidden) for _ in range(layers))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden
