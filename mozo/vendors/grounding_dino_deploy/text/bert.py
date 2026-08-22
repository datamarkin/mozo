# SPDX-License-Identifier: Apache-2.0
"""BERT-base as Grounding DINO uses it: embeddings, twelve encoder layers, nothing else.

Upstream builds ``transformers.BertModel.from_pretrained("bert-base-uncased")`` and then
overwrites every tensor in it from the Grounding DINO checkpoint, which carries its own
fine-tuned copy under ``bert.*``. The download is therefore only ever used for its shape. This
module is that shape, so the checkpoint is the only source of weights and ``transformers`` is not
a dependency at run time.

**The pooler is not built.** It is present in the checkpoint, upstream freezes it, and the
detection path reads only ``last_hidden_state`` -- so it would be weights loaded and never run.
``checkpoint.py`` drops it by prefix.

**The attention mask is three-dimensional.** Grounding DINO does not hand BERT the usual
per-token padding mask; it hands it a full ``(batch, tokens, tokens)`` mask that isolates each
prompt phrase from the others, and position ids that restart at zero within each phrase. That is
the whole of "sub-sentence level text features", and it is why this cannot be an off-the-shelf
encoder call.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["BertEncoder"]

#: HuggingFace's BERT LayerNorm epsilon. Not torch's 1e-5 default.
_LAYER_NORM_EPS = 1e-12


class _SelfAttention(nn.Module):
    """Multi-head self-attention, laid out with the parameter names the checkpoint uses."""

    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.num_attention_heads = heads
        self.attention_head_size = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)

    def _heads(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(shape).permute(0, 2, 1, 3)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        query = self._heads(self.query(hidden))
        key = self._heads(self.key(hidden))
        value = self._heads(self.value(hidden))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        scores = scores + mask
        weights = scores.softmax(dim=-1)

        context = torch.matmul(weights, value).permute(0, 2, 1, 3).contiguous()
        return context.view(context.shape[:-2] + (self.num_attention_heads * self.attention_head_size,))


class _Projection(nn.Module):
    """Project, add the residual, normalise.

    BERT does this twice per layer -- after attention and after the feed-forward -- differing only
    in the projection's input width. One class serves both: the checkpoint keys come from the
    attribute names that hold it (``attention.output.*`` and ``output.*``), not from the class.
    """

    def __init__(self, in_features: int, hidden: int) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=_LAYER_NORM_EPS)

    def forward(self, hidden: Tensor, residual: Tensor) -> Tensor:
        return self.LayerNorm(self.dense(hidden) + residual)


class _Attention(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.self = _SelfAttention(hidden, heads)
        self.output = _Projection(hidden, hidden)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        return self.output(self.self(hidden, mask), hidden)


class _Intermediate(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden, intermediate)

    def forward(self, hidden: Tensor) -> Tensor:
        # Exact erf gelu, which is what BERT's ``"gelu"`` names. The tanh approximation is a
        # different function and a different number; it is what ``"gelu_new"`` selects.
        return nn.functional.gelu(self.dense(hidden))


class _Layer(nn.Module):
    def __init__(self, hidden: int, heads: int, intermediate: int) -> None:
        super().__init__()
        self.attention = _Attention(hidden, heads)
        self.intermediate = _Intermediate(hidden, intermediate)
        self.output = _Projection(intermediate, hidden)

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        attended = self.attention(hidden, mask)
        return self.output(self.intermediate(attended), attended)


class _Embeddings(nn.Module):
    """Word, position and token-type embeddings, summed and normalised."""

    def __init__(self, vocab: int, hidden: int, max_positions: int, type_vocab: int) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab, hidden)
        self.position_embeddings = nn.Embedding(max_positions, hidden)
        self.token_type_embeddings = nn.Embedding(type_vocab, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=_LAYER_NORM_EPS)

    def forward(self, input_ids: Tensor, token_type_ids: Tensor, position_ids: Tensor) -> Tensor:
        embedded = (
            self.word_embeddings(input_ids)
            + self.token_type_embeddings(token_type_ids)
            + self.position_embeddings(position_ids)
        )
        return self.LayerNorm(embedded)


class _EncoderStack(nn.Module):
    def __init__(self, layers: int, hidden: int, heads: int, intermediate: int) -> None:
        super().__init__()
        self.layer = nn.ModuleList(_Layer(hidden, heads, intermediate) for _ in range(layers))

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layer:
            hidden = layer(hidden, mask)
        return hidden


class BertEncoder(nn.Module):
    """BERT-base, returning ``last_hidden_state`` and nothing else.

    Args:
        vocab: Vocabulary size. 30,522 for `bert-base-uncased`.
        hidden: Model width, 768.
        layers: Encoder layers, 12.
        heads: Attention heads, 12.
        intermediate: Feed-forward width, 3072.
        max_positions: Rows in the position embedding, 512.
        type_vocab: Rows in the token-type embedding, 2.

    Examples:
        >>> encoder = BertEncoder()                                  # doctest: +SKIP
        >>> encoder(ids, types, positions, mask).shape               # doctest: +SKIP
        torch.Size([1, 8, 768])
    """

    def __init__(
        self,
        vocab: int = 30522,
        hidden: int = 768,
        layers: int = 12,
        heads: int = 12,
        intermediate: int = 3072,
        max_positions: int = 512,
        type_vocab: int = 2,
    ) -> None:
        super().__init__()
        self.embeddings = _Embeddings(vocab, hidden, max_positions, type_vocab)
        self.encoder = _EncoderStack(layers, hidden, heads, intermediate)

    @staticmethod
    def extend_mask(mask: Tensor, dtype: torch.dtype) -> Tensor:
        """Turn a boolean keep-mask into the additive mask attention wants.

        Only the ``(batch, tokens, tokens)`` phrase-isolating mask Grounding DINO builds is
        accepted; it is the only shape this model ever produces, and a 2-D padding-mask branch
        would be untested flexibility. Blocked positions become the dtype's most
        negative finite value rather than ``-inf``, which is what HuggingFace does: ``-inf``
        survives the softmax as a NaN when an entire row is blocked, and a finite floor does not.
        """
        if mask.dim() != 3:
            raise ValueError(f"expected a (batch, tokens, tokens) mask, got {mask.dim()} dims")
        extended = mask[:, None, :, :].to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Return ``last_hidden_state``: ``(batch, tokens, hidden)``."""
        embedded = self.embeddings(input_ids, token_type_ids, position_ids)
        mask = self.extend_mask(attention_mask, embedded.dtype)
        return self.encoder(embedded, mask)
