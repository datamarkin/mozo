# SPDX-License-Identifier: Apache-2.0
"""CLIP's text tower: a causal transformer pooled at the end-of-text marker.

The cheap half, and the one an archive pipeline runs constantly. A phrase encoded once stays valid
against every image vector ever stored, which is the whole economics of embedding search.

**Pooling is by ``argmax`` over the token ids, not by position.** The end-of-text marker is the
highest id in the vocabulary, so the slot holding it is the slot with the largest id -- which is
how CLIP finds the end of a variable-length prompt without being told its length. It survives
zero padding, because 0 is the lowest id. It would *not* survive a prompt containing a literal
``<|endoftext|>``, which would pool at that one instead; upstream behaves the same way and this
package keeps it.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..layers import LayerNorm, Transformer

__all__ = ["TextTransformer", "causal_mask"]


def causal_mask(context_length: int) -> Tensor:
    """An additive mask that stops a token attending to anything after it.

    Upper triangle above the diagonal is ``-inf``, the rest is zero. Additive rather than boolean
    because that is what ``nn.MultiheadAttention`` takes, and built once per tower rather than per
    call because it depends only on the context length.
    """
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)
    return mask


class TextTransformer(nn.Module):
    """Token ids in, one vector out.

    Args:
        vocab_size: Rows in the token embedding.
        context_length: Row width, and the side of the causal mask.
        width: Transformer width.
        layers: Blocks in the stack.
        heads: Attention heads per block.
        embed_dim: Width of the shared space to project into.

    Examples:
        >>> TextTransformer(49408, 77, 512, 12, 8, 512)(tokens).shape    # doctest: +SKIP
        torch.Size([3, 512])
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        width: int,
        layers: int,
        heads: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.zeros(context_length, width))
        self.transformer = Transformer(width, layers, heads)
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.zeros(width, embed_dim))
        # A buffer, not a parameter: it is derived from the context length and carries no weights,
        # so it must not appear in the state dict the checkpoint is matched against.
        self.register_buffer("attn_mask", causal_mask(context_length), persistent=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return ``(batch, embed_dim)``, unnormalised.

        Args:
            tokens: ``(batch, context_length)`` int ids from the tokenizer.
        """
        x = self.token_embedding(tokens)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.transformer(x.permute(1, 0, 2), self.attn_mask.to(x.dtype)).permute(1, 0, 2)
        x = self.ln_final(x)

        # One row per sequence: the slot holding the end marker. See the module docstring.
        pooled = x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)]
        return pooled @ self.text_projection
