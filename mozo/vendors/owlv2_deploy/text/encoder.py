# SPDX-License-Identifier: Apache-2.0
"""OWLv2's prompt tower: CLIP's text transformer, pooled and projected into one query embedding.

This is the cheap half, and it depends only on the prompt. A phrase encoded once stays valid for
every image afterwards, which is what makes annotating a corpus with a fixed vocabulary cost one
text forward rather than one per picture. Upstream re-runs this tower on every call;
:mod:`~mozo.vendors.owlv2_deploy.network` keeps the two halves apart so it does not have to.

**Pooling is by ``argmax`` over the token ids, not by position.** The end-of-text marker is the
highest id in the vocabulary, so the slot holding it is the slot with the largest id -- which is
how CLIP finds the end of a variable-length prompt without being told its length. It survives
OWLv2's padding token being id 0 and it survives truncation, because a truncated prompt keeps the
marker in its last slot. It would *not* survive a prompt containing a literal ``<|endoftext|>``,
which would pool at that one instead; upstream has the same behaviour and this package keeps it.

**The mask is causal and padding-aware at once.** Prompts are padded to a fixed 16, so a mask that
is only causal would let every prompt attend to the tail of zeros after it, and a mask that is
only padding-aware would let each token see the ones after it. Both are wrong and neither is
visible in the output shape.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import TextSpec
from ..layers import Encoder

__all__ = ["TextTower"]


class Embeddings(nn.Module):
    """Look up the token, add the learned position embedding.

    Args:
        spec: Text geometry.
    """

    def __init__(self, spec: TextSpec):
        super().__init__()
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.width)
        self.position_embedding = nn.Embedding(spec.context_length, spec.width)
        self.register_buffer(
            "position_ids", torch.arange(spec.context_length).expand((1, -1)), persistent=False
        )

    def forward(self, ids: Tensor) -> Tensor:
        """``(B, L)`` in, ``(B, L, width)`` out."""
        positions = self.position_ids[:, : ids.shape[-1]]
        return self.token_embedding(ids) + self.position_embedding(positions)


class TextTower(nn.Module):
    """Token ids in, one pooled embedding per prompt out.

    Args:
        spec: Text geometry.
    """

    def __init__(self, spec: TextSpec):
        super().__init__()
        self.embeddings = Embeddings(spec)
        self.encoder = Encoder(
            spec.layers, spec.width, spec.heads, spec.intermediate, spec.layer_norm_eps
        )
        self.final_layer_norm = nn.LayerNorm(spec.width, eps=spec.layer_norm_eps)

    def forward(self, ids: Tensor, mask: Tensor) -> Tensor:
        """Encode a batch of tokenized prompts.

        Args:
            ids: ``(B, L)`` from :class:`~.tokenizer.Tokenizer`.
            mask: ``(B, L)``, 1 where the row carries a real token. Passed in rather than derived
                from ``ids``, because id 0 means both "padding" and "``!``".

        Returns:
            ``(B, width)``, one embedding per prompt, taken at the end-of-text marker.
        """
        hidden = self.embeddings(ids)
        hidden = self.encoder(hidden, _attention_mask(mask, hidden.dtype))
        hidden = self.final_layer_norm(hidden)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, ids.int().argmax(dim=-1)]


def _attention_mask(mask: Tensor, dtype: torch.dtype) -> Tensor:
    """Build the ``(B, 1, L, L)`` additive mask: causal, and blind to padding.

    Args:
        mask: ``(B, L)``, 1 where the row carries a real token.
        dtype: What the hidden states are, so the fill matches their finite minimum.
    """
    batch, length = mask.shape
    blocked = torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=mask.device), diagonal=1
    )
    # Broadcast the padding columns across every query row, then take either reason to block.
    blocked = blocked[None, None] | (mask == 0)[:, None, None, :]
    return torch.zeros(batch, 1, length, length, dtype=dtype, device=mask.device).masked_fill(
        blocked, torch.finfo(dtype).min
    )
