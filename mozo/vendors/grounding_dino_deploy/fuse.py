# SPDX-License-Identifier: Apache-2.0
"""Bi-directional image/text fusion, run once per encoder layer.

Image tokens attend to text tokens and text tokens attend to image tokens, from a single attention
matrix read both ways. This is where "grounding" happens: after six of these, an image token knows
which words it looks like and a word knows where in the picture it is.

Both updates are scaled by a learned per-channel gain (``gamma_v``, ``gamma_l``) before being
added back, which is layer scale -- initialised tiny so fusion starts as a no-op and the model
learns how much of it to let through.

Upstream's ``fuse_modules.py`` also carries ``FeatureResizer``, ``func_attention``, ``l1norm`` and
``l2norm``. None is reachable from the detection path; none is carried.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["BiAttentionBlock"]

#: Ceiling and floor applied to the attention logits before softmax. Upstream's comment is
#: explicit that these exist for fp16's range and must not be widened. Kept at the published
#: values because they are inside the arithmetic, not around it -- a wider clamp is a different
#: number wherever a logit actually reaches them.
_CLAMP = 50000


class _BiMultiHeadAttention(nn.Module):
    """One attention matrix between image and text, consumed in both directions.

    Args:
        v_dim: Image token width.
        l_dim: Text token width.
        embed_dim: Internal attention width.
        num_heads: Attention heads.
    """

    def __init__(self, v_dim: int, l_dim: int, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} is not divisible by num_heads {num_heads}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** (-0.5)

        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.l_proj = nn.Linear(l_dim, embed_dim)
        self.values_v_proj = nn.Linear(v_dim, embed_dim)
        self.values_l_proj = nn.Linear(l_dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, v_dim)
        self.out_l_proj = nn.Linear(embed_dim, l_dim)

    def _shape(self, tensor: Tensor, seq_len: int, batch: int) -> Tensor:
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        attention_mask_v: Tensor | None = None,
        attention_mask_l: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(image_update, text_update)``.

        Upstream calls these ``v`` and ``l``, for vision and language, and the projection names
        in the checkpoint keep those letters -- ``v_proj``, ``l_proj``. The arguments are spelled
        out because a lowercase ``l`` is indistinguishable from a ``1`` in most fonts.

        Args:
            image: Image tokens, ``(batch, image_tokens, v_dim)``.
            text: Text tokens, ``(batch, text_tokens, l_dim)``.
            attention_mask_v: ``(batch, image_tokens)``, True where padded.
            attention_mask_l: ``(batch, text_tokens)``, True where padded.
        """
        batch, image_tokens, _ = image.size()

        query = self._shape(self.v_proj(image) * self.scale, image_tokens, batch)
        key = self._shape(self.l_proj(text), -1, batch)
        values_v = self._shape(self.values_v_proj(image), -1, batch)
        values_l = self._shape(self.values_l_proj(text), -1, batch)

        flat = (batch * self.num_heads, -1, self.head_dim)
        query = query.view(*flat)
        key = key.view(*flat)
        values_v = values_v.view(*flat)
        values_l = values_l.view(*flat)

        text_tokens = key.size(1)
        attn = torch.bmm(query, key.transpose(1, 2))

        # Subtracting the global maximum, not a per-row one. That is what upstream does, and a
        # per-row subtraction -- the usual stable-softmax spelling -- is a different number.
        attn = attn - attn.max()
        attn = torch.clamp(attn, min=-_CLAMP, max=_CLAMP)

        attn_l = attn.transpose(1, 2)
        attn_l = attn_l - torch.max(attn_l, dim=-1, keepdim=True)[0]
        attn_l = torch.clamp(attn_l, min=-_CLAMP, max=_CLAMP)

        if attention_mask_v is not None:
            expanded = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_l = attn_l.masked_fill(expanded, float("-inf"))
        attn_l = attn_l.softmax(dim=-1)

        if attention_mask_l is not None:
            expanded = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn = attn.masked_fill(expanded, float("-inf"))
        attn_v = attn.softmax(dim=-1)

        out_v = torch.bmm(attn_v, values_l)
        out_l = torch.bmm(attn_l, values_v)

        out_v = (
            out_v.view(batch, self.num_heads, image_tokens, self.head_dim)
            .transpose(1, 2)
            .reshape(batch, image_tokens, self.embed_dim)
        )
        out_l = (
            out_l.view(batch, self.num_heads, text_tokens, self.head_dim)
            .transpose(1, 2)
            .reshape(batch, text_tokens, self.embed_dim)
        )
        return self.out_v_proj(out_v), self.out_l_proj(out_l)


class BiAttentionBlock(nn.Module):
    """Pre-norm bi-directional fusion with layer scale.

    Examples:
        >>> block = BiAttentionBlock(256, 256, 1024, 4)      # doctest: +SKIP
        >>> image, text = block(image, text)                 # doctest: +SKIP
    """

    def __init__(self, v_dim: int, l_dim: int, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = _BiMultiHeadAttention(v_dim, l_dim, embed_dim, num_heads)
        # Layer scale. Trained values, not the 1e-4 initialisation.
        self.gamma_v = nn.Parameter(torch.ones(v_dim))
        self.gamma_l = nn.Parameter(torch.ones(l_dim))

    def forward(
        self,
        image: Tensor,
        text: Tensor,
        attention_mask_v: Tensor | None = None,
        attention_mask_l: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the updated ``(image, text)`` tokens.

        Note the residual is added to the *normalised* input, not to the input: upstream
        reassigns ``v`` and ``l`` before the attention call, so the pre-norm is not merely a
        pre-norm. Reading it as the usual ``x + f(norm(x))`` gives a different model.
        """
        image = self.layer_norm_v(image)
        text = self.layer_norm_l(text)
        delta_v, delta_l = self.attn(
            image, text, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        return image + self.gamma_v * delta_v, text + self.gamma_l * delta_l
