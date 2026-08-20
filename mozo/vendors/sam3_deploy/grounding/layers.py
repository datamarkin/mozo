# SPDX-License-Identifier: Apache-2.0
"""Blocks shared across the grounding stage.

:class:`FusionLayer` is derived from ``transformers/models/sam3/modeling_sam3.py`` (Apache-2.0),
which spells the same block out twice -- once as ``Sam3GeometryEncoderLayer``, once as
``Sam3DetrEncoderLayer``. The checkpoint shows they are one structure: both carry ``self_attn``,
``cross_attn_image``, ``linear1``/``linear2`` and ``norm1``/``norm2``/``norm3`` with identical
shapes. So there is one class here, and the two uses differ only in what they hand it:

===================  ==============  =============  =================  ==============
use                  target          memory         self-attn gets     layout
===================  ==============  =============  =================  ==============
geometry encoder     prompt tokens   image tokens   no position        sequence-first
fusion encoder       image tokens    prompt tokens  ``target_pos``     batch-first
===================  ==============  =============  =================  ==============

The layout column is not cosmetic. ``nn.MultiheadAttention`` transposes internally when
``batch_first=True``, and the resulting memory layout selects a different kernel, which
accumulates differently: running the geometry encoder batch-first lands one float32 ulp from the
published model on half its outputs, and that grows through the stack. Each use gets the layout
its weights were run under.

Attention is ``nn.MultiheadAttention`` because the checkpoint stores a fused ``in_proj_weight``,
which is what that module expects. ``transformers`` splits it into three projections instead;
that is algebraically identical and numerically not, the same trap the trunk's rotary embedding
set in phase 1.

**Padding masks are ``True`` for padding**, PyTorch's ``key_padding_mask`` convention and the one
the checkpoint's own text encoder emits. ``transformers`` uses the opposite polarity internally.
Feeding the wrong one attends to exactly the tokens that should be ignored, and produces output
that looks plausible.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ..config import BlockSpec

__all__ = ["FusionLayer", "Mlp"]


class Mlp(nn.Module):
    """A stack of linear layers with ReLU between them, but not after the last.

    Lives here rather than in any one consumer because the DETR decoder, the scoring head and the
    mask head all build one. Named ``layers`` because that is what the checkpoint calls it.
    """

    def __init__(self, sizes: tuple[int, ...]):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        last = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            x = layer(x) if index == last else F.relu(layer(x))
        return x


class FusionLayer(nn.Module):
    """Pre-norm self-attention, cross-attention, then a feed-forward.

    Args:
        spec: Width, head count and feed-forward width.
        batch_first: Whether sequences arrive as ``(B, N, C)`` or ``(N, B, C)``. See the module
            docstring -- this changes the numbers, not just the shapes.
    """

    def __init__(self, spec: BlockSpec, *, batch_first: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(spec.hidden)
        self.self_attn = nn.MultiheadAttention(spec.hidden, spec.heads, batch_first=batch_first)
        self.norm2 = nn.LayerNorm(spec.hidden)
        self.cross_attn_image = nn.MultiheadAttention(
            spec.hidden, spec.heads, batch_first=batch_first
        )
        self.norm3 = nn.LayerNorm(spec.hidden)
        self.linear1 = nn.Linear(spec.hidden, spec.intermediate)
        self.linear2 = nn.Linear(spec.intermediate, spec.hidden)

    def forward(
        self,
        target: Tensor,
        memory: Tensor,
        memory_key: Tensor,
        *,
        target_padding: Tensor | None = None,
        memory_padding: Tensor | None = None,
        target_pos: Tensor | None = None,
    ) -> Tensor:
        """Update ``target`` by attending to itself and then to ``memory``.

        Args:
            target: the sequence being updated, in this layer's layout.
            memory: what it cross-attends to -- the attention's *value*.
            memory_key: the same sequence as the attention's *key*. The caller passes it
                separately so a position encoding can be added to the key and not the value, and
                so a stack can add it once rather than once per layer.
            target_padding: ``(B, N)`` True where ``target`` is padding. Always batch-first,
                whatever ``batch_first`` is -- that is what ``nn.MultiheadAttention`` expects.
            memory_padding: ``(B, M)`` True where ``memory`` is padding.
            target_pos: position encoding added to the self-attention's query and key, but never
                to its value. Omitted where the weights were run without one.

        Returns:
            ``target``, updated, in the layout it arrived in.
        """
        normed = self.norm1(target)
        query = normed if target_pos is None else normed + target_pos
        attended, _ = self.self_attn(
            query, query, normed, need_weights=False, key_padding_mask=target_padding
        )
        target = target + attended

        normed = self.norm2(target)
        attended, _ = self.cross_attn_image(
            normed, memory_key, memory, need_weights=False, key_padding_mask=memory_padding
        )
        target = target + attended

        normed = self.norm3(target)
        return target + self.linear2(F.relu(self.linear1(normed)))
