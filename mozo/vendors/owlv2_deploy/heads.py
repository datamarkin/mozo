# SPDX-License-Identifier: Apache-2.0
"""The three heads OWLv2 adds to CLIP, and the bias that makes the box head predict locally.

Every head reads the same thing: one feature vector per patch. There are no queries, no decoder
and no attention here -- the patch *is* the detection slot, which is why the answer needs no NMS
stage to collapse duplicate proposals and why the model returns exactly ``patches^2`` boxes
whatever you ask it.

**The heads activate with plain ``GELU``, the transformer blocks with ``quick_gelu``.** That is
upstream's arrangement -- ``nn.GELU()`` constructed directly in the box head, against
``ACT2FN[config.hidden_act]`` in the MLP -- and it is invisible in the weights. Using one for both
loads strictly and moves every box.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["BoxHead", "ClassHead", "ObjectnessHead", "box_bias"]


class ClassHead(nn.Module):
    """Score every patch against every prompt.

    Args:
        width: The trunk's width -- what a patch feature is.
        query_width: The prompt embedding's width, which the patch is projected to.
    """

    def __init__(self, width: int, query_width: int):
        super().__init__()
        self.dense0 = nn.Linear(width, query_width)
        self.logit_shift = nn.Linear(width, 1)
        self.logit_scale = nn.Linear(width, 1)
        self.elu = nn.ELU()

    def forward(self, patches: Tensor, queries: Tensor, query_mask: Tensor) -> Tensor:
        """Cosine similarity per (patch, prompt), then a learned per-patch affine correction.

        Args:
            patches: ``(B, P, width)``.
            queries: ``(B, Q, query_width)``, already L2-normalised by the caller -- and
                normalised again below, which is upstream's arrangement and not a no-op at float
                precision, because the ``+ 1e-6`` shifts a unit vector very slightly.
            query_mask: ``(B, Q)`` bool. False marks a prompt slot that carries no prompt.

        Returns:
            ``(B, P, Q)`` logits, before any sigmoid.
        """
        projected = self.dense0(patches)
        projected = projected / (torch.linalg.norm(projected, dim=-1, keepdim=True) + 1e-6)
        queries = queries / (torch.linalg.norm(queries, dim=-1, keepdim=True) + 1e-6)
        logits = torch.einsum("...pd,...qd->...pq", projected, queries)

        # Read off the *unprojected* patch, not the normalised one. A cosine similarity has no
        # scale of its own, so these two numbers are what turn it into something a sigmoid can be
        # thresholded on -- and they are per patch, so a crowded patch and an empty one get
        # different calibrations.
        shift = self.logit_shift(patches)
        scale = self.elu(self.logit_scale(patches)) + 1
        logits = (logits + shift) * scale

        # An empty prompt slot must score nothing anywhere, rather than scoring whatever the
        # zeroed embedding happens to correlate with. Driven to the dtype's floor so the sigmoid
        # downstream reads exactly zero.
        return torch.where(query_mask[:, None, :], logits, torch.finfo(logits.dtype).min)


class _MLP(nn.Module):
    """Two ``width``-wide layers with ``GELU`` between, then down to ``out``."""

    def __init__(self, width: int, out: int):
        super().__init__()
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out)

    def forward(self, x: Tensor) -> Tensor:
        return self.dense2(self.gelu(self.dense1(self.gelu(self.dense0(x)))))


class BoxHead(_MLP):
    """Predict one box per patch, as a logit offset from that patch's own position."""

    def __init__(self, width: int):
        super().__init__(width, 4)

    def forward(self, patches: Tensor, bias: Tensor) -> Tensor:
        """``(B, P, width)`` and ``(P, 4)`` in, ``(B, P, 4)`` normalised cxcywh out.

        The head predicts in logit space and the bias is added there, before the sigmoid -- so a
        head that predicts nothing at all still emits a box centred on its own patch and one patch
        wide. That is the prior, and it is what lets a patch-per-box model localise at all.
        """
        return torch.sigmoid(super().forward(patches) + bias)


class ObjectnessHead(_MLP):
    """Predict how likely a patch is to hold an object at all, whatever was asked for.

    OWLv2's addition over OWL-ViT. Query-agnostic, so it ranks and filters independently of the
    prompt -- and it is trained on a detached feature, so it never shapes the trunk.
    """

    def __init__(self, width: int):
        super().__init__(width, 1)

    def forward(self, patches: Tensor) -> Tensor:
        """``(B, P, width)`` in, ``(B, P)`` logits out."""
        return super().forward(patches)[..., 0]


def box_bias(patches: int) -> Tensor:
    """The per-patch prior the box head predicts against, in logit space.

    Args:
        patches: Patches along one side. The grid is this squared.

    Returns:
        ``(patches^2, 4)``: the logit of each patch's own centre, then the logit of one patch's
        width and height. Row-major over the grid, matching the order the trunk flattens in.

    The ``1e-4`` is upstream's, and it is doing real work at both ends: without it the first
    column's centre is ``log(0)``. It is inside both terms of the logit rather than only the
    first, which is what keeps the transform monotone.
    """
    across = torch.arange(1, patches + 1, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(across, across, indexing="xy"), dim=-1) / patches
    grid = grid.view(-1, 2).clip(0.0, 1.0)
    centre = torch.log(grid + 1e-4) - torch.log1p(-grid + 1e-4)
    size = torch.full_like(grid, 1.0 / patches)
    return torch.cat([centre, torch.log(size + 1e-4) - torch.log1p(-size + 1e-4)], dim=-1)
