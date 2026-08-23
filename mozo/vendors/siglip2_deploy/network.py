# SPDX-License-Identifier: Apache-2.0
"""The one arithmetic helper both towers share."""

from __future__ import annotations

import torch

__all__ = ["normalise"]


def normalise(vectors: torch.Tensor) -> torch.Tensor:
    """L2-normalise along the last axis, the way upstream's ``forward`` does.

    Written as the explicit division rather than ``F.normalize`` because that is what
    ``SiglipModel.forward`` runs. The two are bit-identical here -- measured -- but the gate
    compares against upstream's expression, not against an equivalent one, and an equivalence that
    holds today is not a reason to depend on it.
    """
    return vectors / vectors.norm(p=2, dim=-1, keepdim=True)
