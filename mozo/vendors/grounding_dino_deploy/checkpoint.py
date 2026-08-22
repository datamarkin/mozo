# SPDX-License-Identifier: Apache-2.0
"""Load a published Grounding DINO checkpoint into :class:`~.network.GroundingDino`, strictly.

Upstream loads with ``strict=False``. That is not carried. A permissive load cannot tell a tensor
this package deliberately does not build from one it named differently by mistake -- the second
leaves a module at its random initialisation, runs, and returns confident wrong boxes. So what is
genuinely absent is dropped **by prefix**, deliberately and in one place, and everything else must
match or the load fails.

Three groups are dropped, and each is dropped because it is not on the inference path:

``label_enc``
    A 2,001-row embedding for denoising training. Never read at inference.

``bert.pooler``
    BERT's ``[CLS]`` pooler. Upstream freezes it and the detection path reads only
    ``last_hidden_state``.

``bert.embeddings.position_ids``
    A buffer old ``transformers`` registered and new versions do not. It holds ``arange(512)``
    and carries no information; Grounding DINO passes its own position ids anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Spec
from .network import GroundingDino

__all__ = ["build", "load_state_dict"]

#: Weights present in every published checkpoint that this package does not build. Dropping them
#: by prefix -- rather than passing ``strict=False`` and hoping -- keeps the load strict about
#: everything else, so a genuinely missing tensor is still an error.
_UNUSED = (
    "label_enc.",
    "bert.pooler.",
    "bert.embeddings.position_ids",
)

#: The one prefix upstream spells differently. Its backbone is a ``Joiner(nn.Sequential)`` holding
#: the Swin at index 0 and the position encoding -- which has no parameters -- at index 1, so every
#: backbone tensor arrives under ``backbone.0.``. This package holds the Swin directly, because a
#: two-element sequential whose second element is stateless is a wrapper, not a structure.
_JOINER_PREFIX = "backbone.0."


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a checkpoint and return its tensors, ready to load.

    Handles the two wrappers upstream's files carry: the state dict lives under ``"model"``, and
    every key is prefixed ``module.`` from the ``DataParallel`` the weights were saved through.
    """
    raw: Any = torch.load(path, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def build(spec: Spec, weights: str | Path, device: str = "cpu") -> GroundingDino:
    """Build the network *spec* describes and load *weights* into it.

    Takes the spec rather than its name. Looking the name up again here would mean a caller who
    passed a modified spec -- a different short side, say -- got a network built from the
    registry's copy instead, running its preprocessing against someone else's geometry.

    Args:
        spec: Which variant to build. See :data:`~.config.SPECS`.
        weights: Path to the checkpoint.
        device: Where to put the model.

    Raises:
        RuntimeError: If the checkpoint does not fit the architecture.
    """
    state = {
        key.replace(_JOINER_PREFIX, "backbone.", 1) if key.startswith(_JOINER_PREFIX) else key: value
        for key, value in load_state_dict(weights).items()
        if not any(key.startswith(prefix) for prefix in _UNUSED)
    }

    # Built on the meta device, so its 173M parameters are described rather than allocated and
    # initialised. Every one of them is overwritten by the load, so the usual path spends 0.62 s
    # on random numbers it throws away and holds the checkpoint twice while copying into them.
    # ``assign=True`` adopts the loaded tensors instead of copying into empty ones: measured
    # 0.77 s and 1670 MB peak against 0.14 s and 978 MB. The load stays strict, so a tensor left
    # on ``meta`` -- which is what a missing key would now look like -- is still an error.
    with torch.device("meta"):
        model = GroundingDino(spec)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().to(device)
