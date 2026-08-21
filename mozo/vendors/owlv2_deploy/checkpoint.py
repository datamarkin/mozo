# SPDX-License-Identifier: Apache-2.0
"""Mapping Google's published checkpoint onto this package's module names.

mozo consumes ``pytorch_model.bin`` as Google publishes it, byte for byte -- no repacking, no
pruning, no separate mozo-format artifact. The published file already follows ``transformers``'
names, which is what this package derives from, so the translation is four rules rather than the
several dozen a differently-organised upstream would need.

Two keys are dropped rather than renamed. ``visual_projection`` and ``logit_scale`` implement
CLIP's contrastive head, which the detector never reads -- so there is nothing here to load them
into, and a strict load has to be told that on purpose rather than being quietly loosened. Keeping
the drop explicit is the point: a *third* unexpected key would still be an error.

The official checkpoints are JAX; these are the PyTorch conversions Google publishes beside them
on Hugging Face, under the same Apache-2.0 terms. See ``PROVENANCE.md``.
"""

from __future__ import annotations

from pathlib import Path

import torch

__all__ = ["DROPPED", "RULES", "load_state_dict", "translate"]

#: Published prefix -> this package's prefix, applied to the first match only.
RULES: tuple[tuple[str, str], ...] = (
    ("owlv2.vision_model.", "vision."),
    ("owlv2.text_model.", "text."),
    ("owlv2.text_projection.", "text_projection."),
)

#: Present in every published checkpoint, built by no module here. See the module docstring.
DROPPED: frozenset[str] = frozenset({"owlv2.visual_projection.weight", "owlv2.logit_scale"})


def translate(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename a published state dict onto this package's modules, dropping the two dead tensors.

    Args:
        state: What ``torch.load`` returned for a published checkpoint.

    Returns:
        A new dict. The tensors are the same objects; only the keys change.
    """
    translated = {}
    for key, tensor in state.items():
        if key in DROPPED:
            continue
        for published, ours in RULES:
            if key.startswith(published):
                key = ours + key[len(published) :]
                break
        translated[key] = tensor
    return translated


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a published checkpoint from disk and translate it.

    Args:
        path: The ``torch-fp32.pth`` mozo publishes, which is Google's ``pytorch_model.bin``.

    Returns:
        A state dict keyed for :class:`~.network.OwlV2`.
    """
    return translate(torch.load(path, map_location="cpu", weights_only=True))
