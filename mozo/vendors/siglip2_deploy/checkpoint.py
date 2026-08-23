# SPDX-License-Identifier: Apache-2.0
"""Loading one published checkpoint into whichever tower asked for it.

Upstream builds one ``SiglipModel`` holding both towers and loads the checkpoint into it whole.
This partitions the state dict by prefix and loads each tower on its own, so an ingest job never
allocates the text half and a query service never allocates the image half.

That is worth more here than it was for CLIP. SigLIP 2's text tower carries Gemma's 256,000-piece
vocabulary, which is 786 MB of a base checkpoint and 1,180 MB of an ``so400m`` one -- most of the
file, for a table an image-encoding job never reads. ``mmap=True`` means the untouched half is
never faulted in rather than merely being freed afterwards.

**The load stays strict within each half.** A key belonging to neither partition, or a module left
unfilled, is an error. That strictness is what holds ``config.py``'s written-down geometry against
the checkpoint's actual shapes: a spec that is inferred cannot be checked, and one that is written
down is only worth writing if something verifies it.

``logit_scale`` and ``logit_bias`` belong to neither tower. They are the scoring head, they are read
separately, and both are on the inference path -- unlike CLIP, where mozo loads the scale and never
multiplies by it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Spec
from .text.encoder import TextTower
from .vision.vit import VisionTower

__all__ = ["load_scoring", "load_state_dict", "load_text_tower", "load_vision_tower"]

_VISION = "vision_model."
_TEXT = "text_model."


def load_state_dict(path: str | Path) -> dict[str, Any]:
    """Read a published checkpoint, without faulting in more of it than is asked for."""
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _partition(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    """The tensors under *prefix*, with it stripped.

    Upstream's module names and this package's are the same words in the same order, so a prefix
    strip is the whole translation. That is deliberate: a rename here would be a second table to
    keep in step with a checkpoint nobody can change.
    """
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def load_vision_tower(spec: Spec, path: str | Path, device: str = "cpu") -> VisionTower:
    """Build and fill the image tower alone."""
    tower = VisionTower(spec)
    tower.load_state_dict(_partition(load_state_dict(path), _VISION), strict=True)
    return tower.eval().to(device)


def load_text_tower(spec: Spec, path: str | Path, device: str = "cpu") -> TextTower:
    """Build and fill the text tower alone."""
    tower = TextTower(spec)
    tower.load_state_dict(_partition(load_state_dict(path), _TEXT), strict=True)
    return tower.eval().to(device)


def load_scoring(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """The learned temperature and bias, as ``(logit_scale, logit_bias)``.

    Both are stored as one-element tensors and both are used as they come: upstream computes
    ``similarity * logit_scale.exp() + logit_bias``. The exponential is taken at use, not here,
    because that is where upstream takes it.

    **Copied out of the mapping, and that is the whole reason this is three lines.** ``mmap=True``
    backs every tensor in the checkpoint with one shared mapping over the entire file, so a tensor
    read out of it is a *view*: keeping these two four-byte scalars would keep the 1.5 GB to 7.5 GB
    mapping resident for as long as the encoder lives, which under a served model is the life of
    the process. Cloning eight bytes lets the mapping go at the end of this function.
    """
    state = load_state_dict(path)
    return state["logit_scale"].clone(), state["logit_bias"].clone()
