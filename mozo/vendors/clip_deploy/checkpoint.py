# SPDX-License-Identifier: Apache-2.0
"""Load a published CLIP checkpoint into one tower at a time, strictly.

Upstream's ``build_model`` infers the geometry from the state dict, deletes three bookkeeping keys,
and loads the whole model at once. None of that is carried:

**Three keys are not weights.** ``input_resolution``, ``context_length`` and ``vocab_size`` are
scalars recording how the model was built. :mod:`~mozo.vendors.clip_deploy.config` writes those
down instead, so they are dropped here -- and the strict load that follows is what holds the two
descriptions in step. Upstream deletes them for the same reason and then loads strict too.

**One tower at a time.** The checkpoint holds both, under ``visual.*`` and everything else. A text
service should not allocate the image tower, so the state dict is partitioned by prefix and each
half loaded into its own module. Loading is still strict within a half: a key that belongs to
neither partition, or a module left unfilled, is an error rather than a shrug.

**The file is not retained.** Each call re-reads it, so a process that builds one tower holds one
tower's parameters and not the 605 MB the whole state dict would cost. Reading twice is cheaper
than keeping it, because a tower is usually built once per process and the file is in the page
cache the second time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import Spec
from .network import build_text_tower, build_vision_tower
from .text.encoder import TextTransformer
from .vision.vit import VisionTransformer

__all__ = ["load_state_dict", "load_text_tower", "load_vision_tower", "read_logit_scale"]

#: Scalars recording how the model was built, not weights to load. See the module docstring.
_GEOMETRY = ("input_resolution", "context_length", "vocab_size")

#: The image tower's prefix. Everything else in the checkpoint belongs to the text side.
_VISION = "visual."


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a published checkpoint and return plain tensors.

    What mozo publishes is an ordinary ``.pth``, so that is tried first and is the only path a
    normal load takes. OpenAI publishes TorchScript archives instead -- a scripted module rather
    than a state dict -- and ``torch.jit.load`` recovers one. That stays supported so a checkpoint
    downloaded straight from OpenAI can be passed as ``checkpoint_path``, but it is the fallback:
    tried first it would run the recovery on every inference load and emit torch's TorchScript
    deprecation warning each time, for a format the published weights are not in.

    ``mmap=True`` because each tower uses only its half. Without it the whole two-tower checkpoint
    is materialised and ``_partition`` then discards the half it did not want, so building one
    tower allocates the other one's parameters on the way past. Mapped, only the tensors
    ``load_state_dict`` copies are faulted in. Measured on ``large``: peak RSS 3211 MB -> 2719 MB
    building the image tower, which is the 495 MB text half no longer being paid for. The tower
    itself is copied memory either way, and the tensors are identical -- ``tools/verify/clip.py``
    still reports every stage bit-exact with this on.
    """
    try:
        loaded: Any = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        return loaded
    except RuntimeError:
        return torch.jit.load(path, map_location="cpu").eval().state_dict()


def _partition(state: dict[str, torch.Tensor], vision: bool) -> dict[str, torch.Tensor]:
    """Return one tower's tensors, with the prefix stripped for the image side."""
    if vision:
        return {
            key[len(_VISION) :]: value
            for key, value in state.items()
            if key.startswith(_VISION)
        }
    return {
        key: value
        for key, value in state.items()
        if not key.startswith(_VISION) and key not in _GEOMETRY and key != "logit_scale"
    }


def load_vision_tower(spec: Spec, weights: str | Path, device: str = "cpu") -> VisionTransformer:
    """Build the image tower and fill it from *weights*.

    Raises:
        RuntimeError: If the checkpoint does not fit the architecture.
    """
    tower = build_vision_tower(spec)
    tower.load_state_dict(_partition(load_state_dict(weights), vision=True), strict=True)
    return tower.eval().to(device)


def load_text_tower(spec: Spec, weights: str | Path, device: str = "cpu") -> TextTransformer:
    """Build the text tower and fill it from *weights*.

    Raises:
        RuntimeError: If the checkpoint does not fit the architecture.
    """
    tower = build_text_tower(spec)
    tower.load_state_dict(_partition(load_state_dict(weights), vision=False), strict=True)
    return tower.eval().to(device)


def read_logit_scale(weights: str | Path) -> torch.Tensor:
    """Return the checkpoint's learned temperature, stored as a log and exponentiated here.

    It scales similarities into the logits upstream softmaxes over, and lands at 100.0 for every
    published variant. mozo returns raw cosine similarities, so nothing on the inference path
    multiplies by it -- it is read only by ``tools/verify/clip.py``, which compares against
    upstream's ``logits_per_image`` and needs it to do so.
    """
    return load_state_dict(weights)["logit_scale"].exp()
