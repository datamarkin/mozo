"""Loading mozo's fused checkpoint into the two networks.

mozo publishes one file per variant holding both graphs, because a variant is one download.
Upstream publishes two files and shares the detector across all of them; converting to one is
``tools/fetch/easyocr.py``'s job, and this is the reading half of that agreement.
"""

from __future__ import annotations

__all__ = ["DETECTOR", "RECOGNISER", "load"]

from pathlib import Path
from typing import Any

import torch

from .config import Spec
from .craft import CRAFT
from .crnn import CRNN

#: The two keys ``tools/fetch/easyocr.py`` writes.
DETECTOR = "detector"
RECOGNISER = "recogniser"


def _strip_data_parallel(state: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``module.`` every upstream checkpoint carries.

    Both were saved from a ``DataParallel`` wrapper, so every key is prefixed. This is the only
    renaming either graph needs -- the module names in :mod:`.craft` and :mod:`.crnn` are
    upstream's, so a key here can be grepped for over there unchanged.
    """
    return {key[len("module."):] if key.startswith("module.") else key: value
            for key, value in state.items()}


def load(path: Path | str, spec: Spec) -> tuple[CRAFT, CRNN]:
    """Build both networks for ``spec`` and fill them from the checkpoint at ``path``.

    Raises:
        ValueError: If the file is not a fused checkpoint, or if the recogniser inside it was
            trained on a different-sized alphabet than ``spec`` describes. The second is the one
            that matters in practice: every variant's detector is identical and only the
            alphabet tells them apart, so a mismatch means the wrong variant's weights.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    missing = [key for key in (DETECTOR, RECOGNISER) if key not in blob]
    if missing:
        raise ValueError(
            f"{path} is not an easyocr checkpoint: no {' or '.join(missing)} inside it."
        )

    recogniser_state = _strip_data_parallel(blob[RECOGNISER])
    published = recogniser_state["Prediction.weight"].shape[0]
    if published != spec.num_class:
        raise ValueError(
            f"{path} holds a {published}-symbol alphabet but {spec.variant} has "
            f"{spec.num_class} ({len(spec.characters)} characters plus CTC's blank). "
            "This is another variant's checkpoint."
        )

    detector = CRAFT()
    detector.load_state_dict(_strip_data_parallel(blob[DETECTOR]))
    detector.eval()

    recogniser = CRNN(spec.num_class)
    recogniser.load_state_dict(recogniser_state)
    recogniser.eval()
    return detector, recogniser
