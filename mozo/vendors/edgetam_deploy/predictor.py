# SPDX-License-Identifier: Apache-2.0
"""The public segmenter: load a checkpoint, prompt an image, get masks in its own pixels."""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch

from .image import MASK_THRESHOLD, preprocess, to_model_coords, to_original
from .network import EdgeTam

__all__ = ["Segmentation", "Segmenter"]

#: How a box is spelled, since EdgeTAM has no separate box input: its two corners, carrying label
#: values reserved for them. Clicks use 1 and 0, which the caller passes and this module only
#: forwards, so they are described where they are asked for rather than named here.
BOX_TOP_LEFT, BOX_BOTTOM_RIGHT = 2, 3

#: Bound on the low-resolution logits handed back for refinement. Upstream's value.
LOGIT_LIMIT = 32.0

#: How many images' encoder outputs to keep. Each is ~21 MB at fp32, so five is ~105 MB -- small
#: enough to be unremarkable on any deployment target, which is why there is one number here and
#: not a tier system.
CACHE_SIZE = 5


@dataclass
class Segmentation:
    """Masks for one prompt batch, in the source image's own pixels."""

    masks: np.ndarray  # (b, c, h, w) bool
    scores: np.ndarray  # (b, c) predicted IoU of each mask
    # Feed one of these back as ``mask_input`` to refine, choosing the candidate first:
    # ``logits[:, scores.argmax(1)]``. The decoder takes a single channel, so handing it all
    # three raises rather than picking for you.
    logits: np.ndarray  # (b, c, 256, 256) low-res logits

    def __len__(self) -> int:
        return len(self.masks)


class Segmenter:
    """Promptable segmentation for the published EdgeTAM checkpoint.

    Args:
        checkpoint: Path to a ``.pt`` holding an EdgeTAM state dict.
        device: Where to run. mozo decides this; the default is only for direct use.

    Attributes:
        image_size: Square side the encoder runs at.
    """

    def __init__(self, checkpoint: str | os.PathLike, device: str | torch.device = "cpu") -> None:
        record = torch.load(os.fspath(checkpoint), map_location="cpu", weights_only=True)
        state = record.get("model", record)

        self.network = EdgeTam()
        # The video tracker's weights are in the file and have nowhere to go in an image-only
        # build. Dropping them by name -- rather than passing strict=False and hoping -- keeps
        # the load strict about everything else, so a genuinely missing tensor is still an error.
        wanted = self.network.state_dict()
        self.network.load_state_dict({k: v for k, v in state.items() if k in wanted})
        self.network.eval().to(device)

        self.device = device
        # Read off the network rather than the config, so the size the encoder was built at and
        # the size prompts are scaled into cannot come from two lookups that could disagree.
        self.image_size = self.network.image_size
        self._cache: OrderedDict[bytes, dict] = OrderedDict()
        # One adapter instance is shared across requests -- mozo.server runs handlers in a
        # threadpool -- and check-then-act on an OrderedDict is not safe across threads.
        self._lock = Lock()

    def encode(self, image: np.ndarray) -> dict[str, torch.Tensor]:
        """Return the encoder features for one image, computing them only if not already held.

        This is the expensive half of EdgeTAM and depends on nothing but the image, so an
        annotator clicking repeatedly on one photograph pays for it once. Keyed on pixel content
        rather than on a filename or an object identity, because the same image arriving twice
        over HTTP is two different arrays and should still be one encode.
        """
        # sha256 over the pixels, not a sample of them: a key that skipped content would collide
        # on two images differing only where it did not look, and hand back a confident mask of
        # the wrong object. Hashed straight from the array -- ``tobytes`` would copy every byte --
        # and sha256 rather than blake2b because it reaches the CPU's crypto instructions.
        key = hashlib.sha256(np.ascontiguousarray(image)).digest()
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        batch = preprocess(image, self.image_size).to(self.device)
        features = self.network.encode(batch)
        with self._lock:
            self._cache[key] = features
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return features

    def predict(
        self,
        image: np.ndarray,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
        mask_input: np.ndarray | None = None,
        multimask_output: bool = True,
    ) -> Segmentation:
        """Segment what the prompt points at.

        Args:
            image: ``HxWx3`` RGB ``uint8`` array.
            points: ``(N, 2)`` or ``(B, N, 2)`` x, y clicks in the image's own pixels.
            labels: ``(N,)`` or ``(B, N)``, 1 for a point to include and 0 for one to exclude.
                Required with *points*; there is no default, because guessing between include and
                exclude silently returns a plausible mask of the wrong thing.
            boxes: ``(4,)`` or ``(B, 4)`` x1, y1, x2, y2 in the image's own pixels.
            mask_input: ``(B, 1, 256, 256)`` logits from a previous call, to refine. A previous
                :class:`Segmentation` carries three candidates, so select one before passing it.
            multimask_output: Return three candidate masks instead of one. Worth keeping on for a
                single click, which is ambiguous about whether you meant the part or the whole.

        Returns:
            A :class:`Segmentation` whose masks are in the source image's pixels.

        Raises:
            ValueError: If no prompt is given, if points arrive without labels, or if a batch of
                boxes and a batch of point sets disagree on how many prompts there are.
        """
        if points is None and boxes is None and mask_input is None:
            raise ValueError("a prompt is required: give points, boxes or mask_input")
        if (points is None) != (labels is None):
            raise ValueError("points and labels go together; got one without the other")

        shape = image.shape[:2]
        features = self.encode(image)
        coords, marks = self._prompt(points, labels, boxes, shape)
        low_res, iou = self.network.decode(
            features,
            coords,
            marks,
            None if mask_input is None else torch.as_tensor(mask_input).float().to(self.device),
            multimask_output,
        )
        masks = to_original(low_res, shape)
        return Segmentation(
            masks=(masks > MASK_THRESHOLD).cpu().numpy(),
            scores=iou.cpu().numpy(),
            # Bounded before being handed back, because this is what a refining caller feeds in
            # as ``mask_input`` next -- and a logit that is free to grow each round would
            # eventually swamp the click that is meant to correct it. The masks above come from
            # the unclamped tensor, so the bound costs nothing at the threshold.
            logits=low_res.clamp(-LOGIT_LIMIT, LOGIT_LIMIT).cpu().numpy(),
        )

    def _prompt(self, points, labels, boxes, shape):
        """Fold points and boxes into the one point list the prompt encoder takes.

        EdgeTAM has no box input. A box is spelled as its two corners carrying reserved labels,
        and the corners are placed before any clicks because that is the order upstream's own
        image predictor builds -- which is what keeps ``tools/verify/edgetam.py`` exact.

        The order is *not* semantically load-bearing, and it is worth saying so because the
        obvious guess is that it would be. The prompt encoder picks each token's learned
        embedding from its **label**, not from its position, and the sparse tokens are only ever
        read through attention rather than sliced by index -- so a different order is the same
        multiset of tokens. Measured: putting the click first instead moves the mask logits by
        3.1e-05, which is float summation order, not a different mask.
        """
        if points is None and boxes is None:
            return None, None

        groups, marks = [], []
        if boxes is not None:
            corners = np.asarray(boxes, dtype=np.float32).reshape(-1, 2, 2)
            groups.append(to_model_coords(corners, shape, self.image_size))
            corner_labels = torch.tensor([[BOX_TOP_LEFT, BOX_BOTTOM_RIGHT]], dtype=torch.int32)
            marks.append(corner_labels.repeat(len(corners), 1))
        if points is not None:
            clicks = np.asarray(points, dtype=np.float32)
            flags = np.asarray(labels, dtype=np.int32)
            if clicks.ndim == 2:
                clicks, flags = clicks[None], flags[None]
            if clicks.shape[:2] != flags.shape[:2]:
                raise ValueError(f"{clicks.shape[1]} points but {flags.shape[1]} labels")
            groups.append(to_model_coords(clicks, shape, self.image_size))
            marks.append(torch.as_tensor(flags, dtype=torch.int32))

        if len(groups) == 2 and groups[0].shape[0] != groups[1].shape[0]:
            raise ValueError(
                f"{groups[0].shape[0]} boxes and {groups[1].shape[0]} point sets: give one point "
                "set per box, or prompt with only one of the two"
            )
        return (
            torch.cat(groups, dim=1).to(self.device),
            torch.cat(marks, dim=1).to(self.device),
        )
