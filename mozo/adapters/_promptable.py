"""What every promptable-segmentation adapter does, which is the same thing.

A promptable model answers "what is *here*" rather than "where are the 80 COCO classes". You give
it a click or a box; it gives you back the thing you pointed at. SAM 2, EdgeTAM, MobileSAM and
EfficientViT-SAM all answer that question with the same prompt vocabulary and the same output
shape, and they differ only in the trunk that produces the features -- which lives entirely in
their vendored packages.

So the vendors are deliberately independent copies of each other and the adapters are not. The
vendors have a fidelity argument for duplication: each must be reproducible against its own
upstream, and a shared substrate would let one family's re-sync move another's masks. Nothing
here touches a number.

A family subclasses this with three class attributes and its own docstring.

**There is no class name, and none is invented.** A click does not say what it clicked. PixelFlow
leaves ``class_name`` as ``None`` when no labels are given, which is the honest answer, so that is
what comes back. A caller who knows what they pointed at can pass ``name=`` and have it attached
-- the same rule the rest of mozo follows, where a name comes from the weights or from the user
and never from the library.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Union

import numpy as np
import torch

from ..image import load_image
from ..runtimes import get_default_device, select_runtime
from ..weights import artifacts, resolve

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["PromptablePredictor"]


def bounds(masks: np.ndarray) -> np.ndarray:
    """Return the ``(N, 4)`` x1, y1, x2, y2 box that tightly encloses each mask.

    A promptable model returns a mask, not a box, but every consumer of a PixelFlow ``Detections``
    expects one -- so it is derived here rather than left empty.

    Args:
        masks: ``(N, H, W)`` boolean masks.

    Returns:
        ``(N, 4)`` float32 boxes. A mask with no foreground gets all zeros, which is the only
        honest answer for a box around nothing; the row is kept rather than dropped so that
        scores, masks and boxes stay index-aligned with what the model returned.
    """
    boxes = np.zeros((len(masks), 4), dtype=np.float32)
    for index, mask in enumerate(masks):
        rows = np.flatnonzero(mask.any(axis=1))
        columns = np.flatnonzero(mask.any(axis=0))
        if len(rows) and len(columns):
            boxes[index] = (columns[0], rows[0], columns[-1] + 1, rows[-1] + 1)
    return boxes


class PromptablePredictor:
    """One loaded variant of one promptable-segmentation family, ready to prompt.

    Subclasses set:

    ``FAMILY``    the name the manifest and the registry use, e.g. ``"edgetam"``.
    ``DISPLAY``   what to call it when talking to a person, e.g. ``"EdgeTAM"``.
    ``VENDOR``    the vendored package, which must expose ``Segmenter``.
    ``VARIANTS``  the published variant names, the first of which is the default.

    ``EXECUTES`` is shared rather than per-family: every promptable model splits into an encoder
    graph and a decoder graph, and mozo has no runner that keeps the two apart, so none of them
    can execute a graph runtime yet. Declared to :func:`~mozo.runtimes.select_runtime` rather
    than checked afterwards, so ``auto`` never *chooses* an artifact this adapter would then have
    to refuse -- SAM 2 publishes CoreML and ONNX, and ``auto`` picking one is a preference-table
    edit away.

    Args:
        variant: A published variant.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which published artifact to execute. ``"auto"`` takes the best one published
            for the device, which is a torch one for every family here.
        checkpoint_path: A checkpoint of your own instead of the published weights.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        variant: The variant in use.
        device: The device actually in use.
        runtime: The artifact key actually in use.
    """

    FAMILY: str
    DISPLAY: str
    VENDOR: ModuleType
    VARIANTS: tuple[str, ...]

    #: Frameworks this adapter has code to run. See the class docstring.
    EXECUTES: tuple[str, ...] = ("torch",)

    def __init__(
        self,
        variant: str | None = None,
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        variant = variant or self.VARIANTS[0]
        # Checked even with your own checkpoint: ``self.variant`` reports it back, and a name
        # that names nothing would be reported as though it meant something.
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        # What is runnable is the manifest's to declare and what is executable is
        # ``select_runtime``'s to decide. It reads both and raises with the available names, so
        # there is nothing for this adapter to restate or to re-check afterwards.
        self.runtime = (
            "torch-fp32" if checkpoint_path is not None
            else select_runtime(
                self.device,
                artifacts(self.FAMILY, variant, revision=revision),
                runtime,
                executes=self.EXECUTES,
            )
        )

        weights = (Path(checkpoint_path) if checkpoint_path
                   else resolve(self.FAMILY, variant, self.runtime, revision=revision))
        self._segmenter = self.VENDOR.Segmenter(weights, device=self.device)
        print(f"{self.DISPLAY} {variant} ready on {self.device} via {self.runtime}.")

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        points: Any = None,
        labels: Any = None,
        boxes: Any = None,
        multimask_output: bool = True,
        name: Union[str, Sequence[str], None] = None,
    ) -> "pf.detections.Detections":
        """Segment whatever the prompt points at.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            points: ``(N, 2)`` or ``(B, N, 2)`` x, y clicks in the image's own pixels.
            labels: ``(N,)`` or ``(B, N)``, 1 for a point to include and 0 for one to exclude.
                Required with *points*; guessing between the two returns a confident mask of the
                wrong thing, so it raises instead.
            boxes: ``(4,)`` or ``(B, 4)`` x1, y1, x2, y2 in the image's own pixels.
            multimask_output: Return three candidate masks per prompt rather than one. Worth
                keeping on for a single click, which is genuinely ambiguous about whether you
                meant the handle, the door or the car.
            name: What to call what you pointed at. One name, or one per prompt. Omitted, every
                detection carries ``class_name=None`` -- the model does not know what it
                segmented and this adapter will not invent it.

        Returns:
            A PixelFlow ``Detections`` with one row per candidate mask, carrying the mask, the
            box that encloses it, and the model's own predicted IoU as the score. With
            ``multimask_output`` on, a single click returns three rows ranked by that score,
            which is the ambiguity rather than a failure to resolve it. ``class_id`` is the index
            of the prompt that produced the row, so a batch of prompts stays separable.

        Raises:
            ValueError: If no prompt is given, if points arrive without labels, or if *name* is a
                sequence whose length is not the number of prompts.
        """
        pixels = load_image(image)
        found = self._segmenter.predict(
            pixels, points=points, labels=labels, boxes=boxes,
            multimask_output=multimask_output,
        )

        prompts, candidates = found.masks.shape[:2]
        names = [name] if isinstance(name, str) else (list(name) if name is not None else None)
        if names is not None and len(names) != prompts:
            raise ValueError(f"{len(names)} names for {prompts} prompt(s); give one each or none")

        # Each prompt's candidates, best first. The model emits them in its own order -- the
        # mask tokens are trained to specialise towards whole, part and subpart -- which means
        # the highest-scoring one is not the first, and ``found[0]`` would otherwise be whichever
        # candidate the model happened to put in slot zero. Every other family in mozo hands back
        # ranked detections, so these are ranked too. Sorted *within* each prompt rather than
        # globally, so a batch stays grouped and ``class_id`` still runs 0, 0, 0, 1, 1, 1.
        order = np.argsort(-found.scores, axis=1, kind="stable")
        rows = (np.arange(prompts)[:, None], order)
        masks = found.masks[rows].reshape(prompts * candidates, *found.masks.shape[2:])
        scores = found.scores[rows].reshape(-1).astype(np.float32)
        # The prompt index, repeated across that prompt's candidates. With one prompt this is all
        # zeros, which is what a single unnamed thing should look like; with several it is what
        # keeps them apart.
        class_ids = np.repeat(np.arange(prompts, dtype=np.int64), candidates)

        # Where ``_yolo`` and ``rfdetr`` release it: at the end of a call, which is a point that
        # runs. ``ModelManager`` holds every model for the life of the process, so a finalizer
        # would fire at interpreter shutdown or never.
        if self.device == "mps":
            torch.mps.empty_cache()

        return pf.detections.from_arrays(
            boxes=bounds(masks),
            scores=scores,
            class_ids=class_ids,
            masks=masks,
            labels=names,
        )
