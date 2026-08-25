"""What every YOLO family's adapter does, which is the same thing.

The families differ in their architecture, and that difference lives entirely in their vendored
packages -- each rebuilds its own network from its own checkpoint and supplies its own
letterboxing, coordinate mapping, and suppression where the family needs any. By the time an adapter is involved there is no
model maths left: resolve the weights, choose a runtime, run the vendor's ``detect`` with the
forward pass plugged in, hand the numbers to PixelFlow.

So the vendors are deliberately independent copies of each other and the adapters are not. Two
adapters written out separately came to 149 identical lines of serving-path logic out of 159, with
nothing family-specific in them at all; a third family would have made it three. The vendors have
a fidelity argument for duplication -- their numbers must be reproducible against their own
upstream, and a shared substrate would let one family's refactor move another's boxes. Nothing
here touches a number.

A family subclasses this with four class attributes and its own docstring. It does not override
any method; if one ever needs to, that is the signal the families have actually diverged.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Union

import numpy as np
import torch

from ..image import load_image
from ..labels import resolve as labels_for
from ..runtimes import get_default_device, make_runner, select_runtime
from ..weights import artifacts, resolve

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None


class YOLOPredictor:
    """One loaded variant of one YOLO family, ready to run.

    Subclasses set:

    ``FAMILY``    the name the manifest and the registry use, e.g. ``"yolov8"``.
    ``DISPLAY``   what to call it when talking to a person, e.g. ``"YOLOv8"``.
    ``VENDOR``    the vendored package, which must expose ``Detector`` and ``detect``.
    ``VARIANTS``  the published variant names, the first of which is the default.

    Args:
        variant: A published variant. Defaults to the smallest, which is what every YOLO family
            calls ``nano``.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which published artifact to execute, e.g. ``"torch-fp32"`` or ``"onnx-fp32"``.
            ``"auto"`` takes the best one published for the device. A family that publishes no
            CoreML needs no special case here: ``auto`` only ever chooses among published keys.
        checkpoint_path: A checkpoint of your own instead of the published weights. The variant
            then names nothing at all -- the network is rebuilt from your file -- and ``runtime``
            must be a torch one, since there is no graph for weights mozo has never seen.
        labels: Class names to attach to results, overriding the checkpoint's own.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        runtime: The artifact key actually in use.
        device: The device actually in use.
        imgsz: The square side actually in use.
    """

    FAMILY: str
    DISPLAY: str
    VENDOR: ModuleType
    VARIANTS: tuple[str, ...]

    def __init__(
        self,
        variant: str = "nano",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        labels: list[str] | None = None,
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS and checkpoint_path is None:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()

        if checkpoint_path is None:
            self.runtime = select_runtime(
                self.device, artifacts(self.FAMILY, variant, revision=revision), runtime)
        else:
            # A checkpoint mozo did not publish: the architecture is read from it, the graph is not.
            self.runtime = "torch-fp32"

        self._runner = None
        self._detector = None
        if self.runtime.startswith("torch"):
            weights = (Path(checkpoint_path) if checkpoint_path
                       else resolve(self.FAMILY, variant, self.runtime, revision=revision))
            self._detector = self.VENDOR.Detector(weights, device=self.device)
            self.imgsz = self._detector.imgsz
            names: Any = self._detector.names
        else:
            artifact = resolve(self.FAMILY, variant, self.runtime, revision=revision)
            self._runner = make_runner(artifact, self.runtime, device=self.device)
            # The graph fixes its own input side, so ``imgsz`` is read off it rather than trusted.
            # Letterboxing to any other size would feed the session a shape it cannot accept.
            self.imgsz = self._runner.input_shape[-1]
            # Nothing torch-side is loaded here: the graph carries the architecture, and the class
            # names come from the published labels artifact instead of a checkpoint.
            names = None

        self._labels = labels_for(
            self.FAMILY, variant, caller=labels, checkpoint=names,
            revision=revision, published=checkpoint_path is None,
        )
        if self._labels is None:
            print(
                "[mozo] no class names for this checkpoint. Detections will carry class_id "
                "and class_name=None. Pass labels=[...] to name them."
            )
        print(f"{self.DISPLAY} {variant} ready on {self.device} via {self.runtime} at {self.imgsz}px.")

    def _forward(self, batch: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run the batch through whichever runtime is loaded, returning the raw head output.

        The only step that differs between runtimes. Everything around it is the vendor's.

        A detection graph has one output; a segmentation graph has two, the rows and the
        prototypes those rows carry coefficients of. The vendor's ``detect`` tells them apart by
        type rather than by length, so the lone tensor is handed back bare and the pair as a
        tuple. Wrapping the lone one would hand ``detect`` rows with their batch axis still
        attached, which is plausible, wrong and silent; taking only the first of the pair would
        hand it a tensor whose trailing mask rows it would read as class scores -- wrong classes,
        wrong scores, no masks, nothing raised. The order is the graph's own, which
        ``tools/export/_detection.py`` names ``predictions`` then ``prototypes``.
        """
        if self._runner is None:
            return self._detector.forward(batch)
        outputs = self._runner(batch.numpy())
        if len(outputs) == 1:
            return torch.from_numpy(outputs[0])
        if len(outputs) == 2:
            return tuple(torch.from_numpy(output) for output in outputs)
        raise NotImplementedError(
            f"{self.runtime} for {self.FAMILY}/{self.variant} returned {len(outputs)} outputs; "
            "this seam carries one from a detection graph and two -- rows and prototypes -- from "
            "a segmentation one."
        )

    def predict(
        self,
        image: Union[str, np.ndarray],
        threshold: float = 0.5,
        labels: list[str] | None = None,
    ) -> Any:
        """Detect objects in *image*.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.
            threshold: Minimum confidence to keep.
            labels: Class names for this call only, overriding the adapter's.

        Returns:
            A PixelFlow ``Detections``. A segmentation variant carries a mask per detection as
            well; a detection variant carries none, and the field is absent rather than empty.
        """
        # The vendor wants RGB and mozo's contract already is RGB, so nothing is converted here.
        # Letterboxing and suppression are the vendor's, not restated here, so the answer cannot
        # depend on which of the runtimes ran the forward pass. A family whose head fires once
        # per object has no suppression step, and needs no special case here either. Nor does a
        # segmentation head: the vendor answers with a fourth value that is None unless the
        # checkpoint has a mask branch, so this passes it on without asking which kind it holds.
        # Every vendor answers with four values, the last being masks or None. That is one
        # contract rather than three-plus-one: a seam whose width varies by family has to be
        # measured at the call site, and this is not the only place that reads it -- 
        # ``tools/export/_detection.py`` unpacks the same tuple, and was silently broken for a
        # release by a width it never learned about.
        boxes, scores, class_ids, masks = self.VENDOR.detect(
            load_image(image), self._forward, self.imgsz, threshold)

        if self.device == "mps" and self._runner is None:
            torch.mps.empty_cache()

        return pf.detections.from_arrays(
            boxes=boxes.numpy(),
            scores=scores.numpy(),
            class_ids=class_ids.numpy(),
            labels=labels if labels is not None else self._labels,
            masks=None if masks is None else masks.numpy(),
        )
