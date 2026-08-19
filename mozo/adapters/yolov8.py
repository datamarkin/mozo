"""YOLOv8 detection, on whichever runtime the host can execute.

The architecture lives in :mod:`mozo.vendors.yolov8_deploy`, which rebuilds the network from what
the checkpoint itself records rather than from any model definition. The weights come from
:func:`mozo.weights.resolve`. Which of them runs the forward pass depends on what mozo publishes
for the variant and what the machine can execute -- a torch module, or the same graph as ONNX.

Only the middle step changes between runtimes. Letterboxing, non-maximum suppression and the
mapping back to source pixels come from the vendor either way, so the two paths cannot drift.

**These weights are AGPL-3.0.** mozo's code is Apache-2.0; the two are separate works travelling
together. Serving predictions from them over a network puts AGPL-3.0 section 13 obligations on
whoever runs the service. The NOTICE published beside every checkpoint says so in full.

    >>> model = YOLOv8Predictor("nano")                        # doctest: +SKIP
    >>> model = YOLOv8Predictor("nano", runtime="onnx-fp32")   # doctest: +SKIP
    >>> detections = model.predict(image, threshold=0.25)      # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from ..image import load_image
from ..labels import resolve as labels_for
from ..runtimes import get_default_device, make_runner, select_runtime
from ..vendors.yolov8_deploy import Detector, detect
from ..weights import artifacts, resolve

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None


class YOLOv8Predictor:
    """One loaded YOLOv8 variant, ready to run.

    Args:
        variant: A published variant -- ``nano``, ``small``, ``medium``, ``large`` or ``xlarge``.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which published artifact to execute -- ``"torch-fp32"`` or ``"onnx-fp32"``.
            ``"auto"`` takes the best one published for the device.
        checkpoint_path: A checkpoint of your own instead of the published weights. The variant
            then names nothing at all -- the network is rebuilt from your file -- and ``runtime``
            must be a torch one, since there is no ONNX graph for weights mozo has never seen.
        labels: Class names to attach to results, overriding the checkpoint's own.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        runtime: The artifact key actually in use.
        device: The device actually in use.
        imgsz: The square side actually in use.
    """

    VARIANTS = ("nano", "small", "medium", "large", "xlarge")

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
                self.device, artifacts("yolov8", variant, revision=revision), runtime)
        else:
            # A checkpoint mozo did not publish: the architecture is read from it, the graph is not.
            self.runtime = "torch-fp32"

        self._runner = None
        self._detector = None
        if self.runtime.startswith("torch"):
            weights = (Path(checkpoint_path) if checkpoint_path
                       else resolve("yolov8", variant, self.runtime, revision=revision))
            self._detector = Detector(weights, device=self.device)
            self.imgsz = self._detector.imgsz
            names: Any = self._detector.names
        else:
            artifact = resolve("yolov8", variant, self.runtime, revision=revision)
            self._runner = make_runner(artifact, self.runtime, device=self.device)
            # The graph fixes its own input side, so ``imgsz`` is read off it rather than trusted.
            # Letterboxing to any other size would feed the session a shape it cannot accept.
            self.imgsz = self._runner.input_shape[-1]
            # Nothing torch-side is loaded here: the graph carries the architecture, and the class
            # names come from the published labels artifact instead of a checkpoint.
            names = None

        self._labels = labels_for(
            "yolov8", variant, caller=labels, checkpoint=names,
            revision=revision, published=checkpoint_path is None,
        )
        if self._labels is None:
            print(
                "[mozo] no class names for this checkpoint. Detections will carry class_id "
                "and class_name=None. Pass labels=[...] to name them."
            )
        print(f"YOLOv8 {variant} ready on {self.device} via {self.runtime} at {self.imgsz}px.")

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Run the batch through whichever runtime is loaded, returning the raw head output.

        The only step that differs between runtimes. Everything around it is the vendor's, via
        :func:`~mozo.vendors.yolov8_deploy.detect`.
        """
        if self._runner is None:
            return self._detector.forward(batch)
        return torch.from_numpy(self._runner(batch.numpy())[0])

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
            A PixelFlow ``Detections``.
        """
        # The vendor wants RGB and mozo's contract already is RGB, so nothing is converted here.
        # Letterboxing and suppression are the vendor's, not restated here, so the answer cannot
        # depend on which of the two runtimes ran the forward pass.
        boxes, scores, class_ids = detect(load_image(image), self._forward, self.imgsz, threshold)

        if self.device == "mps" and self._runner is None:
            torch.mps.empty_cache()

        return pf.detections.from_arrays(
            boxes=boxes.numpy(),
            scores=scores.numpy(),
            class_ids=class_ids.numpy(),
            labels=labels if labels is not None else self._labels,
        )
