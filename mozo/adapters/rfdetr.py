"""RF-DETR detection and instance segmentation, on whichever runtime the host can execute.

The architecture lives in :mod:`mozo.vendors.rfdetr_deploy`, extracted from Roboflow's research
repository and reduced to inference. The weights come from :func:`mozo.weights.resolve`. Which of
them actually runs the forward pass depends on what mozo publishes for the variant and what the
machine can execute -- a torch module, or the same graph exported to ONNX.

Only the middle step changes between runtimes. Pre-processing and post-processing come from the
vendor either way, so the two paths cannot drift: an ONNX artifact is verified at export time to
produce the same detections as the torch model, and it stays that way because nothing downstream
of the graph is reimplemented here.

    >>> model = RFDETRPredictor("small")                      # doctest: +SKIP
    >>> model = RFDETRPredictor("small", runtime="onnx-fp32")  # doctest: +SKIP
    >>> detections = model.predict(image, threshold=0.5)      # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
import torch

from ..device import get_default_device
from ..labels import resolve as labels_for
from ..runtimes import make_runner, select_runtime
from ..vendors.rfdetr_deploy import Predictor, get_spec
from ..vendors.rfdetr_deploy.models.postprocess import PostProcess
from ..weights import artifacts, resolve

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

class _ProcessingOnly(Predictor):
    """The vendor's pre/post-processing without its model, for the graph runtimes.

    ``Predictor.device`` reads it off the model's parameters, which is the one thing a
    model-less predictor cannot answer. It is always the CPU here: a graph is fed CPU numpy
    whatever device ends up executing it.
    """

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")


class RFDETRPredictor:
    """One loaded RF-DETR variant, ready to run.

    Args:
        variant: A published variant -- ``nano``, ``small``, ``medium``, ``large``, or their
            ``seg-`` counterparts.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which published artifact to execute -- ``"torch-fp32"``, ``"onnx-fp32"`` or
            ``"coreml-fp32"``. ``"auto"`` takes the best one published for the device.
        checkpoint_path: A checkpoint of your own, instead of the published weights. The variant
            then names the architecture to build, and ``runtime`` must be a torch one -- there is
            no ONNX graph for weights mozo has never seen.
        labels: Class names to attach to results, overriding the checkpoint's own.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        runtime: The artifact key actually in use.
        device: The device actually in use.

    Examples:
        >>> RFDETRPredictor("small").runtime                       # doctest: +SKIP
        'onnx-fp32'
        >>> RFDETRPredictor("small", runtime="torch-fp32").runtime  # doctest: +SKIP
        'torch-fp32'
    """

    VARIANTS = (
        "nano", "small", "medium", "large",
        "seg-nano", "seg-small", "seg-medium", "seg-large",
    )

    def __init__(
        self,
        variant: str = "medium",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        labels: list[str] | None = None,
        revision: str | None = None,
        **_ignored: Any,
    ) -> None:
        if variant not in self.VARIANTS and checkpoint_path is None:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()

        if checkpoint_path is None:
            self.runtime = select_runtime(
                self.device, artifacts("rfdetr", variant, revision=revision), runtime)
        else:
            # A checkpoint mozo did not publish: the architecture is known, the graph is not.
            self.runtime = "torch-fp32"

        self._runner = None
        if not self.runtime.startswith("torch"):
            artifact = resolve("rfdetr", variant, self.runtime, revision=revision)
            self._runner = make_runner(artifact, self.runtime, device=self.device)
            # The graph carries the architecture, so nothing torch-side is loaded here. Only the
            # spec is needed, and it costs no weights: pre- and post-processing are pure maths
            # over the variant's resolution and query count. Loading the checkpoint anyway would
            # cost a second, larger download and 128 MB of parameters that never run.
            self._predictor = self._build_processing_only()
        else:
            weights = (Path(checkpoint_path) if checkpoint_path
                       else resolve("rfdetr", variant, self.runtime, revision=revision))
            self._predictor = Predictor.from_pretrained(
                f"rfdetr-{variant}", weights=weights, device=self.device)
            if self.runtime.endswith("-fp16"):
                # The published fp16 checkpoint is a bandwidth saving; loading it into an fp32
                # model upcasts it back, so the cast is what actually makes the model half.
                self._predictor.model.half()

        self._labels = labels_for(
            "rfdetr", variant, caller=labels, checkpoint=self._predictor.class_names,
            revision=revision, published=checkpoint_path is None,
        )
        if self._labels is None:
            print(
                "[mozo] no class names for this checkpoint. Detections will carry class_id "
                "and class_name=None. Pass labels=[...] to name them."
            )
        print(f"RF-DETR {variant} ready on {self.device} via {self.runtime}.")

    def _build_processing_only(self) -> Predictor:
        """Build the vendor's pre/post-processing without its model.

        The predictor is used for :meth:`preprocess` and :attr:`postprocess` alone on the ONNX
        path, and neither touches ``model``. Its ``device`` property would, but nothing asks:
        an ONNX batch is always handed over as CPU numpy.
        """
        spec = get_spec(f"rfdetr-{self.variant}")
        return _ProcessingOnly(
            spec=spec,
            model=None,
            postprocess=PostProcess(
                num_select=spec.num_select,
                num_keypoints_per_class=list(spec.num_keypoints_per_class),
                trace_alpha=spec.postprocess_trace_alpha,
            ),
            class_names=[],
        )

    def _forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the batch through whichever runtime is loaded, returning named raw outputs."""
        if self._runner is None:
            with torch.inference_mode():
                return self._predictor.model(batch)
        # The exporters name each graph's outputs after the keys the post-processor reads, so
        # the mapping travels with the artifact instead of being restated here.
        raw = self._runner(batch.numpy())
        return {name: torch.from_numpy(array) for name, array in zip(self._runner.outputs, raw)}

    def predict(
        self,
        image: Union[str, np.ndarray],
        threshold: float = 0.5,
        labels: list[str] | None = None,
    ) -> Any:
        """Detect objects in *image*.

        Args:
            image: A file path, or an ``HWC`` BGR array as OpenCV produces.
            threshold: Minimum confidence to keep.
            labels: Class names for this call only, overriding the adapter's.

        Returns:
            A PixelFlow ``Detections``. Segmentation variants carry masks as well as boxes.
        """
        # cv2 hands over BGR; the vendor's preprocessing expects RGB. ``ascontiguousarray`` is
        # required, not cosmetic: the reversed view has a negative stride and torch rejects it.
        source = np.ascontiguousarray(image[..., ::-1]) if isinstance(image, np.ndarray) else image
        batch, sizes = self._predictor.preprocess([source])
        if self._runner is None:
            batch = batch.to(dtype=next(self._predictor.model.parameters()).dtype)

        outputs = self._forward(batch)
        # Post-processing is numerically delicate (top-k over near-tied scores), so it runs at
        # full width whatever the forward pass used. The torch path also returns auxiliary
        # decoder outputs as lists, which post-processing ignores and which cannot be cast.
        outputs = {name: value.float() if torch.is_tensor(value) else value
                   for name, value in outputs.items()}
        results = self._predictor.postprocess(
            outputs, target_sizes=torch.tensor(sizes, device=batch.device), score_threshold=threshold
        )
        result = results[0]
        keep = result["scores"] > threshold
        if not bool(keep.all()):
            # The mask path is already filtered inside postprocess, precisely so masks below
            # threshold are never resized; re-indexing there would copy every full-size mask.
            result = {name: value[keep] for name, value in result.items()}
        masks = result.get("masks")
        if masks is not None and masks.ndim == 4:
            # The vendor emits (N, 1, H, W); every pixelflow converter produces (N, H, W), and
            # the annotators index masks that way.
            masks = masks[:, 0]

        if self.device == "mps" and self._runner is None:
            torch.mps.empty_cache()

        return pf.detections.from_arrays(
            boxes=result["boxes"],
            scores=result["scores"],
            class_ids=result["labels"],
            masks=masks,
            labels=labels if labels is not None else self._labels,
        )
