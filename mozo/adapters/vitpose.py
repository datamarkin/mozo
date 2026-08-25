"""ViTPose++ pose estimation: person boxes in, the same detections back with joints attached.

The architecture lives in :mod:`mozo.vendors.vitpose_deploy`, extracted from HuggingFace's PyTorch
implementation and reduced to inference. The weights come from :func:`mozo.weights.resolve`.

**This is the one family in mozo that takes detections as an argument.** ViTPose is top-down: it
does not find people, it is told where they are. Pair it with anything that produces boxes.

    >>> people = RFDETRPredictor("medium").predict(frame)              # doctest: +SKIP
    >>> posed = ViTPosePredictor("base").predict(frame, people)        # doctest: +SKIP
    >>> posed[0].keypoints[0].name                                     # doctest: +SKIP
    'nose'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

from ..labels import resolve as labels_for
from ..runtimes import get_default_device, make_runner, select_runtime
from ..image import load_image
from ..vendors.vitpose_deploy import Predictor
from ..weights import artifacts, resolve

try:
    import pixelflow as pf
except ImportError:  # pragma: no cover - depends on the install
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None


def _joint_names(labels: Any) -> list[str]:
    """The joint names in *labels*, or ``[]`` when it carries none.

    Two shapes are accepted, because two things supply them. The published ``labels.json`` is
    PixelFlow's rich format -- ``[{"id", "name", "keypoints"}]`` -- and this model has exactly one
    category, so the joint list is the one entry that has them. A caller passing ``labels=`` is
    naming *joints*, so a plain list of seventeen strings is taken as exactly that.

    Deliberately not routed through ``get_label_info``, which reaches a joint vocabulary by
    matching a class id. The ids here would be the incoming detector's -- person is 1 under COCO's
    original ids and 0 under a contiguous one -- and a vocabulary reached through the wrong id
    silently returns no names at all.
    """
    if not labels:
        return []
    if isinstance(labels[0], dict):
        # The first category that names joints, not every category's concatenated: a vocabulary
        # with two of them describes two heads, and this checkpoint has one.
        return next(([joint["name"] for joint in entry["keypoints"]]
                     for entry in labels if entry.get("keypoints")), [])
    return [str(name) for name in labels]


class ViTPosePredictor:
    """One loaded ViTPose++ variant, ready to run.

    Args:
        variant: A published variant -- ``small``, ``base``, ``large`` or ``huge``.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which published artifact to execute -- ``"torch-fp32"`` or ``"onnx-fp32"``.
            ``"auto"`` takes the best one published for the device.
        checkpoint_path: A checkpoint of your own, instead of the published weights. The variant
            then names the architecture to build, and the runtime is torch -- there is no ONNX
            graph for weights mozo has never seen.
        labels: Joint names to attach to results, overriding the published vocabulary. Either a
            plain list of names in the model's own joint order, or the published ``labels.json``
            shape.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        runtime: The artifact key actually in use.
        device: The device actually in use.
    """

    VARIANTS = ("small", "base", "large", "huge")

    def __init__(
        self,
        variant: str = "base",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        labels: list | None = None,
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS and checkpoint_path is None:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()

        if checkpoint_path is None:
            self.runtime = select_runtime(
                self.device, artifacts("vitpose", variant, revision=revision), runtime)
        else:
            # A checkpoint mozo did not publish: the architecture is known, the graph is not.
            self.runtime = "torch-fp32"

        artifact = (Path(checkpoint_path) if checkpoint_path
                    else resolve("vitpose", variant, self.runtime, revision=revision))
        if self.runtime.startswith("torch"):
            self._predictor = Predictor(artifact, variant, device=self.device)
        else:
            # The graph carries the architecture, so nothing torch-side is loaded here -- and the
            # vendor still owns both ends of the call. Its ``forward`` seam is all that changes,
            # which is what stops an exported artifact from drifting: the crop, the affine and the
            # decode are the same code either way.
            runner = make_runner(artifact, self.runtime, device=self.device)
            self._predictor = Predictor(
                None, variant, device=self.device,
                forward=lambda batch: runner(batch.numpy())[0])
        self._labels = labels_for(
            "vitpose", variant, caller=labels, revision=revision,
            published=checkpoint_path is None,
        )
        if self._labels is None:
            print(
                "[mozo] no joint names for this checkpoint. Keypoints will carry an id and "
                "name=None. Pass labels=[...] to name them."
            )
        print(f"ViTPose {variant} ready on {self.device} via {self.runtime}.")

    def predict(
        self,
        image: Union[str, np.ndarray],
        detections: Any,
        labels: list | None = None,
    ) -> Any:
        """Find the joints of everyone in *detections*.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array. The **whole frame**, not
                a crop: the model's own cropping reaches outside each box, so a tight crop has
                already discarded pixels it wants.
            detections: A PixelFlow ``Detections`` saying where the people are. Every row is
                posed -- this does not filter, because which boxes are people is the caller's
                fact, not the model's. Filter first if the boxes came from a general detector::

                    posed = model.predict(frame, found.filter_by_class_id(1))

            labels: Joint names for this call only, overriding the adapter's. A plain list of
                names, in the model's own joint order.

        Returns:
            A copy of *detections* with ``keypoints`` set on every row: the same boxes, class ids,
            scores, names and tracker ids, plus 17 joints each. Not a new set of detections -- this
            model annotates the ones it was given.

            A joint the model cannot see comes back with a confidence near zero and coordinates
            that mean nothing, so filter on the confidence before reading a position.
        """
        boxes = [row.bbox for row in detections]
        joints = self._predictor.predict(load_image(image), np.array(boxes, dtype=np.float64))
        names = _joint_names(labels if labels is not None else self._labels)

        # The rows handed back are the caller's own, copied and annotated, which is what keeps
        # tracker ids, zone membership and the detector's class names intact. Building fresh ones
        # through ``from_arrays`` would need a class id and a score this model does not have, and
        # would drop everything it was not told to carry.
        #
        # A keypoint's id is its position, which is the convention ``from_arrays`` uses for every
        # other family. An unnamed joint keeps its slot rather than being skipped: the index *is*
        # the joint's identity, so renumbering around a missing name would rename the rest.
        found = detections.__class__()
        for row, person in zip(detections, joints):
            annotated = row.copy()
            annotated.keypoints = [
                pf.KeyPoint(x=float(x), y=float(y), id=index,
                            name=names[index] if index < len(names) else None,
                            confidence=float(score))
                for index, (x, y, score) in enumerate(person)
            ]
            found.add_detection(annotated)
        return found
