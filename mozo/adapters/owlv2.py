# SPDX-License-Identifier: Apache-2.0
"""mozo's OWLv2 adapter: phrases and an image in, PixelFlow detections out.

Most families here answer a fixed question -- "where are the 80 COCO classes", "how far away is
each pixel". OWLv2 answers whatever you ask it, so its ``predict`` takes a *vocabulary*, and the
class names come from the caller rather than from a ``labels.json`` beside the weights. This
adapter does not consult :mod:`mozo.labels` at all.

**Its weights are Apache-2.0.** That is the point of it. mozo's other text-prompted family, SAM 3,
carries Meta's SAM License, which restricts what its predictions may be used for and binds whoever
you serve them to. OWLv2 is Apache-2.0 on the code and on all four published checkpoints, so this
is the open-vocabulary path with nothing attached.

**Boxes only, no masks.** Pair it with SAM 2 or EdgeTAM if you want a mask: feed a box from here
in as a prompt there.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np

from ..image import load_image
from ..runtimes import get_default_device, select_runtime
from ..vendors.owlv2_deploy import Detector
from ..vendors.owlv2_deploy.predictor import THRESHOLD
from ..weights import artifacts, resolve

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["OwlV2Predictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "owlv2"


class OwlV2Predictor:
    """Open-vocabulary detection: name a thing, get boxes for it.

    Args:
        variant: Which published model. ``-ensemble`` averages the self-trained and fine-tuned
            checkpoints and is the one the paper reports; the plain ones are self-training only.
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one.
        checkpoint_path: Your own checkpoint, instead of the published weights.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.

    Examples:
        >>> model = OwlV2Predictor()                                  # doctest: +SKIP
        >>> found = model.predict("street.jpg", "taxi")               # doctest: +SKIP
        >>> found[0].class_name                                       # doctest: +SKIP
        'taxi'
        >>> many = model.predict("street.jpg", ["taxi", "cyclist"])   # doctest: +SKIP
    """

    #: Google publishes six; these are the four anyone uses. See ``PROVENANCE.md``.
    VARIANTS = ["base-ensemble", "base", "large-ensemble", "large"]

    #: Which runtimes this adapter can execute. Only torch, because the vendor builds a torch
    #: module from a checkpoint and has no graph path at all. Declared rather than checked
    #: afterwards so that ``auto`` cannot pick an artifact this would then have to refuse --
    #: inert while OWLv2 publishes only torch, and the line that keeps ``auto`` correct if it
    #: ever publishes more. (``PROVENANCE.md`` covers a different question: why no graph is
    #: published, which is that the one that traced ran twice as slow.)
    EXECUTES = ("torch",)

    def __init__(
        self,
        variant: str = "base-ensemble",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        # Checked even with your own checkpoint: the variant selects a geometry here, so a wrong
        # one is not a mislabelling but a strict load that cannot succeed.
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        # What is runnable is the manifest's to declare, and what is *executable* is this
        # adapter's. ``select_runtime`` reads both and raises with the available names.
        self.runtime = (
            "torch-fp32" if checkpoint_path is not None
            else select_runtime(
                self.device,
                artifacts(FAMILY, variant, revision=revision),
                runtime,
                executes=self.EXECUTES,
            )
        )

        weights = (Path(checkpoint_path) if checkpoint_path
                   else resolve(FAMILY, variant, self.runtime, revision=revision))
        self._detector = Detector(weights, variant, device=self.device)
        print(f"OWLv2 {variant} ready on {self.device} via {self.runtime}.")

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        text: Union[str, Sequence[str]],
        threshold: float = THRESHOLD,
    ) -> "pf.detections.Detections":
        """Find every instance of each phrase in ``text``.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            text: One phrase -- ``"taxi"``, ``"yellow school bus"`` -- or several, as a sequence.
                Each is up to 16 tokens; longer is truncated. All of them share one image encode
                *and* one text encode, so asking for twenty phrases costs barely more than one.
                Passed to the model verbatim: OWLv2's own examples wrap phrases as ``"a photo of
                a cat"``, but that is your wording to choose, not a template this applies.
            threshold: Confidence floor. Upstream's default is 0.1 and OWLv2's scores run lower
                than a closed-vocabulary detector's -- 0.3 is a confident detection here.

        Returns:
            A PixelFlow ``Detections`` carrying a box and a score per instance, with
            ``class_name`` naming the phrase that found it, ordered best first.

            Every candidate is scored against every phrase and keeps only its best one, so the
            same box cannot come back twice under two names -- ask for ``"car"`` and
            ``"vehicle"`` and each detection picks a side. That is upstream's behaviour.

        Raises:
            ValueError: If any phrase is empty, or none is given.
        """
        prompts = [text] if isinstance(text, str) else list(text)
        if not prompts:
            raise ValueError("OWLv2 needs a phrase to look for; no text was given.")
        if any(not phrase.strip() for phrase in prompts):
            raise ValueError("OWLv2 needs a phrase to look for; text was empty.")

        found = self._detector.predict(load_image(image), prompts, threshold=threshold)

        # Ranked here rather than in the vendor, because the vendor returns candidates in patch
        # order -- which is the model's order and the right one to verify against -- while every
        # other family in mozo hands back detections best first.
        order = found.scores.argsort(descending=True)
        return pf.detections.from_arrays(
            boxes=found.boxes[order],
            scores=found.scores[order],
            class_ids=found.labels[order],
            labels=prompts,
        )
