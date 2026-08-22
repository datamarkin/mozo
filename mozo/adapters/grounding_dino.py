# SPDX-License-Identifier: Apache-2.0
"""mozo's Grounding DINO adapter: phrases and an image in, PixelFlow detections out.

The second open-vocabulary family here, and it answers the same question OWLv2 does through the
same endpoint: name a thing in words, get boxes for it, with no class list and no training. The
two are substitutable, which is the whole reason ``open_vocabulary_detection`` is one task rather
than one per family.

Where it differs from OWLv2 is inside. OWLv2 embeds each phrase separately and compares them to
patch embeddings; Grounding DINO fuses text into the image features six times over and lets the
decoder attend back to the words, which is why it reads a phrase like ``"the man on the left"``
rather than treating it as a bag of words. It costs more to run and it is the one to reach for
when the prompt is a description rather than a noun.

**Its weights are Apache-2.0**, as the authors state on the HuggingFace repositories that serve
them. The GitHub project carries the licence for the code and says nothing about the checkpoints;
see the ``NOTICE`` published beside them.

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
from ..vendors.grounding_dino_deploy import SPECS, VARIANTS, Predictor
from ..weights import artifacts, resolve

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["GroundingDinoPredictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "grounding_dino"


class GroundingDinoPredictor:
    """Open-vocabulary detection: describe a thing, get boxes for it.

    Args:
        variant: Which published model. ``tiny`` is Swin-T and 82% of upstream's own downloads;
            ``base`` is Swin-B and 8.3 box AP better on COCO zero-shot, for 35% more weights.
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one.
        checkpoint_path: Your own checkpoint, instead of the published weights.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.

    Examples:
        >>> model = GroundingDinoPredictor()                            # doctest: +SKIP
        >>> found = model.predict("street.jpg", "taxi")                 # doctest: +SKIP
        >>> found[0].class_name                                         # doctest: +SKIP
        'taxi'
        >>> many = model.predict("street.jpg", ["taxi", "a cyclist"])   # doctest: +SKIP
    """

    #: Upstream publishes exactly two checkpoints. See ``PROVENANCE.md``.
    VARIANTS = list(VARIANTS)

    #: Which runtimes this adapter can execute. Only torch: the input size is not fixed -- the
    #: image is resized rather than letterboxed, so its height and width depend on the
    #: photograph -- and a graph exported at one shape cannot take another. Declared rather than
    #: checked afterwards so ``auto`` cannot pick an artifact this would then have to refuse.
    EXECUTES = ("torch",)

    def __init__(
        self,
        variant: str = "tiny",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        # Checked even with your own checkpoint: the variant selects the backbone geometry here,
        # so a wrong one is not a mislabelling but a strict load that cannot succeed.
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
        self._predictor = Predictor(weights, SPECS[variant], device=self.device)
        print(f"Grounding DINO {variant} ready on {self.device} via {self.runtime}.")

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        text: Union[str, Sequence[str]],
        threshold: float | None = None,
    ) -> "pf.detections.Detections":
        """Find every instance of each phrase in ``text``.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            text: One phrase -- ``"taxi"``, ``"a yellow school bus"`` -- or several, as a
                sequence. They are joined into the single caption the model was trained on, so
                asking for several costs one forward pass rather than several. A phrase may not
                contain ``.`` or ``?``: those separate concepts, and a prompt carrying one would
                be silently split.
            threshold: Confidence floor. Omitted, upstream's published default of 0.35 applies.
                Grounding DINO's scores run higher than OWLv2's, whose default is 0.1 -- the two
                numbers are not comparable and neither is restated at the endpoint.

        Returns:
            A PixelFlow ``Detections`` carrying a box and a score per instance, with
            ``class_name`` naming the phrase that found it and ``class_id`` its index in
            ``text``, ordered best first.

            The name is the phrase you passed, not a span decoded from the model's tokens.
            Upstream returns the latter, which can hand back ``"yellow school"`` for
            ``"a yellow school bus"``; see ``PROVENANCE.md`` for why this does not.

            Nothing is suppressed. Two prompts that describe the same thing can both find it,
            and overlapping boxes for one prompt are the model's answer rather than a failure to
            deduplicate.

        Raises:
            ValueError: If no phrase is given, if one is empty, if a phrase contains a separator,
                or if the phrases together exceed the model's 256-token budget.
        """
        prompts = [text] if isinstance(text, str) else list(text)
        if not prompts:
            raise ValueError("Grounding DINO needs a phrase to look for; no text was given.")
        if any(not phrase.strip() for phrase in prompts):
            raise ValueError("Grounding DINO needs a phrase to look for; text was empty.")

        found = self._predictor(load_image(image), prompts, box_threshold=threshold)

        # Ranked here rather than in the vendor, which returns queries in the model's own order
        # -- the right one to verify against -- while every other family in mozo hands back
        # detections best first.
        found.sort(key=lambda d: d.score, reverse=True)

        return pf.detections.from_arrays(
            boxes=np.array([d.box for d in found], dtype=np.float32).reshape(-1, 4),
            scores=np.array([d.score for d in found], dtype=np.float32),
            class_ids=np.array([d.prompt_index for d in found], dtype=np.int64),
            labels=prompts,
        )
