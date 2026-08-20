# SPDX-License-Identifier: Apache-2.0
"""mozo's SAM 3 adapter: a phrase and an image in, PixelFlow detections out.

Every other family in mozo answers a fixed question -- "where are the 80 COCO classes", "how far
away is each pixel". SAM 3 answers whatever you ask it, so its ``predict`` takes a *prompt*, and
that one difference propagates: the caller supplies the vocabulary, one concept or several.

That fits mozo's rule that a class name comes from the weights or from the user rather than being
invented. Here it comes from the user, literally -- the phrase you searched for is the name every
detection carries. There is no class list to resolve, so this adapter does not consult
:mod:`mozo.labels` at all.

**The weights are not Apache-2.0.** SAM 3's checkpoints carry Meta's SAM License, which restricts
what they may be used for and binds whoever you serve predictions to. See the NOTICE published
beside them.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np
import torch

from ..image import load_image
from ..runtimes import get_default_device, select_runtime
from ..vendors.sam3_deploy import Segmenter
from ..weights import artifacts, resolve

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["Sam3Predictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "sam3"


class Sam3Predictor:
    """Concept segmentation: name a thing, get every instance of it.

    Args:
        variant: Which published model. Meta ships one, so there is one name.
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one.
        checkpoint_path: Your own checkpoint, instead of the published weights.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.

    Examples:
        >>> model = Sam3Predictor()                            # doctest: +SKIP
        >>> found = model.predict("street.jpg", "taxi")        # doctest: +SKIP
        >>> found[0].class_name                                # doctest: +SKIP
        'taxi'
        >>> many = model.predict("street.jpg", ["taxi", "cyclist"])   # doctest: +SKIP
    """

    #: Meta publishes a single SAM 3 rather than a size ladder, so there is no choice to make.
    VARIANTS = ["sam3"]

    def __init__(
        self,
        variant: str = "sam3",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        # Checked even with your own checkpoint: unlike families where the variant selects an
        # architecture, here it names a published model, and a wrong one would be reported back
        # by ``self.variant`` as though it meant something.
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        # What is runnable is the manifest's to declare. ``select_runtime`` reads it and raises
        # with the available names, so there is nothing for this adapter to restate.
        self.runtime = (
            "torch-fp32" if checkpoint_path is not None
            else select_runtime(self.device, artifacts(FAMILY, variant, revision=revision), runtime)
        )

        weights = (Path(checkpoint_path) if checkpoint_path
                   else resolve(FAMILY, variant, self.runtime, revision=revision))
        self._segmenter = Segmenter(weights, device=self.device)
        print(f"SAM 3 ready on {self.device} via {self.runtime}.")

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        text: Union[str, Sequence[str]],
        threshold: float = 0.5,
    ) -> "pf.detections.Detections":
        """Find every instance of each concept in ``text``.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            text: One concept as a noun phrase -- ``"taxi"``, ``"yellow school bus"`` -- or
                several, as a sequence. Each is up to 32 tokens; longer is truncated. Several
                prompts share one image encode but each pays its own decode.
            threshold: Confidence floor.

        Returns:
            A PixelFlow ``Detections`` carrying a box, a mask and a score per instance, with
            ``class_name`` naming the prompt that found it. With several prompts this is one
            result carrying several classes rather than one result per prompt.

            Instances found by different prompts may overlap: ask for ``"car"`` and
            ``"vehicle"`` and the same car comes back twice, under each name. That is what was
            asked for, so nothing here suppresses it.

        Raises:
            ValueError: If any concept is empty, or none is given. SAM 3 will happily encode the
                empty string and return whatever it finds most salient, which is not what an
                empty prompt means.
        """
        prompts = [text] if isinstance(text, str) else list(text)
        if not prompts:
            raise ValueError("SAM 3 needs a concept to look for; no text was given.")
        if any(not p.strip() for p in prompts):
            raise ValueError("SAM 3 needs a concept to look for; text was empty.")

        pixels = load_image(image)

        # One decode per prompt, because the head takes one prompt's features against one
        # image's -- there is no batched form to reach for. The encode is what they share, and
        # it is the expensive half, so this is the cheap direction of that trade.
        boxes, scores, masks, class_ids = [], [], [], []
        for index, prompt in enumerate(prompts):
            found = self._segmenter.predict(pixels, prompt, threshold=threshold)
            boxes.append(found["boxes"])
            scores.append(found["scores"])
            masks.append(found["masks"])
            class_ids.append(np.full(len(found["scores"]), index, dtype=np.int64))

        # The prompt list *is* the vocabulary, so class_ids index it. With one prompt this is
        # the single-class result it has always been; with several it is what class_ids are for.
        # Masks are already boolean and already in the source image's pixels, and PixelFlow's
        # converter detaches and moves tensors itself, so they go in as they are.
        # ``torch.cat`` allocates even for a one-element list, and one prompt is still the
        # common case -- on a 2 MP image that is a pointless copy of the whole mask stack.
        def joined(parts):
            return parts[0] if len(parts) == 1 else torch.cat(parts)

        return pf.detections.from_arrays(
            boxes=joined(boxes),
            scores=joined(scores),
            class_ids=class_ids[0] if len(class_ids) == 1 else np.concatenate(class_ids),
            masks=joined(masks),
            labels=prompts,
        )
