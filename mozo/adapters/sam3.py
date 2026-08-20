# SPDX-License-Identifier: Apache-2.0
"""mozo's SAM 3 adapter: a phrase and an image in, PixelFlow detections out.

Every other family in mozo answers a fixed question -- "where are the 80 COCO classes", "how far
away is each pixel". SAM 3 answers whatever you ask it, so its ``predict`` takes a *prompt*, and
that one difference propagates: the caller supplies the vocabulary, one concept per call.

That fits mozo's rule that a class name comes from the weights or from the user rather than being
invented. Here it comes from the user, literally -- the phrase you searched for is the name every
detection carries. There is no class list to resolve, so this adapter does not consult
:mod:`mozo.labels` at all.

**The weights are not Apache-2.0.** SAM 3's checkpoints carry Meta's SAM License, which restricts
what they may be used for and binds whoever you serve predictions to. See the NOTICE published
beside them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

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
        text: str,
        threshold: float = 0.5,
    ) -> "pf.detections.Detections":
        """Find every instance of ``text``.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            text: The concept to look for, as a noun phrase -- ``"taxi"``, ``"yellow school
                bus"``. Up to 32 tokens; longer is truncated.
            threshold: Confidence floor.

        Returns:
            A PixelFlow ``Detections`` carrying a box, a mask and a score per instance. Every
            detection's ``class_name`` is ``text``, because the prompt is the class.

        Raises:
            ValueError: If ``text`` is empty. SAM 3 will happily encode the empty string and
                return whatever it finds most salient, which is not what an empty prompt means.
        """
        if not text or not text.strip():
            raise ValueError("SAM 3 needs a concept to look for; text was empty.")

        pixels = load_image(image)
        found = self._segmenter.predict(pixels, text, threshold=threshold)

        # One concept per call, so every instance shares a class, and the prompt names it. The
        # masks are already boolean and already in the source image's pixels, and PixelFlow's
        # framework-free converter detaches and moves tensors itself -- so they go in as they are.
        return pf.detections.from_arrays(
            boxes=found["boxes"],
            scores=found["scores"],
            class_ids=np.zeros(len(found["scores"]), dtype=np.int64),
            masks=found["masks"],
            labels=[text],
        )
