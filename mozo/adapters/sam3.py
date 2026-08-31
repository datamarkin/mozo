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
from ..runtimes import CoreMLRunner, get_default_device, select_runtime
from ..vendors.sam3_deploy import Segmenter
from ..weights import artifacts, framework_of, resolve

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["GraphVision", "Sam3Predictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "sam3"

#: The concept pyramid's levels, finest first, as the exported graph names its outputs. The
#: order is the contract: :class:`~mozo.vendors.sam3_deploy.grounding.concept.ConceptHead` reads
#: ``levels[-1]`` as the grid it attends over and the rest as mask features, so a graph that
#: returned them the other way round would segment confidently and wrongly.
LEVELS = ("level0", "level1", "level2")


class GraphVision:
    """A graph artifact standing in for SAM 3's torch vision encoder.

    :class:`~mozo.vendors.sam3_deploy.predictor.Segmenter` asks its encoder one question -- a
    preprocessed batch in, the concept pyramid and its position encoding out -- and this answers
    it from a CoreML package instead of from the trunk. That is the whole of what the graph
    replaces; the text tower and the concept head stay in torch, so the checkpoint is still
    loaded either way.

    Args:
        runner: The loaded package.
        device: Where the torch half runs, and so where the outputs must land.

    Raises:
        ValueError: If asked for the click stack. The published graph carries the concept stack
            alone, and the click path reads a differently preprocessed image -- see
            :meth:`~mozo.vendors.sam3_deploy.predictor.Segmenter.encode_click`. Refusing here
            rather than returning the concept pyramid is the difference between an error and a
            mask of the wrong pixels.
    """

    def __init__(self, runner: CoreMLRunner, device: str) -> None:
        self.runner = runner
        self.device = device

    def __call__(self, batch: torch.Tensor, stacks: tuple[str, ...] = ("concept",)) -> dict:
        if tuple(stacks) != ("concept",):
            raise ValueError(
                f"the SAM 3 graph encoder serves the concept stack only, not {stacks!r}"
            )
        got = dict(zip(self.runner.outputs, self.runner(batch.cpu().numpy())))
        return {
            "concept": [torch.from_numpy(got[name]).to(self.device) for name in LEVELS],
            "positions": torch.from_numpy(got["positions"]).to(self.device),
        }


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

        # The graph covers the vision encoder and nothing else, so the checkpoint is resolved
        # whichever runtime won -- the text tower and the concept head come out of it either
        # way. What the graph changes is that its 1.85 GB of trunk is then never loaded.
        weights = (Path(checkpoint_path) if checkpoint_path
                   else resolve(FAMILY, variant, "torch-fp32", revision=revision))
        vision = None
        if framework_of(self.runtime) == "coreml":
            package = resolve(FAMILY, variant, self.runtime, revision=revision)
            vision = GraphVision(CoreMLRunner(package), self.device)
        self._segmenter = Segmenter(weights, device=self.device, vision=vision)
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
