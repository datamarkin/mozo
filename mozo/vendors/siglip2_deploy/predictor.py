# SPDX-License-Identifier: Apache-2.0
"""The seam: images or phrases in, vectors out -- and the two compared when you want an answer.

**Each tower, and the tokenizer, is built on first use and never before.** An ingest job that only calls
:meth:`Encoder.encode_image` never allocates the text tower, which for SigLIP 2 is most of the
file: Gemma's 256,000-piece vocabulary is 786 MB of a ``base`` checkpoint and 1,180 MB of an
``so400m`` one. A query service that only calls :meth:`Encoder.encode_text` never allocates the
image tower. Neither needs configuring; it falls out of building lazily.

Construction is locked **per part**, not per encoder. :class:`~mozo.manager.ModelManager` holds its
own lock across building the *adapter* and nowhere else, and ``mozo.server`` runs handlers in a
threadpool -- so two concurrent image encodes on one cached encoder would otherwise both find the
tower missing and both build it. One lock across all four parts would fix that and introduce a
worse problem: a request that needs only the text tower would wait out an image tower being built
for somebody else, and a cold ``encode_text`` would block on the 870 ms tokenizer build it is not
using. Threads only queue behind each other for the same part.

**Vectors leave L2-normalised**, so a dot product between any two of them is a cosine similarity and
there is no convention for a caller to get wrong.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Sequence

import numpy as np
import torch

from .checkpoint import load_scoring, load_text_tower, load_vision_tower
from .config import Spec
from .image import preprocess
from .network import normalise
from .text.tokenizer import Tokenizer

__all__ = ["Encoder"]


class Encoder:
    """One loaded SigLIP 2 variant, with each tower built only if it is asked for.

    Args:
        weights: Path to a published checkpoint.
        spec: Which variant it is.
        device: Where to run.

    Attributes:
        spec: The variant's geometry.
        device: The device in use.

    Examples:
        >>> encoder = Encoder(path, SPECS["base-224"])           # doctest: +SKIP
        >>> encoder.classify(image, ["a forklift", "a person"])  # doctest: +SKIP
    """

    def __init__(self, weights: str | Path, spec: Spec, device: str = "cpu") -> None:
        self.spec = spec
        self.device = device
        self._weights = Path(weights)
        self._tokenizer: Tokenizer | None = None
        self._vision = None
        self._text = None
        self._scoring: tuple[torch.Tensor, torch.Tensor] | None = None
        self._locks = {part: Lock() for part in ("tokenizer", "vision", "text", "scoring")}

    # --- the towers ---

    def _once(self, part: str, build):
        """Return ``self._<part>``, building it the first time and only once.

        Four parts load lazily and each needs the same double check -- test, take the part's lock,
        test again -- so the discipline is written here rather than four times. The second test is
        what makes it correct: two threads can both pass the first one, and only the second may
        build.
        """
        attribute = f"_{part}"
        if getattr(self, attribute) is None:
            with self._locks[part]:
                if getattr(self, attribute) is None:
                    setattr(self, attribute, build())
        return getattr(self, attribute)

    @property
    def tokenizer(self) -> Tokenizer:
        """The tokenizer, built on first use.

        Lazy for the same reason the towers are, and worth about 870 ms of construction: building
        the merge tables means decompressing a 4 MB asset and constructing dictionaries over
        580,604 rules and 256,000 pieces. An ingest job that only encodes images never needs one,
        and should not wait for it before its first image.
        """
        return self._once("tokenizer", Tokenizer)

    def vision(self):
        """The image tower, built on first use."""
        return self._once(
            "vision", lambda: load_vision_tower(self.spec, self._weights, self.device))

    def text(self):
        """The text tower, built on first use."""
        return self._once(
            "text", lambda: load_text_tower(self.spec, self._weights, self.device))

    def scoring(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The learned temperature and bias, read on first use.

        Two scalars rather than a tower, but read the same way and for the same reason: an encode
        that never scores anything should not touch the file for them.
        """
        return self._once("scoring", lambda: load_scoring(self._weights))

    @property
    def loaded(self) -> tuple[str, ...]:
        """Which towers are resident. Empty until something is encoded."""
        return tuple(
            name for name, tower in (("vision", self._vision), ("text", self._text))
            if tower is not None
        )

    # --- encoding ---

    def encode_image(self, images: np.ndarray | Sequence[np.ndarray]) -> torch.Tensor:
        """Encode one image or many.

        Args:
            images: An ``HxWx3`` RGB ``uint8`` array, or a sequence of them.

        Returns:
            ``(N, projection)`` float32, L2-normalised. Always two-dimensional, including for a
            single image -- a shape that depends on whether the caller passed a list is the same
            trap as a response shape that depends on a query parameter.
        """
        batch = [images] if isinstance(images, np.ndarray) else list(images)
        if not batch:
            raise ValueError("give at least one image to encode")

        pixels = torch.stack([preprocess(image, self.spec.resolution) for image in batch])
        with torch.inference_mode():
            return normalise(self.vision()(pixels.to(self.device))).float().cpu()

    def encode_text(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Encode one phrase or many.

        Args:
            texts: A phrase, or a sequence of them. Each must fit SigLIP 2's 64-token context.

        Returns:
            ``(N, projection)`` float32, L2-normalised.

        Raises:
            ValueError: If nothing is given, if a phrase is blank, or if one does not fit.
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            return normalise(self.text()(tokens)).float().cpu()

    # --- the answer ---

    def classify(self, image: np.ndarray, prompts: Sequence[str]) -> torch.Tensor:
        """Score *image* against each phrase in *prompts*.

        Both towers run, so this is the one call that needs the whole model.

        Returns:
            ``(len(prompts),)`` probabilities in ``(0, 1)``, in the order the prompts were given.

            These are what upstream's own examples print. SigLIP scores each pair on its own --
            ``sigmoid(cos * exp(logit_scale) + logit_bias)`` -- so adding a phrase does not move
            any other phrase's number and the set does not sum to one. The learned bias is what
            makes the scale absolute: it sits near -17 on ``base-224``, which is why an unrelated
            phrase lands at zero rather than at some floor you would have to calibrate.

            They are still not calibrated class probabilities. Nothing in the training made classes
            compete, so this is how well *this* phrase matches *this* image and not P(class|image).
        """
        prompts = list(prompts)
        if not prompts:
            raise ValueError("give at least one phrase to score against")

        scale, bias = self.scoring()
        image_vector = self.encode_image(image)
        text_vectors = self.encode_text(prompts)
        # Upstream's order: text against image, scaled, biased, then transposed. Computing
        # ``image @ text.T`` directly is the same arithmetic and not guaranteed the same floats.
        logits_per_text = (text_vectors @ image_vector.T) * scale.exp() + bias
        return torch.sigmoid(logits_per_text.T)[0]
