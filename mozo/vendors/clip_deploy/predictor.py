# SPDX-License-Identifier: Apache-2.0
"""The seam: images or phrases in, vectors out — and the two compared when you want an answer.

**Each tower is built on first use and never before.** An ingest job that only ever calls
:meth:`Encoder.encode_image` never allocates the text tower; a query service that only calls
:meth:`Encoder.encode_text` holds 63.4M parameters instead of 151.3M. That is the deployment CLIP
is usually put to -- one expensive pass over a corpus, then a small always-on service answering
queries against a vector database -- and it falls out of building lazily rather than out of any
configuration.

Construction is locked per encoder. :class:`~mozo.manager.ModelManager` holds its own lock across
building the *adapter* and nowhere else, deliberately, and ``mozo.server`` runs handlers in a
threadpool -- so two concurrent image encodes on one cached encoder would otherwise both find the
tower missing and both build it.

**Vectors leave L2-normalised**, so a dot product between any two of them is a cosine similarity
and there is no convention for a caller to get wrong.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Sequence

import numpy as np
import torch

from .checkpoint import load_text_tower, load_vision_tower
from .config import Spec
from .image import preprocess
from .network import normalise
from .text.tokenizer import Tokenizer

__all__ = ["Encoder"]


class Encoder:
    """One loaded CLIP variant, with each tower built only if it is asked for.

    Args:
        weights: Path to a published checkpoint.
        spec: Which variant it is.
        device: Where to run.

    Attributes:
        spec: The variant's geometry.
        device: The device in use.

    Examples:
        >>> encoder = Encoder(path, SPECS["base"])              # doctest: +SKIP
        >>> encoder.encode_text(["a forklift", "a person"])     # doctest: +SKIP
        >>> encoder.classify(image, ["a forklift", "a person"]) # doctest: +SKIP
    """

    def __init__(self, weights: str | Path, spec: Spec, device: str = "cpu") -> None:
        self.spec = spec
        self.device = device
        self._weights = Path(weights)
        self.tokenizer = Tokenizer()
        self._vision = None
        self._text = None
        self._lock = Lock()

    # --- the towers ---

    def vision(self):
        """The image tower, built on first use."""
        if self._vision is None:
            with self._lock:
                if self._vision is None:
                    self._vision = load_vision_tower(self.spec, self._weights, self.device)
        return self._vision

    def text(self):
        """The text tower, built on first use."""
        if self._text is None:
            with self._lock:
                if self._text is None:
                    self._text = load_text_tower(self.spec, self._weights, self.device)
        return self._text

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
            ``(N, embed_dim)`` float32, L2-normalised. Always two-dimensional, including for a
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
            texts: A phrase, or a sequence of them. Each must fit CLIP's 77-token context.

        Returns:
            ``(N, embed_dim)`` float32, L2-normalised.

        Raises:
            ValueError: If nothing is given, if a phrase is blank, or if one does not fit.
        """
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            raise ValueError("give at least one phrase to encode")
        if any(not phrase.strip() for phrase in batch):
            raise ValueError("every phrase must say something; one of them is blank")

        tokens = self.tokenizer(batch).to(self.device)
        with torch.inference_mode():
            return normalise(self.text()(tokens)).float().cpu()

    # --- the answer ---

    def classify(
        self, image: np.ndarray, prompts: Sequence[str]
    ) -> torch.Tensor:
        """Score *image* against each phrase in *prompts*.

        Both towers run, so this is the one call that needs the whole model.

        Returns:
            ``(len(prompts),)`` cosine similarities, in the order the prompts were given. Not
            probabilities: they are not softmaxed, they do not sum to one, and they may be
            negative. See the adapter's docstring for why.
        """
        prompts = list(prompts)
        if not prompts:
            raise ValueError("give at least one phrase to score against")

        image_vector = self.encode_image(image)
        text_vectors = self.encode_text(prompts)
        return (image_vector @ text_vectors.T)[0]
