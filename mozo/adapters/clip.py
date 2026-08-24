# SPDX-License-Identifier: Apache-2.0
"""mozo's CLIP adapter: describe things in words, or get the vectors and compare them yourself.

The first family here that answers with neither a box nor a map. CLIP embeds an image and a phrase
into one shared space, and everything it is used for follows from comparing two of those vectors.

**Two things come out of it, and they are different products.**

``predict`` is a complete answer: hand it an image and some phrases, get each phrase scored. Zero
classes trained, no labelled data, no fine-tuning. That is what most people want CLIP for.

``encode_image`` and ``encode_text`` hand back the vectors themselves. Those are a handoff, not an
answer -- useful only next to other vectors, which means a store mozo does not provide. The usual
shape is: embed a corpus once with ``encode_image``, keep the vectors in a vector database, then
encode a query phrase and let the database find the nearest. mozo is the model in that pipeline
and nothing else.

**The towers load separately.** An ingest job that only encodes images never allocates the text
tower, and a query service that only encodes phrases holds 63.4M parameters rather than 151.3M.
Nothing to configure -- ask for what you need and the rest is never built.

**Both towers come from one checkpoint and cannot be mixed.** They were trained together until
their outputs landed in the same space; a vector from one variant means nothing against a vector
from another. So an index you have built is tied to the variant and revision that built it, and
changing either means re-embedding the corpus. That is the operational cost of an embedding
pipeline and it surprises people, so it is written here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np

from ..image import load_image
from ..runtimes import get_default_device, select_runtime
from ..vendors.clip_deploy import SPECS, VARIANTS, Encoder
from ..weights import artifacts, resolve, revision_of

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["ClipPredictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "clip"


class ClipPredictor:
    """Zero-shot classification, and the embeddings behind it.

    Args:
        variant: Which published model. ``base`` is ViT-B/32 and by far the most used.
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one.
        checkpoint_path: Your own checkpoint, instead of the published weights.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.
        revision: The published revision in use, or ``None`` for your own checkpoint.

    Examples:
        >>> model = ClipPredictor()                                       # doctest: +SKIP
        >>> found = model.predict("aisle.jpg", ["a forklift", "a person"])  # doctest: +SKIP
        >>> found[0].class_name                                            # doctest: +SKIP
        'a forklift'
        >>> model.encode_text(["a forklift"]).shape                        # doctest: +SKIP
        torch.Size([1, 512])
    """

    #: The four Vision Transformer variants. OpenAI also publishes five ResNet ones, which use a
    #: different image tower and are not carried yet. See ``PROVENANCE.md``.
    VARIANTS = list(VARIANTS)

    #: Which runtimes this adapter can execute. Only torch: the vendor builds two torch towers
    #: from a checkpoint and has no graph path. Declared rather than checked afterwards so ``auto``
    #: cannot pick an artifact this would then have to refuse.
    EXECUTES = ("torch",)

    def __init__(
        self,
        variant: str = "base",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        # Checked even with your own checkpoint: the variant selects the geometry here, so a wrong
        # one is not a mislabelling but a strict load that cannot succeed.
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
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
        #: Which published revision produced the vectors, or ``None`` for your own checkpoint. An
        #: embedding is only comparable against others from the same weights, so an index built
        #: from these has to record it -- which means the model has to be able to say.
        self.revision = None if checkpoint_path else revision_of(FAMILY, variant, revision=revision)
        # Neither tower is built here. The first encode decides which, and a caller that only ever
        # encodes phrases never pays for the image tower at all.
        self._encoder = Encoder(weights, SPECS[variant], device=self.device)
        print(f"CLIP {variant} ready on {self.device} via {self.runtime}.")

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        text: Union[str, Sequence[str]],
        threshold: float | None = None,
    ) -> "pf.Classifications":
        """Score *image* against each phrase in *text*.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            text: One phrase, or several. Each is scored independently against the image, so
                adding a phrase does not change the others' scores.
            threshold: Minimum similarity to keep. Omitted, every phrase comes back scored --
                which is what classification usually wants, unlike detection where a floor is the
                point.

        Returns:
            A PixelFlow ``Classifications``, best first, with ``class_name`` the phrase and
            ``class_id`` its position in *text*.

            **The score is a cosine similarity, not a probability.** It is not softmaxed, so it
            does not sum to one across the phrases and it may be negative. It is also compressed:
            CLIP's similarities for a good match sit far below 1.0, so 0.31 does not mean "31%
            sure" — compare scores against each other, or against a threshold you calibrate, not
            against an intuition about percentages.

            A softmax would be worse here, not better: it is relative to whatever phrases you
            happened to pass, so one phrase always scores 1.00 and adding a phrase moves every
            other number. Softmax it yourself if you want CLIP's published closed-set behaviour.

        Raises:
            ValueError: If no phrase is given, if one is blank, or if one exceeds CLIP's 77-token
                context.
        """
        prompts = [text] if isinstance(text, str) else list(text)
        if not prompts:
            raise ValueError("CLIP needs a phrase to score against; no text was given.")
        if any(not phrase.strip() for phrase in prompts):
            raise ValueError("CLIP needs a phrase to score against; text was empty.")

        scores = self._encoder.classify(load_image(image), prompts)
        found = pf.from_scores(scores.numpy(), labels=prompts).top_k(len(prompts))
        return found if threshold is None else found.filter_by_confidence(threshold)

    def encode_image(
        self, image: Union[str, Path, bytes, np.ndarray, Sequence]
    ) -> "np.ndarray":
        """Return ``(N, embed_dim)`` L2-normalised vectors for one image or many.

        Always two-dimensional, including for a single image. Batch where you can: the image tower
        is the expensive half, and a corpus is what it is for.
        """
        batch = image if isinstance(image, (list, tuple)) else [image]
        return self._encoder.encode_image([load_image(item) for item in batch]).numpy()

    def encode_text(self, text: Union[str, Sequence[str]]) -> "np.ndarray":
        """Return ``(N, embed_dim)`` L2-normalised vectors for one phrase or many.

        The cheap half. A phrase encoded once stays valid against every image vector ever stored
        with the same variant and revision, which is what makes searching a corpus by words
        affordable.
        """
        return self._encoder.encode_text(text).numpy()
