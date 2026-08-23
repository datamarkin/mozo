# SPDX-License-Identifier: Apache-2.0
"""mozo's SigLIP 2 adapter: describe things in words, or get the vectors and compare them yourself.

The same two products CLIP offers, with one difference that changes how the first is used.

``predict`` is a complete answer: hand it an image and some phrases, get each phrase scored. Zero
classes trained, no labelled data, no fine-tuning.

``encode_image`` and ``encode_text`` hand back the vectors themselves. Those are a handoff, not an
answer -- useful only next to other vectors, which means a store mozo does not provide. The usual
shape is: embed a corpus once with ``encode_image``, keep the vectors in a vector database, then
encode a query phrase and let the database find the nearest. mozo is the model in that pipeline and
nothing else.

**The score is a probability for that one pair.** SigLIP was trained with a sigmoid loss over
individual image-text pairs rather than a softmax over a batch, and the checkpoint carries a learned
bias alongside the learned temperature. So a phrase is scored against an image on its own: adding a
phrase does not move any other phrase's number, the set does not sum to one, and asking about a
single phrase is a well-posed question. That is not true of a cosine similarity, and it is the
practical reason to reach for this family.

It is still not a calibrated class probability. Nothing in the training made classes compete, so the
number says how well this phrase matches this image -- not P(class | image). Read it as a score with
a meaningful zero, and calibrate a threshold on your own data before deciding anything with it.

**The towers load separately.** An ingest job that only encodes images never allocates the text
tower, which here is most of the checkpoint: Gemma's 256,000-piece vocabulary is 786 MB of a
``base`` variant. Nothing to configure -- ask for what you need and the rest is never built.

**Both towers come from one checkpoint and cannot be mixed.** They were trained together until their
outputs landed in the same space; a vector from one variant means nothing against a vector from
another. So an index you have built is tied to the variant and revision that built it, and changing
either means re-embedding the corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np

from ..image import load_image
from ..runtimes import get_default_device, select_runtime
from ..vendors.siglip2_deploy import SPECS, VARIANTS, Encoder
from ..weights import artifacts, resolve, revision_of

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["Siglip2Predictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "siglip2"


class Siglip2Predictor:
    """Zero-shot classification, and the embeddings behind it.

    Args:
        variant: Which published model. ``base-224`` is the smallest and most used.
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
        >>> model = Siglip2Predictor()                                      # doctest: +SKIP
        >>> found = model.predict("aisle.jpg", ["a forklift", "a person"])  # doctest: +SKIP
        >>> found[0].class_name                                             # doctest: +SKIP
        'a forklift'
        >>> model.encode_text(["a forklift"]).shape                         # doctest: +SKIP
        (1, 768)
    """

    #: The five carried here, out of the fifteen fixed-resolution variants Google publishes. The
    #: vendor's ``config.py`` says which five and why; ``PROVENANCE.md`` covers the two ``-naflex``
    #: ones, which would need a different image tower.
    VARIANTS = list(VARIANTS)

    #: Which runtimes this adapter can execute. Only torch: the vendor builds two torch towers from
    #: a checkpoint and has no graph path. Declared rather than checked afterwards so ``auto``
    #: cannot pick an artifact this would then have to refuse.
    EXECUTES = ("torch",)

    def __init__(
        self,
        variant: str = "base-224",
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
        print(f"SigLIP 2 {variant} ready on {self.device} via {self.runtime}.")

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        text: Union[str, Sequence[str]],
        threshold: float | None = None,
    ) -> "pf.detections.Classifications":
        """Score *image* against each phrase in *text*.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.
            text: One phrase, or several. Each is scored independently against the image, so
                adding a phrase does not change the others' scores.
            threshold: Minimum score to keep. Omitted, every phrase comes back scored -- which is
                what classification usually wants, unlike detection where a floor is the point.

        Returns:
            A PixelFlow ``Classifications``, best first, with ``class_name`` the phrase and
            ``class_id`` its position in *text*.

            **The score is a per-pair probability in (0, 1), not a share of one.** It will not sum
            to one across the phrases, and every phrase can be near zero if none of them describes
            the image -- which is the useful case a softmax cannot express. See the class docstring
            for what it does and does not mean.

        Raises:
            ValueError: If no phrase is given, if one is blank, or if one exceeds SigLIP 2's
                64-token context.
        """
        prompts = [text] if isinstance(text, str) else list(text)
        if not prompts:
            raise ValueError("SigLIP 2 needs a phrase to score against; no text was given.")
        if any(not phrase.strip() for phrase in prompts):
            raise ValueError("SigLIP 2 needs a phrase to score against; text was empty.")

        scores = self._encoder.classify(load_image(image), prompts)
        found = pf.from_scores(scores.numpy(), labels=prompts).top_k(len(prompts))
        return found if threshold is None else found.filter_by_confidence(threshold)

    def encode_image(
        self, image: Union[str, Path, bytes, np.ndarray, Sequence]
    ) -> "np.ndarray":
        """Return ``(N, projection)`` L2-normalised vectors for one image or many.

        Always two-dimensional, including for a single image. Batch where you can: the image tower
        is the expensive half, and a corpus is what it is for.
        """
        batch = image if isinstance(image, (list, tuple)) else [image]
        return self._encoder.encode_image([load_image(item) for item in batch]).numpy()

    def encode_text(self, text: Union[str, Sequence[str]]) -> "np.ndarray":
        """Return ``(N, projection)`` L2-normalised vectors for one phrase or many.

        The cheap half to run, and the expensive half to hold: this is the tower carrying Gemma's
        vocabulary. A phrase encoded once stays valid against every image vector ever stored with
        the same variant and revision.
        """
        return self._encoder.encode_text(text).numpy()
