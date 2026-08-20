# SPDX-License-Identifier: Apache-2.0
"""The public segmenter: load a checkpoint, name a concept, get masks in the image's own pixels.

SAM 3's cost is lopsided in two directions at once, and both are worth caching:

- the image encoder is 4.8 s and depends only on the image
- the text tower is 88 ms and depends only on the phrase
- everything that joins them is 670 ms and depends on both

So an annotator trying five prompts on one photograph pays the 4.8 s once, and a pipeline running
one phrase over ten thousand photographs pays the 88 ms once. Those are different caches serving
different jobs, which is why there are two.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from threading import Lock

import numpy as np
import torch

from .checkpoint import concept_state_dict, load_state_dict, text_state_dict, vision_state_dict
from .grounding import ConceptHead
from .grounding.boxes import box_cxcywh_to_xyxy
from .image import preprocess, to_original
from .text import TextEncoder, Tokenizer
from .vision import VisionEncoder

__all__ = ["Segmenter", "instances"]

#: How many images' encoder outputs to keep. Each is 223 MB at fp32 -- the two pyramids, one for
#: each prompt modality -- so this is the one number in the package that has to be justified
#: rather than chosen. Two is enough for the job the image cache exists for, which is trying
#: several prompts on the picture in front of you. SAM 2 keeps five because each of its entries
#: is 17 MB; scaling that count here would be 1.1 GB and would rule out a Jetson.
IMAGE_CACHE = 2

#: How many encoded prompts to keep. Each is 33 KB, so this is 1 MB and the number is
#: uninteresting -- it exists only so a server fed unbounded distinct prompts does not grow
#: without limit.
PROMPT_CACHE = 32

#: Above this a mask logit is foreground.
MASK_THRESHOLD = 0.0


def instances(
    result: dict[str, torch.Tensor], shape: tuple[int, int], threshold: float = 0.5
) -> list[dict[str, torch.Tensor]]:
    """Reduce a forward pass to the instances that survive ``threshold``.

    Args:
        result: What :meth:`~.grounding.concept.ConceptHead.forward` returned.
        shape: The source image's ``(height, width)``.
        threshold: Minimum score.

    Returns:
        One entry per image in the batch, each with ``masks`` ``(N, height, width)`` bool,
        ``boxes`` ``(N, 4)`` in source pixels as xyxy, and ``scores`` ``(N,)``.
    """
    # A query's score is its own confidence gated by whether the concept is in the picture at
    # all. Without the presence term, "cow" on a picture of an office still returns the 200
    # queries' best guesses.
    scores = result["logits"].sigmoid() * result["presence"].sigmoid()
    height, width = shape
    scale = result["boxes"].new_tensor([width, height, width, height])

    found = []
    for image in range(scores.shape[0]):
        keep = scores[image] > threshold
        selected = result["masks"][image : image + 1, keep]
        # A prompt that finds nothing is a normal answer, not an error -- ask a picture of an
        # office for "cow" and every query should fall below the threshold. Resizing would raise
        # on an empty batch, so the empty case is built directly.
        masks = (
            to_original(selected, shape)[0] > MASK_THRESHOLD
            if selected.shape[1]
            else torch.zeros(0, height, width, dtype=torch.bool, device=selected.device)
        )
        found.append({
            "masks": masks,
            "boxes": box_cxcywh_to_xyxy(result["boxes"][image][keep]) * scale,
            "scores": scores[image][keep],
        })
    return found


class Segmenter:
    """Concept segmentation for one SAM 3 checkpoint.

    Args:
        checkpoint: Path to Meta's published ``sam3.pt``.
        device: Where to run. mozo decides this; the default is only for direct use.

    Attributes:
        image_size: Square side the encoder runs at.
    """

    def __init__(self, checkpoint: str | os.PathLike, device: str | torch.device = "cpu"):
        state = load_state_dict(os.fspath(checkpoint))

        self.vision = VisionEncoder()
        self.vision.load_state_dict(vision_state_dict(state), strict=True)
        self.text = TextEncoder()
        self.text.load_state_dict(text_state_dict(state), strict=True)
        self.concept = ConceptHead()
        self.concept.load_state_dict(concept_state_dict(state), strict=True)
        del state

        for module in (self.vision, self.text, self.concept):
            module.eval().to(device)

        self.tokenizer = Tokenizer()
        self.device = device
        self.image_size = self.vision.trunk.spec.image_size

        self._images: OrderedDict[bytes, dict] = OrderedDict()
        self._prompts: OrderedDict[str, dict] = OrderedDict()
        # One segmenter instance is shared across requests -- mozo.server runs handlers in a
        # threadpool -- and check-then-act on an OrderedDict is not safe across threads.
        self._lock = Lock()

    def _remember(self, cache: OrderedDict, key, build, limit: int):
        """Return ``cache[key]``, computing it outside the lock if it is not there yet."""
        with self._lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
        value = build()
        with self._lock:
            cache[key] = value
            while len(cache) > limit:
                cache.popitem(last=False)
        return value

    def encode_image(self, image: np.ndarray) -> dict:
        """Return the vision features for one image, computing them only if not already held.

        This is the expensive half and depends on nothing but the image, so an annotator trying
        several prompts on one photograph pays for it once. Keyed on pixel content rather than on
        a filename or an object identity, because the same image arriving twice over HTTP is two
        different arrays and should still be one encode.
        """
        # sha256 over the pixels, not a sample of them: a key that skipped content would collide
        # on two images differing only where it did not look, and hand back a confident mask of
        # the wrong object. Hashed straight from the array -- ``tobytes`` would copy every byte.
        key = hashlib.sha256(np.ascontiguousarray(image)).digest()
        return self._remember(
            self._images,
            key,
            lambda: self.vision(preprocess(image).to(self.device)),
            IMAGE_CACHE,
        )

    def encode_text(self, prompt: str) -> dict:
        """Return the encoded prompt, computing it only if not already held.

        A phrase encoded once is valid for every image afterwards, which is what makes running one
        prompt over a whole corpus cheap. Keyed on the raw string, before tokenization, so two
        spellings that tokenize alike are still two entries -- the tokenizer is cheap and the
        comparison is not worth being clever about.
        """
        return self._remember(
            self._prompts,
            prompt,
            lambda: self.text(self.tokenizer([prompt]).to(self.device)),
            PROMPT_CACHE,
        )

    def predict(
        self,
        image: np.ndarray,
        text: str,
        boxes: np.ndarray | None = None,
        box_labels: np.ndarray | None = None,
        threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Find every instance of ``text`` in ``image``.

        Args:
            image: ``HxWx3`` RGB ``uint8``, as :func:`mozo.image.load_image` returns.
            text: The concept to look for -- a noun phrase, up to 32 tokens.
            boxes: ``(N, 4)`` exemplar boxes as normalised cxcywh, for "find more like this".
            box_labels: ``(N,)`` 1 positive, 0 negative. Required with ``boxes``.
            threshold: Minimum score for an instance to be returned.

        Returns:
            ``masks`` ``(N, height, width)`` bool in the source image's pixels, ``boxes``
            ``(N, 4)`` xyxy in source pixels, and ``scores`` ``(N,)``.
        """
        features = self.encode_image(image)
        encoded = self.encode_text(text)

        exemplars = labels = None
        if boxes is not None:
            exemplars = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)[None]
            if box_labels is not None:
                labels = torch.as_tensor(box_labels, dtype=torch.long, device=self.device)[None]

        result = self.concept(
            features["concept"],
            features["positions"],
            encoded["features"],
            encoded["mask"],
            exemplars,
            labels,
        )
        return instances(result, image.shape[:2], threshold)[0]
