# SPDX-License-Identifier: Apache-2.0
"""The deployable surface: a photograph and some phrases in, boxes out.

Replaces ``Owlv2Processor`` and ``Owlv2ImageProcessor.post_process_object_detection``, and adds
the caching that makes the split in :mod:`~mozo.vendors.owlv2_deploy.network` worth having --
without it, running one vocabulary over a corpus re-encodes the vocabulary for every picture.

**A detection here is a patch.** OWLv2 has no proposals and no decoder queries: it scores every
patch of the image against every phrase and predicts one box per patch. So the model always
returns exactly ``patches^2`` candidates -- 3,600 for the base geometry -- and thresholding is the
whole of the selection. There is no non-maximum suppression, and that is not an omission: the
published postprocessing has none either, and adding one would be mozo inventing a policy the
model was not evaluated under.

**Each patch keeps only its best phrase.** The score is the largest logit across the vocabulary,
put through a sigmoid, and the label is which phrase that was. So asking for ``["cat", "dog"]``
cannot return the same box twice under both names -- which is upstream's behaviour, and worth
knowing before someone reads the count as "how many cats and dogs are there".

**The prompt cache is keyed on the whole vocabulary, not on each phrase.** That looks like a
missed opportunity and is not one: the text tower runs the phrases as a batch, and a batched
matmul does not produce bit-identical rows to the same matmul run one row at a time. Caching per
phrase and re-stacking would have made ``["cat"]`` followed by ``["cat", "dog"]`` return a
different embedding for ``"cat"`` than asking for both at once -- measured at 2.5e-07 on the
score, which is small, invisible, and enough to fail the parity gate. Whole-vocabulary keying
still buys what the cache exists for, which is one text forward for a corpus rather than one per
picture.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch

from .checkpoint import load_state_dict
from .config import SPECS
from .image import preprocess, to_original
from .network import OwlV2
from .text.tokenizer import Tokenizer

__all__ = ["Detection", "Detector", "IMAGE_CACHE", "PROMPT_CACHE", "THRESHOLD"]

#: How many images' patch features to keep. Each is 11 MB at fp32 for the base geometry and 21 MB
#: for the large one -- 3,600 or 5,184 patches of 768 or 1,024 floats. Five, which is what the
#: promptable families here keep at a comparable size, and enough for the job an image cache
#: exists for: trying several vocabularies on the picture in front of you.
IMAGE_CACHE = 5

#: How many encoded vocabularies to keep. Each is 2 KB per phrase, so this is well under a
#: megabyte and the number is uninteresting -- it exists only so a server fed unbounded distinct
#: vocabularies does not grow without limit.
PROMPT_CACHE = 64

#: Minimum score for a patch to be returned. Upstream's ``post_process_object_detection`` default.
THRESHOLD = 0.1


@dataclass(frozen=True)
class Detection:
    """What one image and one vocabulary produced.

    Attributes:
        boxes: ``(N, 4)`` xyxy in the source image's pixels. Not clipped to it -- a box may
            legitimately run off the edge, and clipping is a decision for whoever draws it.
        scores: ``(N,)`` in ``[0, 1]``: the best phrase's logit for that patch, through a sigmoid.
        labels: ``(N,)`` int64, indexing the phrases that were asked for.
        objectness: ``(N,)`` in ``[0, 1]``: how likely the patch holds an object *at all*,
            independent of what was asked. OWLv2's addition over OWL-ViT, and the number to sort
            by when the question is "what is in this picture" rather than "where is the cat".
    """

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    objectness: torch.Tensor

    def __len__(self) -> int:
        return int(self.scores.shape[0])


class Detector:
    """Open-vocabulary detection for one OWLv2 checkpoint.

    Args:
        checkpoint: Path to the published ``pytorch_model.bin``, which mozo republishes as
            ``torch-fp32.pth``.
        variant: Which published geometry. See :data:`~.config.SPECS`.
        device: Where to run. mozo decides this; the default is only for direct use.

    Attributes:
        image_size: Square side the trunk runs at.
        device: Where it runs.
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike,
        variant: str = "base-ensemble",
        device: str | torch.device = "cpu",
    ):
        # No variant check here: ``OwlV2`` raises the same message on the next line, before the
        # checkpoint is read.
        self.model = OwlV2(variant)
        self.model.load_state_dict(load_state_dict(os.fspath(checkpoint)), strict=True)
        self.model.eval().to(device)

        self.tokenizer = Tokenizer(context_length=SPECS[variant].text.context_length)
        self.device = device
        self.image_size = self.model.image_size

        self._images: OrderedDict[bytes, torch.Tensor] = OrderedDict()
        self._prompts: OrderedDict[
            tuple[str, ...], tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        # One detector instance is shared across requests -- mozo.server runs handlers in a
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

    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """Return the patch features for one image, computing them only if not already held.

        This is the expensive half and depends on nothing but the picture, so trying a second
        vocabulary on it is nearly free. Keyed on pixel content rather than on a filename or an
        object identity, because the same image arriving twice over HTTP is two different arrays
        and should still be one encode.
        """
        # sha256 over the pixels, not a sample of them: a key that skipped content would collide
        # on two images differing only where it did not look, and hand back confident boxes for
        # the wrong picture. Hashed straight from the array -- ``tobytes`` would copy every byte.
        key = hashlib.sha256(np.ascontiguousarray(image)).digest()
        return self._remember(
            self._images,
            key,
            lambda: self.model.encode_image(preprocess(image, self.image_size).to(self.device)),
            IMAGE_CACHE,
        )

    def encode_text(self, phrases: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one vocabulary encoded, computing it only if not already held.

        A vocabulary encoded once is valid for every image afterwards, which is what makes running
        it over a corpus cost one text forward rather than one per picture. Keyed on the raw
        strings, before tokenization, so two spellings that tokenize alike are still two entries
        -- the tokenizer is cheap and the comparison is not worth being clever about. Keyed on the
        whole tuple rather than phrase by phrase; see the module docstring for why that is not a
        missed saving.

        Returns:
            ``queries`` ``(Q, projection)`` and the ``(Q,)`` bool mask of which slots carry a
            prompt -- upstream's rule, which is that the first token id is non-zero.
        """
        return self._remember(
            self._prompts,
            tuple(phrases),
            lambda: self._encode_text(phrases),
            PROMPT_CACHE,
        )

    def _encode_text(self, phrases: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = self.tokenizer(list(phrases))
        queries = self.model.encode_text(ids.to(self.device), mask.to(self.device))
        return queries, (ids[:, 0] > 0).to(self.device)

    def predict(
        self,
        image: np.ndarray,
        text: Sequence[str],
        threshold: float = THRESHOLD,
    ) -> Detection:
        """Find every instance of each phrase in ``image``.

        Args:
            image: ``HxWx3`` RGB ``uint8``, as :func:`mozo.image.load_image` returns.
            text: The phrases to look for -- nouns or noun phrases, up to 16 tokens each. Passed
                to the model verbatim: OWLv2's own examples wrap them as ``"a photo of a cat"``,
                but that is the caller's wording to choose, not a template this applies.
            threshold: Minimum score for a patch to be returned.

        Returns:
            A :class:`Detection`, ordered by patch rather than by score. Ranking is the caller's.

        Raises:
            ValueError: If ``text`` is empty, or any phrase is blank. An empty vocabulary has no
                answer, and a blank phrase would be scored against as though it were a concept.
        """
        phrases = list(text)
        if not phrases or any(not phrase.strip() for phrase in phrases):
            raise ValueError("give at least one phrase to look for, and no blank ones")

        queries, mask = self.encode_text(phrases)
        patches = self.encode_image(image)
        logits, boxes, objectness = self.model.detect(patches, queries, mask)

        best = logits[0].max(dim=-1)
        scores = torch.sigmoid(best.values)
        keep = scores > threshold
        return Detection(
            boxes=to_original(boxes[0][keep], image.shape[:2]),
            scores=scores[keep],
            labels=best.indices[keep],
            objectness=torch.sigmoid(objectness[0][keep]),
        )
