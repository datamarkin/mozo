# SPDX-License-Identifier: Apache-2.0
"""The seam: an image and some phrases in, located boxes out.

Everything a caller of this package needs is here. It builds the caption upstream expects from
the list of prompts mozo's endpoint takes, runs the model, and hands back detections that carry
the index of the prompt each one matched.

**mozo owns the caption.** Upstream takes one string and leaves the joining to the caller, which
means the separator convention -- lowercase, phrases joined by ``" . "``, a trailing ``"."`` --
is part of the contract and easy to get subtly wrong. Taking a list instead makes that this
package's business, and it is what lets a detection be traced back to a prompt rather than to a
decoded span.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .boxes import Detection, decode
from .checkpoint import build
from .config import Spec
from .image import preprocess
from .network import phrase_masks
from .text.tokenizer import Tokenizer

__all__ = ["SEPARATORS", "Predictor", "caption_for"]

#: Tokens that end a phrase. ``"?"`` is one of them, so a prompt containing a question mark
#: splits -- upstream's rule, and the reason a question is not a good prompt.
#:
#: Public, and exported from the package, because the gate and the tests need the same set. A
#: private copy retyped in each of them is the one failure a bit-exactness gate may not have:
#: change the set here and the gate keeps passing while feeding ``phrase_masks`` the old one.
SEPARATORS = ("[CLS]", "[SEP]", ".", "?")


def caption_for(prompts: Sequence[str]) -> str:
    """Join prompts into the single caption the model was trained on.

    Lowercased and separated by ``" . "`` with a trailing ``"."``, which is
    ``preprocess_caption`` composed with the joining upstream leaves to its caller. Case is
    destroyed here because upstream destroys it; a proper noun is not preserved.
    """
    return " . ".join(prompt.strip().lower() for prompt in prompts) + " ."


class Predictor:
    """One loaded Grounding DINO variant.

    Args:
        weights: Path to a published checkpoint.
        spec: Which variant it is.
        device: Where to run.

    Attributes:
        spec: The variant's geometry.
        device: The device in use.

    Examples:
        >>> model = Predictor(path, SPECS["tiny"])          # doctest: +SKIP
        >>> model(image, ["a cat", "a laptop"])             # doctest: +SKIP
    """

    def __init__(self, weights: str | Path, spec: Spec, device: str = "cpu") -> None:
        self.spec = spec
        self.device = device
        self.tokenizer = Tokenizer()
        # Built once, and on the CPU: the text setup runs there. See __call__.
        self._separators = torch.tensor(
            self.tokenizer.convert_tokens_to_ids(list(SEPARATORS)), dtype=torch.long
        )
        self.model = build(spec, weights, device=device)

    def __call__(
        self,
        image: np.ndarray,
        prompts: Sequence[str],
        box_threshold: float | None = None,
    ) -> list[Detection]:
        """Find every instance of each prompt in *image*.

        Args:
            image: ``HxWx3`` RGB ``uint8``.
            prompts: What to look for. Each is free text and may contain spaces.
            box_threshold: Confidence floor. ``None`` takes the variant's published default.

        Returns:
            Detections carrying a box in source pixels, a score, and the index of the prompt
            that matched.

        Raises:
            ValueError: If no prompt is given, if one is blank, or if the caption exceeds the
                model's 256-token budget. Upstream truncates silently there; a prompt that was
                dropped without being mentioned is a wrong answer that never raises.
        """
        if not prompts:
            raise ValueError("give at least one prompt naming what to look for")
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("every prompt must name a concept; one of them is blank")

        caption = caption_for(prompts)
        ids, types, mask = self.tokenizer.encode(caption)
        if len(ids) > self.spec.max_text_len:
            raise ValueError(
                f"{len(prompts)} prompts tokenize to {len(ids)} tokens, over this model's "
                f"{self.spec.max_text_len}-token budget. Ask for fewer, or shorter ones."
            )

        source_height, source_width = image.shape[:2]
        batch = preprocess(image, self.spec.short_side, self.spec.max_side)[None].to(self.device)

        # The text setup is built on the CPU and moved once, rather than built where the model
        # lives. ``phrase_masks`` is twenty tiny integer kernels and a ``nonzero`` -- which on an
        # accelerator is a synchronisation -- for a caption of a dozen tokens: 2.10 ms on MPS
        # against 0.067 ms here. The phrase map is wanted back on the CPU anyway, for decoding.
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention, position_ids, phrase_map = phrase_masks(input_ids, self._separators)

        if phrase_map[0].shape[0] != len(prompts):
            # The caption split into a different number of phrases than prompts were given --
            # a prompt containing its own '.' or '?' is the way that happens. Refused rather
            # than reported against the wrong prompt.
            raise ValueError(
                f"{len(prompts)} prompt(s) became {phrase_map[0].shape[0]} phrase(s); a prompt "
                "may not contain '.' or '?', which separate concepts."
            )

        with torch.inference_mode():
            logits, boxes = self.model(
                batch,
                input_ids.to(self.device),
                torch.tensor([types], dtype=torch.long, device=self.device),
                torch.tensor([mask], dtype=torch.bool, device=self.device),
                attention.to(self.device),
                position_ids.to(self.device),
            )

        return decode(
            logits[0].float().cpu(),
            boxes[0].float().cpu(),
            phrase_map[0],
            source_height,
            source_width,
            self.spec.box_threshold if box_threshold is None else box_threshold,
        )
