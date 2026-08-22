# SPDX-License-Identifier: Apache-2.0
"""Turn 900 queries into detections: threshold, name, and put the boxes back in source pixels.

Two decisions live here, and the second is where this package deliberately differs from upstream.

**Which queries survive.** Each query carries a similarity against every text token. Its score is
the largest of those, and it is kept when that clears the box threshold. There is no class
softmax and no suppression -- like OWLv2, Grounding DINO is trained to answer per query and
overlapping answers are not deduplicated for you.

**What a detection is called.** Upstream decodes the tokens above ``text_threshold`` back into a
string, which can return a fragment of a phrase (``"yellow school"`` for ``"yellow school bus"``)
or a span running across two of them. mozo instead reports *which prompt* the query matched, by
the phrase map the tokenizer already produced -- so the name a caller gets back is the string the
caller passed in. See ``PROVENANCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["Detection", "decode"]


@dataclass(frozen=True, slots=True)
class Detection:
    """One found thing.

    Attributes:
        box: ``(x1, y1, x2, y2)`` in the source image's own pixels.
        score: The query's largest similarity against the prompt, after sigmoid.
        prompt_index: Which prompt matched, indexing the list the caller passed.
    """

    box: tuple[float, float, float, float]
    score: float
    prompt_index: int


def decode(
    logits: Tensor,
    boxes: Tensor,
    phrase_map: Tensor,
    source_height: int,
    source_width: int,
    box_threshold: float,
) -> list[Detection]:
    """Turn one image's raw outputs into detections.

    Args:
        logits: ``(queries, max_text_len)`` raw similarities for one image.
        boxes: ``(queries, 4)`` as ``cxcywh``, normalised to the *resized* image.
        phrase_map: ``(phrases, tokens)`` boolean -- which tokens belong to which prompt.
        source_height: Height of the original image, in its own pixels.
        source_width: Width of the original image.
        box_threshold: Confidence floor on the per-query maximum.

    Returns:
        Detections in the model's own order, which is neither sorted nor deduplicated.
    """
    scored = logits.sigmoid()
    best, _ = scored.max(dim=1)
    keep = best > box_threshold
    if not bool(keep.any()):
        return []

    kept_scores = scored[keep]
    kept_boxes = boxes[keep]
    kept_best = best[keep]

    # Which prompt a query matched: the token its similarity peaks at, looked up in the phrase
    # map. The map is padded out to max_text_len the same way the logits are, so the argmax and
    # the lookup index the same axis.
    tokens = phrase_map.shape[1]
    padded = torch.zeros(
        (phrase_map.shape[0], scored.shape[1]), dtype=torch.bool, device=phrase_map.device
    )
    padded[:, :tokens] = phrase_map
    peak = kept_scores.argmax(dim=1)
    # (kept, phrases) -- True where that query's peak token belongs to that prompt.
    belongs = padded[:, peak].t()

    # Normalised cxcywh -> absolute xyxy. The boxes are relative to the resized image, but the
    # resize preserved the aspect ratio and padded nothing, so the same normalised numbers scale
    # straight onto the source. Multiplying by the *resized* size here is the classic error and
    # it is off by exactly the resize factor.
    centre_x = kept_boxes[:, 0] * source_width
    centre_y = kept_boxes[:, 1] * source_height
    half_w = kept_boxes[:, 2] * source_width / 2
    half_h = kept_boxes[:, 3] * source_height / 2

    # Every column below is computed for all survivors at once and read out once. Done per
    # detection instead -- which is the shape this started in -- each row costs a boolean gather
    # over all 900 queries and four 0-dim tensor reads: 4.03 ms against 0.75 ms at full keep.
    # The arithmetic is the same elementwise fp32 op on the same pairs, so the numbers are
    # identical; only the batching changes.
    named = belongs.any(dim=1)
    # The first prompt whose tokens contain the peak. ``argmax`` on a boolean row returns the
    # first True, which is what indexing ``nonzero(...)[0]`` did.
    prompts = belongs.int().argmax(dim=1).tolist()
    x1 = (centre_x - half_w).tolist()
    y1 = (centre_y - half_h).tolist()
    x2 = (centre_x + half_w).tolist()
    y2 = (centre_y + half_h).tolist()
    scores = kept_best.tolist()

    return [
        Detection(box=(x1[i], y1[i], x2[i], y2[i]), score=scores[i], prompt_index=prompts[i])
        for i in range(len(prompts))
        # A peak that landed on a separator or a special token names no prompt. Dropped rather
        # than guessed: this package will not invent a class name.
        if named[i]
    ]
