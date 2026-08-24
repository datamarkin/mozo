"""What a person typed into a text box, as a list.

One rule -- split on commas, strip, drop the empties -- and one home, because it is the contract for
every field where several things go in one box: the phrases a zero-shot classifier scores, the
concepts an open-vocabulary detector looks for, the class names a prediction is filtered to, the
keypoints a box is redrawn around. It was written three times before this existed, in two packages,
and widening it (a semicolon, a newline) would have meant finding all three.

Commas rather than Grounding DINO's upstream ``"person . car . dog"``: mozo's adapters refuse a
prompt containing ``.``, and one separator everywhere is one thing to learn instead of four.
"""

from __future__ import annotations

from typing import Optional


def comma_separated(text: Optional[str]) -> list:
    """Split *text* on commas. Unset or empty gives an empty list."""
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]
