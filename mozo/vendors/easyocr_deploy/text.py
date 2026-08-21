"""The alphabet, and turning a step sequence into a string.

A CTC model does not emit characters, it emits one distribution per horizontal step over an
alphabet plus a blank. Two steps in a row naming the same character are one character; a blank
between them means they are two. That collapse is :meth:`Alphabet.decode`, and it is the only
place in this package where a number becomes text.
"""

from __future__ import annotations

__all__ = ["BLANK", "Alphabet"]

import numpy as np
import torch
import torch.nn.functional as F

#: CTC's blank symbol takes index 0, so every real character sits one higher than its position
#: in the published charset. Off by one here shifts every string the model produces.
BLANK = 0


def _confidence(probabilities: np.ndarray) -> float:
    """Upstream's ``custom_mean``: the product raised to ``2 / sqrt(n)``.

    Not an average. It is a geometric mean stretched by the square root of the length, so a long
    confident line does not get pulled down as far as multiplying probabilities would take it.
    Worth keeping exactly, because it is also the number the low-confidence retry compares to
    decide which of two reads to keep.
    """
    return float(probabilities.prod() ** (2.0 / np.sqrt(len(probabilities))))


class Alphabet:
    """The charset a variant was trained on, plus the blank, plus how to read one back.

    The whole charset is decodable. Upstream additionally masks out every character outside the
    languages the caller asked for, which makes its output depend on a language list rather than
    on the weights; a mozo variant is a checkpoint, so there is nothing to mask. See
    PROVENANCE.md.
    """

    def __init__(self, characters: str) -> None:
        #: Index 0 is the blank; index i+1 is ``characters[i]``.
        self.characters = ["[blank]"] + list(characters)
        # Built once. Upstream rebuilds it inside every decode, which for the 6,719-symbol
        # Chinese alphabet is 0.54 ms a line -- 40% of that variant's decode.
        self._table = np.array(self.characters)

    def decode(self, logits: torch.Tensor) -> list[tuple[str, float]]:
        """``(B, T, num_class)`` logits to one ``(text, confidence)`` per row.

        The renormalisation looks redundant -- softmax already sums to one -- and it is where
        upstream applies its language mask, which mozo has none of. It is kept because a float32
        softmax sums to *almost* one, and dividing by that almost changes the low bits of every
        probability, which changes the confidence and can change an argmax.
        """
        probabilities = F.softmax(logits, dim=2).cpu().detach().numpy()
        probabilities = probabilities / np.expand_dims(probabilities.sum(axis=2), axis=-1)
        # A no-op for the float32 the recogniser emits, and upstream's own dtype dance; kept
        # because it is the only thing pinning this to float32 if a caller ever hands in wider.
        probabilities = torch.from_numpy(probabilities).float().numpy()

        best = probabilities.argmax(axis=2)
        scores = probabilities.max(axis=2)

        results = []
        for steps, step_scores in zip(best, scores):
            # A step counts only if it names something new: not a repeat of the step before it,
            # and not the blank.
            fresh = np.insert(steps[1:] != steps[:-1], 0, True)
            real = steps != BLANK
            text = "".join(self._table[steps[(fresh & real).nonzero()]])

            # Confidence is measured over every non-blank step, including the repeats the
            # collapse just dropped -- so it reflects how sure the model was across the whole
            # line, not only at the characters that survived.
            kept = step_scores[steps != BLANK]
            results.append((text, _confidence(kept) if len(kept) else 0.0))
        return results
