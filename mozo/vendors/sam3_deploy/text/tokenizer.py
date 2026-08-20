# SPDX-License-Identifier: MIT
"""The CLIP byte-pair tokenizer SAM 3 prompts go through.

This is OpenAI's tokenizer, not Meta's. ``assets/bpe_simple_vocab_16e6.txt.gz`` is byte-identical
to the file published with ``openai/CLIP`` (sha256 ``924691ac...48d6804a``), and the algorithm
below is the one that file was built for -- so both are MIT, and neither is SAM Materials.

Two details decide whether a prompt tokenizes the way the weights expect:

**Prompts are lowercased.** SAM 3 builds its tokenizer with ``clean="lower"``, so ``"A Red Hat"``
and ``"a red hat"`` are the same prompt. Skipping that is not a nicety -- uppercase words BPE into
different subwords and segment differently.

**The context length is 32, not 77.** Every other CLIP model uses 77; SAM 3's position embedding
is ``(32, 1024)``. Longer prompts are truncated with the end-of-text id forced back into the last
slot, so a truncated prompt is still terminated.
"""

from __future__ import annotations

import gzip
import html
from functools import lru_cache
from pathlib import Path

import ftfy
import regex
import torch

from ..config import TEXT

__all__ = ["Tokenizer", "VOCAB_PATH"]

#: The vocabulary shipped with this package.
VOCAB_PATH = Path(__file__).resolve().parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

#: Where the merge list stops. CLIP's file carries a header line and trailing entries that are
#: not merges; upstream slices exactly this range and the vocabulary ids depend on it.
MERGE_LIMIT = 49152 - 256 - 2 + 1

SPECIAL_TOKENS = ("<start_of_text>", "<end_of_text>")


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Map every byte to a printable unicode character.

    BPE runs on text, but prompts are bytes. Mapping the unprintable bytes up into unused
    codepoints keeps the merge table reversible without the algorithm ever meeting a control
    character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapped = printable[:]
    spare = 0
    for byte in range(2**8):
        if byte not in printable:
            printable.append(byte)
            mapped.append(2**8 + spare)
            spare += 1
    return dict(zip(printable, (chr(code) for code in mapped)))


def pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Every adjacent symbol pair in ``word``."""
    return {(first, second) for first, second in zip(word, word[1:])}


class Tokenizer:
    """CLIP byte-pair encoding, as SAM 3 configures it.

    Args:
        vocab_path: The gzipped merge list. Defaults to the vendored copy.
        context_length: Fixed width of the returned id matrix.
    """

    def __init__(self, vocab_path: Path = VOCAB_PATH, context_length: int = TEXT.context_length):
        self.context_length = context_length

        self.byte_encoder = bytes_to_unicode()
        with gzip.open(vocab_path) as handle:
            merges = handle.read().decode("utf-8").split("\n")
        merges = [tuple(merge.split()) for merge in merges[1:MERGE_LIMIT]]

        vocabulary = list(self.byte_encoder.values())
        vocabulary += [token + "</w>" for token in vocabulary]
        vocabulary += ["".join(merge) for merge in merges]
        vocabulary += list(SPECIAL_TOKENS)

        self.encoder = {token: index for index, token in enumerate(vocabulary)}
        self.ranks = {merge: rank for rank, merge in enumerate(merges)}
        self._cache: dict[str, str] = {token: token for token in SPECIAL_TOKENS}

        self.start_id = self.encoder[SPECIAL_TOKENS[0]]
        self.end_id = self.encoder[SPECIAL_TOKENS[1]]

        # ``\p{L}`` and ``\p{N}`` are Unicode categories, which the standard library's ``re``
        # cannot express -- hence ``regex``. Substituting ``\w`` here silently changes how
        # accented and non-Latin prompts split.
        self.pattern = regex.compile(
            "|".join(SPECIAL_TOKENS)
            + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )

    def clean(self, text: str) -> str:
        """Repair encoding damage, collapse whitespace, lowercase."""
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text)).strip()
        return regex.sub(r"\s+", " ", text).strip().lower()

    def merge(self, token: str) -> str:
        """Apply the merge table to one whitespace-free token, returning space-joined symbols."""
        if token in self._cache:
            return self._cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        current = pairs(word)
        while current:
            # Always merge the highest-ranked (earliest-learned) pair available. That ordering is
            # the encoding -- taking any other pair first produces different, valid-looking ids.
            best = min(current, key=lambda pair: self.ranks.get(pair, float("inf")))
            if best not in self.ranks:
                break
            first, second = best
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    found = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:found])
                index = found
                if word[index] == first and index < len(word) - 1 and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            current = pairs(word) if len(word) > 1 else set()

        joined = " ".join(word)
        self._cache[token] = joined
        return joined

    def encode(self, text: str) -> list[int]:
        """Turn one prompt into token ids, without the start and end markers."""
        ids: list[int] = []
        for token in self.pattern.findall(self.clean(text)):
            token = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
            ids.extend(self.encoder[symbol] for symbol in self.merge(token).split(" "))
        return ids

    def __call__(self, texts: str | list[str]) -> torch.Tensor:
        """Tokenize one prompt or a batch into a fixed-width id matrix.

        Args:
            texts: A prompt, or a list of them.

        Returns:
            ``(len(texts), context_length)`` int64. Rows are zero-padded on the right, which is
            what makes ``ids != 0`` the attention mask.
        """
        if isinstance(texts, str):
            texts = [texts]
        result = torch.zeros(len(texts), self.context_length, dtype=torch.long)
        for row, text in enumerate(texts):
            ids = [self.start_id, *self.encode(text), self.end_id]
            if len(ids) > self.context_length:
                ids = ids[: self.context_length]
                # Keep the sequence terminated even when the prompt did not fit.
                ids[-1] = self.end_id
            result[row, : len(ids)] = torch.tensor(ids)
        return result
