# SPDX-License-Identifier: Apache-2.0
"""BERT's WordPiece tokenizer, over the vocabulary that ships beside this file.

Grounding DINO tokenizes prompts with `bert-base-uncased`. Upstream reaches that through
``transformers.AutoTokenizer.from_pretrained``, which downloads the vocabulary on first use;
this package carries the file instead, so a prompt never needs a network round trip to become
token ids.

Only the uncased configuration is implemented, because only it is published: lowercase, strip
accents, split on punctuation, then greedy longest-match WordPiece. Whatever a fast tokenizer
does differently for other settings is not reachable from here.

The vocabulary is Google's ``vocab.txt`` for `bert-base-uncased` (Apache-2.0), byte-identical to
what upstream downloads. See ``NOTICE``.
"""

from __future__ import annotations

import gzip
import unicodedata
from functools import lru_cache
from pathlib import Path

__all__ = ["Tokenizer", "vocabulary"]

_VOCAB = Path(__file__).resolve().parent.parent / "assets" / "vocab.txt.gz"

_UNK = "[UNK]"
_CLS = "[CLS]"
_SEP = "[SEP]"

#: Longest WordPiece BERT will attempt before giving up on a word and emitting ``[UNK]``.
_MAX_CHARS_PER_WORD = 100


@lru_cache(maxsize=1)
def vocabulary() -> dict[str, int]:
    """Return token -> id, read once and kept for the life of the process.

    Read as a list and enumerated rather than split on newlines and filtered: a vocabulary entry
    may be an empty string, and dropping it would shift every id after it by one -- which loads,
    runs, and reads the wrong word.
    """
    with gzip.open(_VOCAB, "rt", encoding="utf-8") as handle:
        return {line.rstrip("\n"): index for index, line in enumerate(handle)}


def _is_control(char: str) -> bool:
    """Is this a control character? Tab, newline and carriage return are treated as whitespace."""
    if char in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(char).startswith("C")


def _is_whitespace(char: str) -> bool:
    if char in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(char) == "Zs"


def _is_punctuation(char: str) -> bool:
    """BERT's notion of punctuation, which is wider than Unicode's.

    Everything in the ASCII ranges around the alphanumerics counts, plus anything Unicode
    categorises as punctuation. ``$`` and ``+`` are not punctuation to Unicode and are here,
    which is upstream's rule and not a mistake to correct.
    """
    code = ord(char)
    if 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126:
        return True
    return unicodedata.category(char).startswith("P")


def _is_cjk(code: int) -> bool:
    """Is this codepoint a CJK ideograph? Each one is tokenized as its own word."""
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
        or 0x2F800 <= code <= 0x2FA1F
    )


class Tokenizer:
    """`bert-base-uncased` WordPiece: text in, token ids out.

    Examples:
        >>> Tokenizer().encode("person . laptop . cup.")[0][:4]
        [101, 2711, 1012, 12191]
    """

    def __init__(self) -> None:
        self.vocab = vocabulary()
        self.cls_id = self.vocab[_CLS]
        self.sep_id = self.vocab[_SEP]
        self.unk_id = self.vocab[_UNK]

    # --- the basic tokenizer: clean, lowercase, split ---

    @staticmethod
    def _clean(text: str) -> str:
        """Drop control characters and replacement chars; normalise whitespace to a plain space."""
        out = []
        for char in text:
            code = ord(char)
            if code == 0 or code == 0xFFFD or _is_control(char):
                continue
            out.append(" " if _is_whitespace(char) else char)
        return "".join(out)

    @staticmethod
    def _space_cjk(text: str) -> str:
        """Put spaces around every CJK ideograph, so each becomes its own word."""
        out = []
        for char in text:
            if _is_cjk(ord(char)):
                out.extend((" ", char, " "))
            else:
                out.append(char)
        return "".join(out)

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Decompose and drop combining marks. Paired with lowercasing, never on its own."""
        return "".join(c for c in unicodedata.normalize("NFD", text)
                       if unicodedata.category(c) != "Mn")

    @staticmethod
    def _split_punctuation(text: str) -> list[str]:
        """Split a word so every punctuation character stands alone."""
        pieces: list[list[str]] = []
        fresh = True
        for char in text:
            if _is_punctuation(char):
                pieces.append([char])
                fresh = True
            else:
                if fresh:
                    pieces.append([])
                fresh = False
                pieces[-1].append(char)
        return ["".join(piece) for piece in pieces]

    def _basic_tokenize(self, text: str) -> list[str]:
        text = self._space_cjk(self._clean(text))
        words: list[str] = []
        for word in text.split():
            word = self._strip_accents(word.lower())
            words.extend(self._split_punctuation(word))
        return words

    # --- WordPiece ---

    def _wordpiece(self, word: str) -> list[str]:
        """Greedy longest-match-first, with ``##`` on every piece after the first."""
        if len(word) > _MAX_CHARS_PER_WORD:
            return [_UNK]

        pieces: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    found = piece
                    break
                end -= 1
            if found is None:
                # One unmatchable piece makes the *whole word* unknown, not just that piece.
                return [_UNK]
            pieces.append(found)
            start = end
        return pieces

    def tokenize(self, text: str) -> list[str]:
        """Return the WordPiece tokens for *text*, without the ``[CLS]``/``[SEP]`` wrapper."""
        return [piece for word in self._basic_tokenize(text) for piece in self._wordpiece(word)]

    def encode(self, text: str) -> tuple[list[int], list[int], list[int]]:
        """Return ``(input_ids, token_type_ids, attention_mask)`` for one caption.

        Wrapped in ``[CLS]`` and ``[SEP]``. One sequence, so the token types are all zero, and
        nothing is padded -- a single caption is its own longest.
        """
        ids = [self.cls_id]
        ids.extend(self.vocab.get(piece, self.unk_id) for piece in self.tokenize(text))
        ids.append(self.sep_id)
        return ids, [0] * len(ids), [1] * len(ids)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Look up ids for tokens already split, e.g. the separator set."""
        return [self.vocab.get(token, self.unk_id) for token in tokens]
