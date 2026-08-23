# SPDX-License-Identifier: Apache-2.0
"""CLIP's byte-pair tokenizer, over the vocabulary that ships beside this file.

A prompt becomes a fixed-width row of 77 token ids: a start marker, the text, an end marker, and
zeros to the end. The end marker matters beyond padding -- it is the highest id in the vocabulary,
which is how the text tower finds where a variable-length prompt stopped without being told its
length. See :mod:`~mozo.vendors.clip_deploy.text.encoder`.

``ftfy`` and ``regex`` are both core mozo dependencies and both are load-bearing here rather than
conveniences. ``ftfy`` repairs mis-decoded input the way upstream does; the split pattern uses
Unicode categories (``\\p{L}``, ``\\p{N}``) that the standard library's ``re`` cannot express.

The vocabulary is OpenAI's ``bpe_simple_vocab_16e6.txt.gz``, byte-identical to the file published
with CLIP (sha256 924691ac…, 1,356,917 bytes) and to the copies :mod:`mozo.vendors.owlv2_deploy`
and :mod:`mozo.vendors.sam3_deploy` already carry. A vendored package may not import another, so
this is a third copy of one file rather than a shared asset.
"""

from __future__ import annotations

import gzip
import html
from functools import lru_cache
from pathlib import Path

import ftfy
import regex

__all__ = ["CONTEXT_LENGTH", "Tokenizer"]

_VOCAB = Path(__file__).resolve().parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

#: Row width every CLIP model was trained at. Not a padding convenience -- the position embedding
#: has exactly this many rows, so a longer prompt has nowhere to go.
CONTEXT_LENGTH = 77

#: Merges to read from the vocabulary file. Upstream slices ``[1:49152-256-2+1]``: one header line
#: is skipped, and the tail is left out so the final vocabulary lands exactly on 49,408 entries --
#: 256 byte symbols, 256 with a word-end marker, the merges, and the two special tokens.
_MERGE_SLICE = slice(1, 49152 - 256 - 2 + 1)

_START = "<|startoftext|>"
_END = "<|endoftext|>"

#: Upstream's split pattern. Contractions first, then letters, then single digits, then runs of
#: anything else. ``[\p{N}]`` is deliberately one digit at a time, so "2024" is four tokens.
_PATTERN = regex.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
    regex.IGNORECASE,
)


@lru_cache(maxsize=1)
def byte_encoder() -> dict[int, str]:
    """Map every byte to a printable character.

    Byte-pair merges operate on text, and a raw byte may be whitespace or a control character that
    the merge rules cannot carry. So the printable ASCII and Latin-1 ranges stand for themselves
    and everything else is lifted into a private range above 255, reversibly.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapped = printable[:]
    spare = 0
    for byte in range(2**8):
        if byte not in printable:
            printable.append(byte)
            mapped.append(2**8 + spare)
            spare += 1
    return dict(zip(printable, (chr(code) for code in mapped)))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Every adjacent pair of symbols in *word*."""
    return {(first, second) for first, second in zip(word, word[1:])}


def _clean(text: str) -> str:
    """Repair mis-decoded text, unescape HTML twice, and collapse whitespace.

    ``html.unescape`` is applied twice because upstream does: web-scraped captions carry
    double-escaped entities, and one pass leaves ``&amp;amp;`` as ``&amp;``.
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return regex.sub(r"\s+", " ", text).strip()


class Tokenizer:
    """CLIP's byte-pair encoder: text in, a fixed-width row of token ids out.

    Examples:
        >>> Tokenizer()("a diagram").shape                 # doctest: +SKIP
        torch.Size([1, 77])
    """

    def __init__(self) -> None:
        merges = gzip.open(_VOCAB).read().decode("utf-8").split("\n")[_MERGE_SLICE]
        merges = [tuple(merge.split()) for merge in merges]

        symbols = list(byte_encoder().values())
        vocabulary = symbols + [symbol + "</w>" for symbol in symbols]
        vocabulary.extend("".join(merge) for merge in merges)
        vocabulary.extend((_START, _END))

        self.encoder: dict[str, int] = {token: index for index, token in enumerate(vocabulary)}
        self.ranks: dict[tuple[str, str], int] = {
            merge: rank for rank, merge in enumerate(merges)
        }
        self.byte_encoder = byte_encoder()
        self.start_id = self.encoder[_START]
        self.end_id = self.encoder[_END]
        #: Seeded with the two special tokens so they survive the merge loop untouched.
        self._cache: dict[str, str] = {_START: _START, _END: _END}

    def bpe(self, token: str) -> str:
        """Merge *token* into subwords, most frequent pair first, space separated."""
        if token in self._cache:
            return self._cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.ranks.get(pair, float("inf")))
            if bigram not in self.ranks:
                break
            first, second = bigram

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
            if len(word) == 1:
                break
            pairs = _pairs(word)

        joined = " ".join(word)
        self._cache[token] = joined
        return joined

    def encode(self, text: str) -> list[int]:
        """Return the token ids for *text*, without the start and end markers."""
        ids: list[int] = []
        for token in regex.findall(_PATTERN, _clean(text).lower()):
            token = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
            ids.extend(self.encoder[piece] for piece in self.bpe(token).split(" "))
        return ids

    def __call__(self, texts: str | list[str]) -> "torch.Tensor":  # noqa: F821
        """Return ``(len(texts), 77)`` int32 token ids, start- and end-marked and zero-padded.

        Raises:
            ValueError: If a prompt does not fit. Upstream offers a ``truncate`` flag that keeps
                the first 76 tokens and overwrites the last with the end marker; it is not carried,
                because a prompt silently shortened is a different prompt and the caller is the only
                one who can decide what to drop.
        """
        import torch

        if isinstance(texts, str):
            texts = [texts]

        rows = torch.zeros(len(texts), CONTEXT_LENGTH, dtype=torch.int)
        for row, text in enumerate(texts):
            ids = [self.start_id, *self.encode(text), self.end_id]
            if len(ids) > CONTEXT_LENGTH:
                raise ValueError(
                    f"{text!r} is {len(ids)} tokens, over CLIP's {CONTEXT_LENGTH}-token context. "
                    f"Shorten it."
                )
            rows[row, : len(ids)] = torch.tensor(ids, dtype=torch.int)
        return rows
