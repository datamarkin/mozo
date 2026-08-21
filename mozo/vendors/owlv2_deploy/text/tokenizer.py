# SPDX-License-Identifier: MIT
"""The CLIP byte-pair tokenizer OWLv2 prompts go through.

``assets/bpe_simple_vocab_16e6.txt.gz`` is byte-identical to the file published with
``openai/CLIP`` (sha256 ``924691ac...48d6804a``), so both it and the algorithm below are MIT.
Its vocabulary is identical, id for id, to the ``vocab.json`` and ``merges.txt`` Google publishes
beside the weights -- 49,408 entries and 48,894 merges -- which is what makes shipping the 1.3 MB
gzip instead of the 1.6 MB pair a compression choice rather than a different tokenizer. That was
established against Google's files by hand; what travels with the repository is the digest, pinned
in ``tools/verify/owlv2.py``, because neither of those files is carried here.

**This is not OpenAI's ``SimpleTokenizer``, and the difference is not cosmetic.** OWLv2 is
prompted through ``transformers``' ``CLIPTokenizer``, which is what the published checkpoint's
``tokenizer_config.json`` names, so that is the behaviour the weights were fine-tuned under and
the behaviour this file reproduces. Three things follow, all found by comparing ids rather than by
reading:

**No ``ftfy``, and no HTML unescaping.** OpenAI's cleaner runs ``ftfy.fix_text`` and then
``html.unescape`` twice. ``CLIPTokenizer`` normalises with NFC, collapses whitespace and
lowercases -- that is all. On ``"&amp; entities"`` the two disagree from the second token onward.

**``!`` is a token in its own right, and its id is zero.** The published config registers ``!``
as the padding token, which puts it in the added-token table, which makes the tokenizer split it
out of any prompt before byte-pair encoding ever sees it. So ``"cat!"`` is ``[bos, cat</w>, !,
eos]`` and not ``[bos, cat</w>, !</w>, eos]``, and the ``!`` carries id 0.

**Which is why padding cannot be inferred from the ids.** Zero means "padding" *and* "exclamation
mark", so :meth:`Tokenizer.__call__` returns the mask it built rather than leaving the caller to
recover it. ``ids != 0`` is the obvious reading, it is what a sibling package here does for a
tokenizer where it happens to be safe, and on ``"a cat!"`` it would silently drop a real token out
of the attention.

**The context length is 16, not 77.** Every general-purpose CLIP model uses 77; OWLv2's position
embedding is ``(16, width)``. Longer prompts are truncated with the end-of-text id kept in the
last slot, so a truncated prompt is still terminated.
"""

from __future__ import annotations

import gzip
import unicodedata
from functools import lru_cache
from pathlib import Path

import regex
import torch

__all__ = ["Tokenizer", "VOCAB_PATH"]

#: The vocabulary shipped with this package.
VOCAB_PATH = Path(__file__).resolve().parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

#: Where the merge list stops. CLIP's file carries a header line and trailing entries that are
#: not merges; upstream slices exactly this range and the vocabulary ids depend on it.
MERGE_LIMIT = 49152 - 256 - 2 + 1

START_TOKEN = "<|startoftext|>"
END_TOKEN = "<|endoftext|>"

#: Split out before byte-pair encoding, because the published tokenizer config lists all three in
#: its added-token table. The two markers are there by convention; ``!`` is there only because it
#: was chosen as the padding token, and that accident changes how every prompt containing one is
#: encoded. Order matters -- the markers must be tried before ``!``, or ``<|startoftext|>`` would
#: never match as a unit.
ADDED = ((START_TOKEN, 49406), (END_TOKEN, 49407), ("!", 0))


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
    """CLIP byte-pair encoding, as OWLv2's published config sets it up.

    Args:
        vocab_path: The gzipped merge list. Defaults to the vendored copy.
        context_length: Fixed width of the returned id matrix.
    """

    def __init__(self, vocab_path: Path = VOCAB_PATH, context_length: int = 16):
        self.context_length = context_length

        self.byte_encoder = bytes_to_unicode()
        with gzip.open(vocab_path) as handle:
            merges = handle.read().decode("utf-8").split("\n")
        merges = [tuple(merge.split()) for merge in merges[1:MERGE_LIMIT]]

        vocabulary = list(self.byte_encoder.values())
        vocabulary += [token + "</w>" for token in vocabulary]
        vocabulary += ["".join(merge) for merge in merges]
        vocabulary += [START_TOKEN, END_TOKEN]

        self.encoder = {token: index for index, token in enumerate(vocabulary)}
        self.ranks = {merge: rank for rank, merge in enumerate(merges)}
        self._cache: dict[str, str] = {}

        self.start_id = self.encoder[START_TOKEN]
        self.end_id = self.encoder[END_TOKEN]

        # ``\p{L}`` and ``\p{N}`` are Unicode categories, which the standard library's ``re``
        # cannot express -- hence ``regex``. Substituting ``\w`` here silently changes how
        # accented and non-Latin prompts split. This is the pre-tokenizer pattern verbatim, minus
        # the two markers: those are handled by ``ADDED`` before this runs.
        self.pattern = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
        )

        # One alternation over the added tokens, capturing so ``split`` keeps them.
        self.added = regex.compile("(" + "|".join(regex.escape(t) for t, _ in ADDED) + ")")
        self.added_ids = dict(ADDED)

    def clean(self, text: str) -> str:
        """Normalise, collapse whitespace, lowercase.

        The three steps ``CLIPTokenizer`` applies and no others. Anything more -- repairing
        mojibake, unescaping entities -- moves the ids away from what the weights were trained on.
        """
        text = unicodedata.normalize("NFC", text)
        return regex.sub(r"\s+", " ", text).lower()

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
        for span in self.added.split(self.clean(text)):
            if span in self.added_ids:
                ids.append(self.added_ids[span])
                continue
            for token in self.pattern.findall(span):
                token = "".join(self.byte_encoder[byte] for byte in token.encode("utf-8"))
                ids.extend(self.encoder[symbol] for symbol in self.merge(token).split(" "))
        return ids

    def __call__(self, texts: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize one prompt or a batch into a fixed-width id matrix and its mask.

        Args:
            texts: A prompt, or a list of them.

        Returns:
            ``ids`` and ``mask``, both ``(len(texts), context_length)`` int64. Rows are
            zero-padded on the right; ``mask`` is 1 where the row carries a real token. The mask
            is returned rather than derived because id 0 is also ``!`` -- see the module docstring.
        """
        if isinstance(texts, str):
            texts = [texts]
        ids = torch.zeros(len(texts), self.context_length, dtype=torch.long)
        mask = torch.zeros(len(texts), self.context_length, dtype=torch.long)
        for row, text in enumerate(texts):
            tokens = [self.start_id, *self.encode(text), self.end_id]
            if len(tokens) > self.context_length:
                tokens = tokens[: self.context_length]
                # Keep the sequence terminated even when the prompt did not fit.
                tokens[-1] = self.end_id
            ids[row, : len(tokens)] = torch.tensor(tokens)
            mask[row, : len(tokens)] = 1
        return ids, mask
