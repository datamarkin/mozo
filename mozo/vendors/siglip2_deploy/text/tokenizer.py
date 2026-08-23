# SPDX-License-Identifier: Apache-2.0
"""SigLIP 2's tokenizer: Gemma's byte-pair encoder, with SigLIP 2's normalisation in front.

Nothing here is shaped like CLIP's tokenizer, and every difference below produces plausible wrong
ids rather than an error.

**It lowercases, and the published tokenizer config does not say so.** The repositories declare
``tokenizer_class: GemmaTokenizer`` with ``do_lower_case: true`` -- but ``do_lower_case`` is not a
``GemmaTokenizer`` parameter and nothing acts on it, so ``AutoTokenizer.from_pretrained`` returns a
tokenizer that preserves case. The authors' own demo notebook preprocesses with
``lower(key="text")`` under a note reading *"SigLIP 2 models work best with lowercase texts"*, and
``transformers``' own ``Siglip2Tokenizer`` prepends a ``Lowercase`` normaliser for what its
docstring calls the *"SigLIP2 training default"*. This package follows training. See
``PROVENANCE.md``; the divergence is from the published config, not from the model.

**The word split never happens.** The normaliser replaces every space with ``U+2581`` first, so the
pre-tokeniser's ``Split(" ")`` finds nothing to split on and byte-pair merging runs over the whole
caption as a single stream. Merges cross word boundaries. CLIP's tokenizer merges within a word and
copying that structure gives different ids.

**There is no prefix space.** Most SentencePiece tokenizers prepend one; this one does not, so a
leading word is unprefixed -- ``"a photo"`` opens with ``'a'`` (235250), where ``'▁a'`` is 476.

**Byte fallback is not a contiguous table.** ``<0x00>`` is id 217, but ``<0x09>`` does not exist at
all and a literal ``'\\x01'`` is an ordinary piece at 238213. The table is read out of the
vocabulary rather than computed from a base offset.

The vocabulary ships beside this module, derived once from the published ``tokenizer.json`` by
``tools/fetch/siglip2.py --derive-vocab``. See ``PROVENANCE.md`` for its hash.
"""

from __future__ import annotations

import gzip
import json
import re
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path
from typing import NamedTuple, Sequence

import torch

from ..config import CONTEXT

__all__ = ["Tokenizer"]

#: The derived vocabulary: pieces by id, merge rules as pairs of ids in priority order, the
#: reserved names that match ahead of normalisation, and the case-folding table.
VOCAB_PATH = Path(__file__).resolve().parent.parent / "assets" / "gemma_bpe.json.gz"

#: Reserved ids, which the vocabulary puts first and this package never looks up by name.
PAD, EOS = 0, 1

#: What the normaliser puts in place of a space.
UNDERLINE = "▁"


class Tables(NamedTuple):
    """Everything read out of the vocabulary asset, named rather than positional.

    Seven lookups of which two have the same type; unpacking them in order would run and be wrong
    if any two were ever swapped.
    """

    #: Every piece, indexed by its id.
    pieces: list[str]
    #: The reverse: piece -> id.
    ids: dict[str, int]
    #: Adjacent pair -> the priority of merging it. Lower wins.
    ranks: dict[tuple[int, int], int]
    #: The id a merge produces, indexed by that merge's rank.
    merged: list[int]
    #: Byte -> the ``<0xNN>`` piece that stands for it.
    fallback: dict[int, int]
    #: The reserved names that match ahead of normalisation, longest first.
    added: "re.Pattern[str]"
    #: Normalisation, as one translation table: case folding plus the space substitution.
    normalise: dict[int, str]


@lru_cache(maxsize=1)
def _vocabulary() -> Tables:
    """Read the asset once and build the lookups tokenizing needs.

    Cached at module level rather than per instance: the tables are 256,000 pieces and 580,604
    merges, and two encoders in one process should not hold two copies of them.
    """
    data = json.loads(gzip.decompress(VOCAB_PATH.read_bytes()))
    pieces: list[str] = data["pieces"]
    ids = {piece: index for index, piece in enumerate(pieces)}

    # ``merged`` is a list indexed by rank rather than a second dict on ``ranks``'s own keys. The
    # merge loop pops a rank off the heap before it needs the result, so indexing by that rank
    # costs nothing and saves both the duplicate 580,604-key hash table -- about 11 MB -- and a
    # second hash of a pair that has just been hashed.
    ranks, merged = {}, []
    for rank, (left, right) in enumerate(data["merges"]):
        ranks[(left, right)] = rank
        merged.append(ids[pieces[left] + pieces[right]])

    # The byte-fallback table, read out of the vocabulary because it is neither complete nor
    # contiguous: <0x09> is simply absent.
    fallback = {byte: ids[f"<0x{byte:02X}>"] for byte in range(256) if f"<0x{byte:02X}>" in ids}

    # Longest first, so that "<unused10>" wins over the "<unused1>" that prefixes it. Python's
    # alternation takes the first branch that matches at a position, not the longest, so the order
    # here is what makes the match longest rather than merely leftmost.
    added = re.compile("|".join(re.escape(token) for token in
                                sorted(data["added"], key=len, reverse=True)))

    # One table for both normalisation steps, so a segment is walked once rather than twice. The
    # case mappings are carried rather than left to ``str.lower()``: see
    # tools/fetch/siglip2.py::case_folding. The space substitution joins them because a space is
    # not a character the case table has any opinion about, so the two cannot interfere.
    normalise = {int(codepoint): folded for codepoint, folded in data["lower"].items()}
    normalise[ord(" ")] = UNDERLINE
    return Tables(pieces, ids, ranks, merged, fallback, added, normalise)


class Tokenizer:
    """Turn phrases into the fixed-width token ids the text tower expects.

    Examples:
        >>> Tokenizer()(["a photo of a cat"]).shape        # doctest: +SKIP
        torch.Size([1, 64])
    """

    def __init__(self) -> None:
        tables = _vocabulary()
        self.pieces, self.ids = tables.pieces, tables.ids
        self._ranks, self._merged = tables.ranks, tables.merged
        self._fallback, self._added = tables.fallback, tables.added
        self._normalise = tables.normalise

    def _symbols(self, text: str) -> list[int]:
        """One id per character, dropping to UTF-8 bytes for anything the vocabulary lacks.

        The byte table is indexed rather than probed. It has exactly one hole -- ``<0x09>`` is not
        in the vocabulary -- and the only character whose UTF-8 contains that byte is the tab,
        which is an ordinary piece and never reaches the fallback. So the lookup cannot miss; if
        the vocabulary ever changed underneath it, a ``KeyError`` naming the byte is a better
        outcome than quietly tokenizing to something else.
        """
        out: list[int] = []
        for character in text:
            index = self.ids.get(character)
            if index is not None:
                out.append(index)
                continue
            try:
                encoded = character.encode("utf-8")
            except UnicodeEncodeError:
                # A lone surrogate, which is a valid Python str and not valid text. The reference
                # refuses it too, with a TypeError from deep inside the tokenizers library; this
                # says what is wrong instead.
                raise ValueError(
                    f"{text!r} contains an unpaired surrogate ({character!r}) and is not encodable "
                    "text; decode the input properly before tokenizing it"
                ) from None
            out.extend(self._fallback[byte] for byte in encoded)
        return out

    def _merge(self, symbols: list[int]) -> list[int]:
        """Apply merge rules in priority order until none applies.

        A heap over adjacent pairs rather than a rescan per merge. The rescan is the obvious
        implementation and is quadratic in the caption's length -- which matters here and did not
        for CLIP, because there the merging happened inside one short word and here it runs over
        the whole phrase at once. Entries are left stale rather than removed, and skipped when
        popped, which is the standard way to avoid deleting from the middle of a heap.
        """
        if len(symbols) < 2:
            return symbols

        following = list(range(1, len(symbols) + 1))
        preceding = list(range(-1, len(symbols) - 1))
        alive = [True] * len(symbols)

        heap: list[tuple[int, int, int]] = []
        for position in range(len(symbols) - 1):
            rank = self._ranks.get((symbols[position], symbols[position + 1]))
            if rank is not None:
                heappush(heap, (rank, position, position + 1))

        while heap:
            rank, left, right = heappop(heap)
            # Stale: one side has been merged away, or they are no longer neighbours.
            if not alive[left] or not alive[right] or following[left] != right:
                continue
            if self._ranks.get((symbols[left], symbols[right])) != rank:
                continue

            symbols[left] = self._merged[rank]
            alive[right] = False
            after = following[right]
            following[left] = after
            if after < len(symbols):
                preceding[after] = left

            before = preceding[left]
            if before >= 0:
                rank = self._ranks.get((symbols[before], symbols[left]))
                if rank is not None:
                    heappush(heap, (rank, before, left))
            if after < len(symbols):
                rank = self._ranks.get((symbols[left], symbols[after]))
                if rank is not None:
                    heappush(heap, (rank, left, after))

        return [symbol for symbol, kept in zip(symbols, alive) if kept]

    def _encode_segment(self, text: str) -> list[int]:
        """Normalise and merge one stretch of ordinary text."""
        return self._merge(self._symbols(text.translate(self._normalise)))

    def encode(self, text: str) -> list[int]:
        """One phrase to its ids, with the end marker but no padding.

        Reserved names written out in the prompt -- ``<eos>``, ``<pad>``, ``<unused7>`` and the
        246 others -- are matched **before** normalising and become that token, exactly as they do
        upstream. So a caption containing the literal text ``<eos>`` ends up carrying a control
        token rather than five characters. That is upstream's behaviour and this reproduces it
        rather than correcting it; ``clip_deploy`` makes the same call about a prompt containing a
        literal ``<|endoftext|>``.

        Matching happens on the raw text because these tokens are declared ``normalized: false``,
        so ``<EOS>`` is ordinary text and only ``<eos>`` is the token.
        """
        tokens: list[int] = []
        position = 0
        for match in self._added.finditer(text):
            tokens += self._encode_segment(text[position:match.start()])
            tokens.append(self.ids[match.group()])
            position = match.end()
        return tokens + self._encode_segment(text[position:]) + [EOS]

    def __call__(self, texts: str | Sequence[str]) -> torch.Tensor:
        """Encode one phrase or many to ``(N, 64)`` int64, right-padded.

        Args:
            texts: A phrase, or a sequence of them.

        Returns:
            ``(N, CONTEXT)`` int64 token ids, padded on the right with ``<pad>``.

            **The padding is not incidental.** SigLIP 2 trained on fully padded sequences, its
            text tower attends the padding, and it pools the last position -- so a row that stopped
            early would be pooled at a real token instead of a pad and would mean something else
            entirely. There is no option to pack.

        Raises:
            ValueError: If nothing is given, if a phrase is blank, or if one does not fit.

        Note:
            Upstream truncates instead, keeping the end marker (``eos="sticky"``). Not carried: a
            prompt shortened without saying so is a different prompt, and only the caller can
            decide what to drop. This is the same choice ``clip_deploy`` makes about
            ``truncate=True``.
        """
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            raise ValueError("give at least one phrase to encode")
        if any(not phrase.strip() for phrase in batch):
            raise ValueError("every phrase must say something; one of them is blank")

        rows = torch.full((len(batch), CONTEXT), PAD, dtype=torch.long)
        for index, phrase in enumerate(batch):
            tokens = self.encode(phrase)
            if len(tokens) > CONTEXT:
                raise ValueError(
                    f"{phrase!r} is {len(tokens)} tokens; SigLIP 2's context is {CONTEXT}")
            rows[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return rows
