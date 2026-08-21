"""What each published variant is: which charset, and which of the two networks' shapes.

A variant here is a *script*, not a language. Upstream conflates the two -- asking its reader for
``['en']`` picks the Latin recogniser and then suppresses every character outside English at
decode time -- so its output depends on a language list that is no property of the weights. A
mozo variant is one checkpoint and decodes everything that checkpoint knows.

The charsets live beside this file rather than inside it. They are part of the model definition,
not of the weights, but the Chinese one is 6,718 characters and a literal that size in a module
is not something anyone can read past.
"""

from __future__ import annotations

__all__ = ["Spec", "SPECS", "VARIANTS", "charset"]

from dataclasses import dataclass
from pathlib import Path

_ASSETS = Path(__file__).parent / "assets"


def charset(variant: str) -> str:
    """The alphabet ``variant`` was trained on, in index order, without the CTC blank."""
    return (_ASSETS / f"{variant}.txt").read_text(encoding="utf-8")


@dataclass(frozen=True)
class Spec:
    """One published variant, which is to say one alphabet.

    The five published second-generation recognisers are the same network in every respect but
    the symbols they were trained on, so the alphabet is all there is to say about them. Where
    the shapes matter -- the width of the final linear layer -- :attr:`num_class` derives them.
    """

    variant: str

    @property
    def characters(self) -> str:
        return charset(self.variant)

    @property
    def num_class(self) -> int:
        """Alphabet plus one, for CTC's blank at index 0."""
        return len(self.characters) + 1


#: Published variants, most used first -- the order is the download counts on upstream's own
#: releases, which is also the order anyone browsing them should see.
SPECS: dict[str, Spec] = {
    variant: Spec(variant) for variant in
    ("english", "latin", "chinese-simplified", "japanese", "korean")
}

VARIANTS = list(SPECS)
