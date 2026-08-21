# SPDX-License-Identifier: Apache-2.0
"""OWLv2's prompt path: the CLIP byte-pair tokenizer and the text tower."""

from .encoder import TextTower
from .tokenizer import Tokenizer

__all__ = ["TextTower", "Tokenizer"]
