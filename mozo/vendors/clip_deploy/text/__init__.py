# SPDX-License-Identifier: Apache-2.0
"""The text half: CLIP's byte-pair tokenizer and its transformer tower."""

from .tokenizer import CONTEXT_LENGTH, Tokenizer

__all__ = ["CONTEXT_LENGTH", "Tokenizer"]
