# SPDX-License-Identifier: Apache-2.0
"""The two towers, and the space they meet in.

    image  ──► VisionTransformer ──► 512 floats ──► normalise ──┐
                                                                 ├──► dot product = similarity
    prompt ──► TextTransformer   ──► 512 floats ──► normalise ──┘

There is no third module. CLIP is two encoders that were trained together until their outputs
landed in the same space, and everything else -- zero-shot classification, retrieval, clustering --
is arithmetic someone does afterwards on the vectors.

The towers are built separately rather than as one ``nn.Module`` with two children, because they
are used separately: an ingest job wants only the image side, a query service only the text side.
:mod:`~mozo.vendors.clip_deploy.predictor` builds whichever is asked for and never the other.
"""

from __future__ import annotations

from torch import Tensor

from .config import Spec
from .text.encoder import TextTransformer
from .text.tokenizer import CONTEXT_LENGTH
from .vision.vit import VisionTransformer

__all__ = ["VOCAB_SIZE", "build_text_tower", "build_vision_tower", "normalise"]

#: Rows in CLIP's token embedding: 256 byte symbols, 256 word-final, the merges, and two markers.
VOCAB_SIZE = 49408


def build_vision_tower(spec: Spec) -> VisionTransformer:
    """Build the image tower *spec* describes, on whatever device is current."""
    return VisionTransformer(
        resolution=spec.resolution,
        patch=spec.patch,
        width=spec.vision_width,
        layers=spec.vision_layers,
        heads=spec.vision_heads,
        embed_dim=spec.embed_dim,
    )


def build_text_tower(spec: Spec) -> TextTransformer:
    """Build the text tower *spec* describes, on whatever device is current."""
    return TextTransformer(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        width=spec.text_width,
        layers=spec.text_layers,
        heads=spec.text_heads,
        embed_dim=spec.embed_dim,
    )


def normalise(vectors: Tensor) -> Tensor:
    """Scale each row to unit length, so a dot product between two of them is a cosine.

    Done here rather than left to the caller because a vector that leaves this package unnormalised
    is one a caller can silently compare wrongly -- and two callers normalising differently is a
    class of bug that never raises and never looks like a bug.
    """
    return vectors / vectors.norm(dim=-1, keepdim=True)
