"""The deployable surface: a photograph in, located strings out.

Replaces upstream's ``Reader``. The split is the same one SAM 2 has -- :meth:`Reader.detect`
finds the lines, :meth:`Reader.read` reads them -- because the two halves have very different
costs and only the first depends on nothing but the picture.

**One line per forward pass.** Upstream's reader has two code paths and takes the per-line one
whenever it is on CPU or its batch size is one, which is its default. That path is not an
optimisation of the other: each line is padded to its *own* width rather than to the widest on
the page, and a batched forward is not bit-identical to single ones anyway -- measured at
1.4e-05 here, which is enough to flip a marginal character. So this reproduces the per-line
path, and gets a result that does not depend on what else happened to be on the page.

**Horizontal lines, then tilted ones.** Also upstream's per-line ordering. Level lines come back
in top-to-bottom order and tilted ones follow, rather than the two being interleaved by
position. It is not a reading order and is not presented as one.

**Nothing is cached.** The split exists because the two halves cost very differently, not because
anything re-runs the cheap one. mozo's other two-stage families cache their expensive half
because a caller can ask a second question about the same picture -- a second vocabulary, another
point to segment from. There is no second question to ask an OCR model: one image has one answer,
so a cache here would hash every pixel of every request to miss every time.
"""

from __future__ import annotations

__all__ = ["CONTRAST_THRESHOLD", "ADJUST_CONTRAST", "Region", "Reader"]

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from . import boxes as _boxes
from . import image as _image
from .checkpoint import load
from .config import Spec
from .text import Alphabet

#: A read scoring below this is tried a second time with its contrast stretched.
CONTRAST_THRESHOLD = 0.1

#: The contrast target that retry aims at.
ADJUST_CONTRAST = 0.5


@dataclass(frozen=True)
class Region:
    """One line of text: where it is, what it says, how sure the model was.

    ``quad`` is four ``(x, y)`` corners in the original image's pixels, clockwise from the top
    left. For a level line it is the rectangle's corners and carries nothing ``bbox`` would not;
    for a tilted one it is the only record of the orientation.
    """

    quad: list
    text: str
    confidence: float


class Reader:
    """Both graphs for one variant, and the pipeline between them."""

    def __init__(self, checkpoint: Path | str, spec: Spec, device: str = "cpu") -> None:
        self.spec = spec
        self.device = torch.device(device)
        self.detector, self.recogniser = load(checkpoint, spec)
        self.detector.to(self.device)
        self.recogniser.to(self.device)
        self.alphabet = Alphabet(spec.characters)

    def detect(self, image: np.ndarray) -> tuple[list, list]:
        """Find every line in ``image``. Returns ``(horizontal, free)``."""
        batch, ratio = _image.for_detector(image)
        with torch.no_grad():
            heatmaps = self.detector(batch.to(self.device))[0].cpu().numpy()
        return _boxes.group(_boxes.rescale(
            _boxes.quads(heatmaps[:, :, 0], heatmaps[:, :, 1]), ratio))

    def _read_one(self, crop: np.ndarray, width: int) -> tuple[str, float]:
        """Read one crop, retrying with stretched contrast if the first answer looks weak.

        The retry is part of the published model rather than a convenience: a crop below
        :data:`CONTRAST_THRESHOLD` is read twice and the higher-scoring answer wins, so skipping
        it changes what low-contrast text says, not merely how confident it looks.
        """
        def run(contrast: float) -> tuple[str, float]:
            batch = _image.align(crop, width, contrast=contrast).to(self.device)
            with torch.no_grad():
                return self.alphabet.decode(self.recogniser(batch))[0]

        text, confidence = run(0.0)
        if confidence >= CONTRAST_THRESHOLD:
            return text, confidence
        retry_text, retry_confidence = run(ADJUST_CONTRAST)
        return (text, confidence) if confidence > retry_confidence else (retry_text, retry_confidence)

    def read(self, grey: np.ndarray, horizontal: list, free: list) -> list[Region]:
        """Read the lines ``detect`` found out of the greyscale page."""
        regions = []
        for line, is_free in [(box, False) for box in horizontal] + [(q, True) for q in free]:
            cut = _image.line_image(line, grey, is_free=is_free)
            if cut is None:
                continue
            quad, crop, width = cut
            text, confidence = self._read_one(crop, width)
            regions.append(Region(quad, text, confidence))
        return regions

    def __call__(self, image: np.ndarray) -> list[Region]:
        """Find and read every line in an RGB image.

        ``image`` is RGB, which is mozo's contract and matches what upstream's reader builds
        when it is handed a path. Its array entry point instead documents its input as BGR and
        derives the greyscale page with ``COLOR_BGR2GRAY``, so passing it an RGB array
        channel-swaps the crops the recogniser sees -- 27 grey levels on a colour photograph.
        Taking RGB and converting from RGB is the same pipeline without that trap.
        """
        horizontal, free = self.detect(image)
        return self.read(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), horizontal, free)
