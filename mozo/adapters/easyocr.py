# SPDX-License-Identifier: Apache-2.0
"""mozo's EasyOCR adapter: an image in, located strings out.

The first family here that answers in words. Every other one names a class from a fixed
vocabulary or measures something per pixel; this one reads what is written. PixelFlow keeps the
two apart -- ``class_name`` is which class out of a vocabulary the model was trained on, ``text``
is content it produced that belongs to no vocabulary -- so detections from here carry ``text``
and leave ``class_id`` and ``class_name`` as ``None``. There is no ``labels.json`` beside these
weights and this adapter does not consult :mod:`mozo.labels`.

**A variant is a script, not a language.** ``english`` and ``latin`` are two checkpoints, not two
settings; ``latin`` covers 41 languages and reads every character its charset holds. Upstream
instead picks a checkpoint from a language list and then suppresses characters outside those
languages at decode time, which makes its output depend on something that is not a property of
the weights. See the vendor's ``PROVENANCE.md``.

**The quad is kept.** Real-world text is rotated, and ``bbox`` alone throws the orientation away,
so each detection carries its four corners in ``segments`` with the axis-aligned hull in
``bbox``. Transforms move all four corners.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from ..image import load_image
from ..runtimes import get_default_device, select_runtime
from ..vendors.easyocr_deploy import SPECS, VARIANTS, Reader
from ..weights import artifacts, resolve

try:
    import pixelflow as pf
except ImportError:
    raise ImportError("PixelFlow is not installed. Install it with: pip install pixelflow") from None

__all__ = ["EasyOCRPredictor"]

#: The family this adapter serves, as the registry and the manifest name it.
FAMILY = "easyocr"


class EasyOCRPredictor:
    """Text detection and recognition: find every line, and read it.

    Args:
        variant: Which script to read. See :attr:`VARIANTS`.
        device: Where to run. Defaults to the best available.
        runtime: Which artifact to execute. ``auto`` picks the fastest published one.
        checkpoint_path: Your own checkpoint, instead of the published weights.
        revision: Which published revision to use. Defaults to the newest.

    Attributes:
        variant: The variant in use.
        device: Where it runs.
        runtime: Which artifact is executing.

    Examples:
        >>> model = EasyOCRPredictor()                       # doctest: +SKIP
        >>> found = model.predict("sign.jpg")                # doctest: +SKIP
        >>> [d.text for d in found]                          # doctest: +SKIP
        ['EXIT 42']
        >>> found[0].class_name is None                      # doctest: +SKIP
        True
    """

    #: Upstream publishes seventeen recognisers; these five are 88% of its own download counts,
    #: and all five share one network. See ``PROVENANCE.md``.
    VARIANTS = list(VARIANTS)

    #: Which runtimes this adapter can execute. Only torch: the vendor builds two torch modules
    #: from a checkpoint and has no graph path. Declared rather than checked afterwards so
    #: ``auto`` cannot pick an artifact this would then have to refuse.
    EXECUTES = ("torch",)

    def __init__(
        self,
        variant: str = "english",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        # Checked even with your own checkpoint: the variant selects an alphabet size here, so a
        # wrong one is not a mislabelling but a strict load that cannot succeed.
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {self.VARIANTS}")

        self.variant = variant
        self.device = device or get_default_device()
        self.runtime = (
            "torch-fp32" if checkpoint_path is not None
            else select_runtime(
                self.device,
                artifacts(FAMILY, variant, revision=revision),
                runtime,
                executes=self.EXECUTES,
            )
        )

        weights = (Path(checkpoint_path) if checkpoint_path
                   else resolve(FAMILY, variant, self.runtime, revision=revision))
        self._reader = Reader(weights, SPECS[variant], device=self.device)
        print(f"EasyOCR {variant} ready on {self.device} via {self.runtime}.")

    def predict(
        self,
        image: Union[str, Path, bytes, np.ndarray],
    ) -> "pf.detections.Detections":
        """Find and read every line of text in ``image``.

        Args:
            image: A file path, encoded image bytes, or an ``HxWx3`` RGB ``uint8`` array.

        Returns:
            A PixelFlow ``Detections``, one per line. Each carries the read string in ``text``,
            the model's confidence in ``confidence``, the four corners as read in ``segments``
            and their axis-aligned hull in ``bbox``. ``class_id`` and ``class_name`` are
            ``None``: OCR reads content, it does not pick a class out of a vocabulary.

            Level lines come back top to bottom, followed by any tilted ones. That is upstream's
            ordering and it is not a reading order -- a two-column page interleaves.

            A line the detector found but the recogniser read as empty is kept, with ``text``
            set to ``""``. The box was still detected, and whether an empty read is worth having
            is the caller's call, not this adapter's.
        """
        regions = self._reader(load_image(image))
        quads = [np.asarray(region.quad, dtype=float) for region in regions]
        return pf.detections.from_arrays(
            boxes=[[quad[:, 0].min(), quad[:, 1].min(), quad[:, 0].max(), quad[:, 1].max()]
                   for quad in quads],
            scores=[region.confidence for region in regions],
            texts=[region.text for region in regions],
            segments=quads,
        )
