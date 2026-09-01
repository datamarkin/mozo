"""BEN2 background removal -- an alpha matte, not a mask.

The architecture lives in :mod:`mozo.vendors.ben2_deploy`, extracted from Prama LLC's single-file
``BEN2.py`` and reduced to inference. The weights come from :func:`mozo.weights.resolve`.
Pre- and post-processing come from the vendor, so the pipeline is upstream's step for step --
verified bit-identical to it on CPU in float32.

Every other family in mozo answers with a decision: a box, a class, a character, a binary mask.
This one answers with an **opacity** -- a per-pixel number for how much of that pixel is
foreground. That is what a compositor needs and what a segmenter cannot give: threshold a mask
around a head of hair and the hair goes with the background.

**The default alpha is not a probability.** Upstream ends its postprocess with a per-image
min-max stretch, so 255 means "the most foreground pixel in this image" rather than "certainly
foreground". mozo reproduces that by default, because it is what the model's own users expect
back, and offers the calibrated sigmoid under ``stretch=False``. Compare a stretched alpha within
an image, never across two.

    >>> model = Ben2Predictor()                      # doctest: +SKIP
    >>> alpha = model.predict(image)                 # doctest: +SKIP
    >>> alpha.shape                                  # doctest: +SKIP
    (1281, 1920)
    >>> rgba = model.cutout(image, refine=True)      # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from ..image import load_image
from ..runtimes import get_default_device, select_runtime
from ..vendors.ben2_deploy import Predictor
from ..weights import artifacts, resolve


class Ben2Predictor:
    """The loaded BEN2 model, ready to matte.

    Args:
        variant: ``"base"``. Upstream publishes exactly one checkpoint; the argument exists so
            the signature matches every other family's and so a second one can be added without
            a breaking change.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which published artifact to execute. Only ``"torch-fp32"`` is published;
            ``"auto"`` takes the best one published for the device.
        checkpoint_path: A checkpoint of your own instead of the published weights. Upstream's
            own ``BEN2_Base.pth`` works here -- it is a training checkpoint, and the loader takes
            ``model_state_dict`` out of it.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        variant: The variant in use.
        runtime: The artifact key actually in use.
        device: The device actually in use.
    """

    VARIANTS = ("base",)
    EXECUTES = ("torch",)
    FAMILY = "ben2"

    def __init__(
        self,
        variant: str = "base",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS and checkpoint_path is None:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()
        self.revision = revision

        if checkpoint_path is None:
            self.runtime = select_runtime(
                self.device, artifacts(self.FAMILY, variant, revision=revision), runtime)
            weights = resolve(self.FAMILY, variant, self.runtime, revision=revision)
        else:
            self.runtime = "torch-fp32"
            weights = Path(checkpoint_path)

        self._predictor = Predictor.from_pretrained(weights, device=self.device)
        print(f"BEN2 {variant} ready on {self.device} via {self.runtime}.")

    def predict(self, image: Union[str, np.ndarray], *, stretch: bool = True) -> np.ndarray:
        """Matte *image*.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.
            stretch: Reproduce upstream's per-image min-max normalisation. ``False`` returns the
                network's calibrated sigmoid, where 0.5 means something and two images are
                comparable.

        Returns:
            An ``HxW`` uint8 array at the input's resolution. 255 is foreground.

            With ``stretch=True`` this is a contrast-stretched probability: 255 is the most
            foreground pixel *in this image*, whatever the model's absolute confidence was. With
            ``stretch=False`` it is the sigmoid itself, scaled to 0-255.

            An image the model reads as uniform returns a flat matte rather than the ``nan``
            upstream's unguarded division produces.
        """
        return self._predictor.matte(load_image(image), stretch=stretch)

    def cutout(self, image: Union[str, np.ndarray], *, stretch: bool = True,
               refine: bool = False) -> np.ndarray:
        """Matte *image* and composite it, background transparent.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.
            stretch: As :meth:`predict`. Has no effect when *refine* is set, because upstream's
                refined path has nowhere to apply it.
            refine: Estimate unmixed foreground colours first, so a soft edge does not carry a
                fringe of the background it came from. Costs two full-resolution box blurs.

        Returns:
            An ``HxWx4`` uint8 RGBA array at the input's resolution.
        """
        return self._predictor.cutout(load_image(image), stretch=stretch, refine=refine)
