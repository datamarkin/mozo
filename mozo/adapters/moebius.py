"""Moebius inpainting: an image and a mask in, the masked thing gone.

The architecture lives in :mod:`mozo.vendors.moebius_deploy`, extracted from ``hustvl/Moebius``
and reduced to inference. The weights come from :func:`mozo.weights.parts` -- **two** artifacts,
because the autoencoder is a separate work from a separate repository and mozo publishes it as
one, not as a silent half of the other.

**This is the one family whose answer is an image.** Every other adapter in mozo describes what is
in a frame; this rewrites one. There is no PixelFlow container on the way out and no confidence
number, because the model does not estimate anything -- it draws a sample. Change the seed and you
get a different, equally valid removal.

    >>> car = SAM3Predictor().predict(frame, "the red car")            # doctest: +SKIP
    >>> clean = MoebiusPredictor("general").predict(frame, car)        # doctest: +SKIP
    >>> clean.shape == frame.shape                                     # doctest: +SKIP
    True

Everything the feathered seam does not reach comes back byte-identical to what was passed
in -- see ``predict`` for what "feathered" costs you at the border.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

from ..image import load_image
from ..runtimes import get_default_device
from ..vendors.moebius_deploy import Predictor
from ..weights import parts


def _mask_from(source: Any, shape: tuple[int, int]) -> np.ndarray:
    """A binary ``(H, W)`` mask from whatever the caller passed.

    Three things are accepted, in order of how much they say:

    *An array* is the mask, thresholded.

    *Detections carrying segments* are unioned -- several regions are removed in one pass, which is
    what the model's conditioning expects and is cheaper and better than one pass each.

    *Detections carrying only boxes* become filled rectangles, and the result looks like it: the
    model removes the rectangle it was given. That is the caller's instruction being followed
    rather than a failure, but it is worth knowing before wondering why the sky has a corner in it.
    """
    if isinstance(source, np.ndarray):
        return source

    mask = np.zeros(shape, dtype=np.uint8)
    segments = getattr(source, "masks", None)
    if segments is not None and len(segments) and segments[0] is not None:
        for segment in segments:
            mask |= np.asarray(segment).astype(np.uint8)
        return mask

    boxes = getattr(source, "xyxy", None)
    if boxes is None:
        raise TypeError(
            "Pass a mask array, or detections carrying segments or boxes. Got "
            f"{type(source).__name__}, which offers neither.")
    for x1, y1, x2, y2 in np.asarray(boxes).astype(int):
        mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 1
    return mask


class MoebiusPredictor:
    """One loaded Moebius variant, ready to remove things.

    Args:
        variant: ``"general"`` for arbitrary photographs, ``"places2"`` for scenes and backgrounds.
        device: Where to run. Defaults to the best device this machine has.
        checkpoint_path: A UNet checkpoint of your own instead of the published weights.
            ``vae_path`` is then required too, since the pair is what makes a model.
        vae_path: An autoencoder checkpoint of your own.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        runtime: The artifact key actually in use. ``"torch-fp32"``; there is no published graph
            yet, and ``moebius-plan.md`` says why not publishing one is a result.
        device: The device actually in use.
    """

    VARIANTS = ("general", "places2")
    EXECUTES = ("torch",)

    def __init__(
        self,
        variant: str = "general",
        device: str | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        vae_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS and checkpoint_path is None:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")
        if (checkpoint_path is None) != (vae_path is None):
            raise ValueError(
                "checkpoint_path and vae_path go together: Moebius is a denoiser and an "
                "autoencoder, and half of a model is not a model.")

        self.variant = variant
        self.device = device or get_default_device()
        self.runtime = "torch-fp32"
        self.revision = revision

        if checkpoint_path is None:
            published = parts("moebius", variant, self.runtime, revision=revision)
            unet, vae = published["unet"], published["vae"]
        else:
            unet, vae = Path(checkpoint_path), Path(vae_path)

        self._predictor = Predictor(unet, vae, variant, device=self.device)
        print(f"Moebius {variant} ready on {self.device} via {self.runtime}.")

    def predict(
        self,
        image: Union[str, np.ndarray],
        mask: Any,
        *,
        seed: int = 0,
        steps: int = 20,
        guidance: float = 2.0,
        dilate: int = 0,
        feather: int = 3,
    ) -> np.ndarray:
        """Remove whatever *mask* selects.

        Args:
            image: Path or ``(H, W, 3)`` RGB array.
            mask: A binary array, or a ``Detections`` whose segments or boxes say what to remove.
            seed: Which sample to draw. The same seed gives the same picture; a different one
                gives a different and equally valid removal. There is no "best" seed to default to.
            steps: Denoising steps requested. **Nineteen of twenty are run** -- upstream trims one
                and mozo reproduces it rather than quietly disagreeing about what twenty means.
            guidance: Classifier-free guidance scale. Upstream's own README uses 2.0.
            dilate: Grow the mask by this many pixels first. A removal that stops at the object's
                edge tends to leave its shadow and its antialiased rim behind.
            feather: Radius of the blur on the mask before compositing.

        Returns:
            ``(H, W, 3)`` uint8 RGB, the same size as the input. Every pixel the feathered
            mask does not reach is byte-identical to it -- about 8 px beyond the selection at the
            default ``feather=3``, and exactly the mask's own edge at ``feather=0``.

        An empty mask returns the image unchanged without running the model.
        """
        frame = load_image(image)
        binary = _mask_from(mask, frame.shape[:2])
        return self._predictor.predict(
            frame, binary, seed=seed, steps=steps, guidance=guidance,
            dilate_pixels=dilate, feather=feather)

