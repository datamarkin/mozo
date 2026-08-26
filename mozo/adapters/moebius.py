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

    An array is taken as the mask itself. Otherwise *source* is a PixelFlow ``Detections``, which
    is a **collection**: iterate it and each ``Detection`` carries what it found, in descending
    order of how exactly it says where the thing is.

    *``masks``* -- a list of ``(H, W)`` rasters, what a segmenter produces. Unioned across every
    detection, because several regions are removed in one pass over their union: that is what the
    9-channel conditioning expects, and one pass each would cost n times as much and do it worse,
    since each pass would be blind to the others' holes.

    *``segments``* -- a list of ``(N, 2)`` polygons, what OCR produces as the four corners it read.
    Filled.

    *``bbox``* -- ``[x1, y1, x2, y2]``. The fallback, and the result looks like it: the model
    removes the rectangle it was given. That is the caller's instruction being followed rather than
    a failure, but it is worth knowing before wondering why the sky has a corner in it.

    A detection may carry all three; the most exact one available wins, per detection rather than
    for the whole set, so a mixed result loses nothing.
    """
    if isinstance(source, np.ndarray):
        return source

    try:
        found = list(source)
    except TypeError:
        raise TypeError(
            f"Pass a mask array, or a Detections. Got {type(source).__name__}, which is neither "
            "an array nor iterable.") from None

    mask = np.zeros(shape, dtype=np.uint8)
    for detection in found:
        rasters = getattr(detection, "masks", None)
        polygons = getattr(detection, "segments", None)
        box = getattr(detection, "bbox", None)

        if rasters is not None and len(rasters):
            for raster in rasters:
                mask |= _fit(np.asarray(raster), shape)
        elif polygons is not None and len(polygons):
            import cv2

            # ``segments`` on a detection is *the* polygon -- an ``(N, 2)`` array of points, which
            # is EasyOCR's four read corners -- not a list of polygons. Iterating it yields points,
            # and filling those fills four single pixels. A 3-D array is several polygons for the
            # one detection, so both are accepted and only the rank tells them apart.
            outlines = np.asarray(polygons, dtype=np.float32)
            if outlines.ndim == 2:
                outlines = outlines[None]
            cv2.fillPoly(mask, [o.round().astype(np.int32) for o in outlines], 1)
        elif box is not None and len(box) == 4:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 1
        else:
            raise TypeError(
                f"a detection carries no masks, no segments and no bbox, so there is nothing to "
                f"remove: {detection!r}")
    return mask


def _fit(raster: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """One raster mask as ``(H, W)`` uint8 at *shape*.

    Resized nearest when it arrives at the model's own resolution rather than the frame's, which
    some segmenters do. Nearest because a smooth filter turns a hard edge into a ramp, and this is
    about to be treated as ``{0, 1}``.
    """
    binary = (raster.astype(bool)).astype(np.uint8)
    if binary.shape == shape:
        return binary

    import cv2

    return cv2.resize(binary, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


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

