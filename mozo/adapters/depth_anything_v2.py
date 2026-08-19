"""Depth Anything V2 monocular depth estimation, on whichever runtime the host can execute.

The architecture lives in :mod:`mozo.vendors.depth_anything_v2_deploy`, extracted from the
authors' repository and reduced to inference. The weights come from :func:`mozo.weights.resolve`.
Pre- and post-processing come from the vendor, so the pipeline is upstream's step for step --
verified bit-identical to it on all nine variants.

Nine variants in two groups, and the difference between them is not cosmetic:

* ``small``, ``base``, ``large`` predict **relative** depth. The output is inverse depth on an
  arbitrary per-image scale: larger means nearer, and that is all it means. Two images cannot be
  compared to each other, and no value is a distance.
* ``indoor-*`` and ``outdoor-*`` predict **metric** depth in metres, over 0-20 m and 0-80 m
  respectively.

:attr:`unit` says which, and it is ``None`` for the relative variants rather than a plausible
guess -- the same rule mozo applies to class names. Code that needs metres must ask a variant
that has them.

    >>> model = DepthAnythingV2Predictor("small")              # doctest: +SKIP
    >>> depth = model.predict(image)                           # doctest: +SKIP
    >>> model.unit                                             # doctest: +SKIP
    None
    >>> DepthAnythingV2Predictor("indoor-base").unit           # doctest: +SKIP
    'metres'
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

from ..device import get_default_device
from ..runtimes import select_runtime
from ..utils import load_image
from ..vendors.depth_anything_v2_deploy import MODEL_SPECS, Predictor
from ..weights import artifacts, resolve


class DepthAnythingV2Predictor:
    """One loaded Depth Anything V2 variant, ready to run.

    Args:
        variant: A published variant -- ``small``, ``base``, ``large``, or those three sizes
            prefixed ``indoor-`` or ``outdoor-`` for metric depth.
        device: Where to run. Defaults to the best device this machine has.
        runtime: Which published artifact to execute. Only ``"torch-fp32"`` is published today;
            ``"auto"`` takes the best one published for the device.
        checkpoint_path: A checkpoint of your own, instead of the published weights. The variant
            then names the architecture to build, and with it the ``max_depth`` scaling, so a
            metric fine-tune must be loaded under a metric variant name.
        revision: Pin a published revision instead of taking the latest.

    Attributes:
        variant: The variant in use.
        runtime: The artifact key actually in use.
        device: The device actually in use.
        unit: ``"metres"`` for the metric variants, ``None`` for the relative ones.

    Examples:
        >>> DepthAnythingV2Predictor("outdoor-small").unit  # doctest: +SKIP
        'metres'
    """

    VARIANTS = tuple(MODEL_SPECS)
    FAMILY = "depth_anything_v2"

    def __init__(
        self,
        variant: str = "small",
        device: str | None = None,
        *,
        runtime: str = "auto",
        checkpoint_path: str | Path | None = None,
        revision: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant {variant!r}. Choose from: {list(self.VARIANTS)}")

        self.variant = variant
        self.device = device or get_default_device()

        if checkpoint_path is None:
            self.runtime = select_runtime(
                self.device, artifacts(self.FAMILY, variant, revision=revision), runtime)
            weights = resolve(self.FAMILY, variant, self.runtime, revision=revision)
        else:
            # A checkpoint mozo did not publish: the architecture is known, the graph is not.
            self.runtime = "torch-fp32"
            weights = Path(checkpoint_path)

        self._predictor = Predictor.from_pretrained(variant, weights=weights, device=self.device)
        print(f"Depth Anything V2 {variant} ready on {self.device} via {self.runtime}.")

    @property
    def unit(self) -> str | None:
        """What one unit of :meth:`predict`'s output means, or ``None`` if it is unitless."""
        return self._predictor.spec.unit

    @property
    def max_depth(self) -> float | None:
        """The metric ceiling in metres, or ``None`` for a relative-depth variant."""
        return self._predictor.spec.max_depth

    def predict(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Estimate depth for *image*.

        Args:
            image: A file path, encoded bytes, or an ``HWC`` RGB array.

        Returns:
            An ``HxW`` float32 array at the input's resolution. Metres when :attr:`unit` says so;
            otherwise inverse depth on an arbitrary scale, where larger means nearer and no value
            is a distance.
        """
        # mozo's contract is RGB; this vendor is upstream's code and upstream reads with
        # ``cv2.imread``, so it wants BGR and converts back internally. Cheaper than editing a
        # vendor that is verified to zero delta against upstream, and small either way against
        # 70-400 ms of inference.
        return self._predictor.predict(cv2.cvtColor(load_image(image), cv2.COLOR_RGB2BGR))
