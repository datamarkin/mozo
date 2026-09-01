# ------------------------------------------------------------------------
# BEN2 -- Background Erase Network
# Copyright (c) 2025 Prama LLC. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""Load a checkpoint, run an image, return a matte.

Upstream keeps this on the model as ``BEN_Base.inference``. It lives here for three reasons, all
of them things a library may not do to its host:

* ``inference`` calls ``set_random_seed(9)``, which writes ``random.seed``, ``np.random.seed``,
  ``torch.manual_seed``, ``torch.cuda.manual_seed_all``, ``torch.backends.cudnn.deterministic``
  and ``torch.backends.cudnn.benchmark`` -- six process-wide writes on every prediction. Nothing
  on the inference path is stochastic, so dropping them changes no number; the gate proves that
  rather than assuming it.
* It selects float16 or float32 by ``torch.cuda.is_available()``, making the model's arithmetic a
  property of the machine rather than of the call. Here the dtype is an argument.
* It mutates its argument. ``original_image.putalpha(mask)`` writes the alpha channel into the
  caller's own ``PIL.Image``. This module returns new arrays and touches nothing it was given.

``BEN2.py`` also runs two statements at import: ``set_random_seed(9)`` again, and
``torch.set_float32_matmul_precision('highest')``. Neither is carried. The second happens to be
torch's own default today, so it is a no-op that pins a default against a future change -- and
pinning global torch state is still not a library's to do.
"""

from __future__ import annotations

__all__ = ["Predictor"]

from pathlib import Path

import numpy as np
import torch

from .image import postprocess, preprocess, refine_foreground
from .network import BEN_Base


class Predictor:
    """One loaded BEN2 model, ready to matte.

    Attributes:
        model: The underlying :class:`~.network.BEN_Base`, in eval mode.
        device: Where the weights live.
        dtype: The compute dtype. Parity is claimed for ``float32`` on CPU and nothing else.
    """

    def __init__(self, model: BEN_Base, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self.model = model.eval().to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(cls, weights: str | Path, device: str | torch.device = "cpu",
                        dtype: torch.dtype = torch.float32) -> "Predictor":
        """Build the model and load *weights* strictly.

        Accepts either the repacked state dict mozo publishes or upstream's own
        ``BEN2_Base.pth``, which is a **training** checkpoint -- epoch, optimiser moments, loss
        scaler and metrics alongside the weights, three times the size of the model. If the file
        carries a ``model_state_dict`` key, that is what is loaded and the rest is ignored.

        The load is strict. Upstream's ``loadcheckpoints`` is also strict, which is one of the
        few places it and this package already agreed.
        """
        blob = torch.load(Path(weights), map_location="cpu", weights_only=True)
        state = blob["model_state_dict"] if "model_state_dict" in blob else blob
        # Released before the model is built. On upstream's own BEN2_Base.pth the rest of the blob
        # is 753 MB of optimiser moments, and holding it through construction and the load would
        # put all of it, the fresh 380 MB of parameters and the 380 MB being copied in on the
        # heap at once.
        blob = None
        model = BEN_Base()
        model.load_state_dict(state, strict=True)
        return cls(model, device=device, dtype=dtype)

    @torch.inference_mode()
    def matte(self, rgb: np.ndarray, *, stretch: bool = True) -> np.ndarray:
        """``(H, W, 3)`` uint8 RGB -> ``(H, W)`` uint8 alpha at the same resolution.

        Args:
            rgb: The image. Must already be RGB.
            stretch: Reproduce upstream's per-image min-max normalisation of the matte. See
                :func:`~.image.postprocess` for what that does to the meaning of the number.

        Returns:
            np.ndarray: ``(H, W)`` uint8. 255 is foreground.
        """
        height, width = rgb.shape[:2]
        tensor = preprocess(rgb, device=self.device, dtype=self.dtype)
        return postprocess(self.model(tensor), (height, width), stretch=stretch)

    @torch.inference_mode()
    def cutout(self, rgb: np.ndarray, *, stretch: bool = True, refine: bool = False) -> np.ndarray:
        """``(H, W, 3)`` uint8 RGB -> ``(H, W, 4)`` uint8 RGBA with the background transparent.

        Args:
            rgb: The image. Must already be RGB.
            stretch: As :meth:`matte`. Ignored when *refine* is set -- see below.
            refine: Estimate unmixed foreground colours before compositing, so a soft edge does
                not carry a fringe of the old background. Costs two box blurs at full resolution.

        Returns:
            np.ndarray: ``(H, W, 4)`` uint8.

        **The two paths do not produce the same alpha, and that is upstream's doing.** With
        ``refine=False`` the alpha is bilinearly resized and then contrast-stretched. With
        ``refine=True`` upstream takes the raw 1024x1024 sigmoid, casts it to uint8, and resizes
        *that* to the image with PIL's default filter -- no stretch anywhere. Both are reproduced
        exactly, which is why ``stretch`` has no effect on the refined path: there is nowhere in
        upstream's refined path for it to apply.
        """
        from PIL import Image  # local: only the refined path needs it

        height, width = rgb.shape[:2]
        tensor = preprocess(rgb, device=self.device, dtype=self.dtype)
        raw = self.model(tensor)

        if not refine:
            alpha = postprocess(raw, (height, width), stretch=stretch)
            return np.dstack([rgb, alpha])

        # Upstream: ToPILImage()(res.squeeze()) -- truncating to uint8 at 1024x1024 -- then a PIL
        # resize to the image. Not the bilinear interpolate the unrefined path uses.
        small = (raw.float().squeeze() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        alpha = np.asarray(Image.fromarray(small, mode="L").resize((width, height)))
        return np.dstack([refine_foreground(rgb, alpha), alpha])
