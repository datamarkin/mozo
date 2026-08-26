# SPDX-License-Identifier: Apache-2.0
"""Moebius, extracted for deployment: an image and a mask in, the masked thing gone.

A 226M-parameter latent diffusion inpainter that matches an 11.9B one. Unlike every other vendor
in mozo this package **rewrites** its input rather than describing it -- there are no boxes, no
masks and no scores on the way out, only pixels -- and unlike every other one it answers with a
*sample* rather than an estimate, so a seed is part of the call.

It runs at 512x512 and cannot run at anything else. That is a property of the published weights,
not a setting: see ``config.py``.

    >>> from mozo.vendors.moebius_deploy import Predictor      # doctest: +SKIP
    >>> model = Predictor("torch-fp32.pth", "general")         # doctest: +SKIP
    >>> clean = model.predict(frame, mask, seed=0)             # doctest: +SKIP

See ``PROVENANCE.md`` for what this derives from and what it deliberately leaves behind -- the
PixelHacker teacher, the MI-GAN pre-fill and two face-specific checkpoints -- and ``README.md``
for how to drive it.
"""

from .attention import CrossLambda, DepthwiseSeparableConv, MixFFN, SelfLambda, fold_positional
from .config import SPECS, Spec, VaeSpec, get_spec
from .image import binarise, composite
from .network import UNet, timestep_embedding
from .predictor import Predictor
from .scheduler import DDIM, timesteps_for
from .vae import AutoencoderKL, Gaussian

__all__ = [
    "DDIM",
    "AutoencoderKL",
    "CrossLambda",
    "DepthwiseSeparableConv",
    "Gaussian",
    "MixFFN",
    "Predictor",
    "SPECS",
    "SelfLambda",
    "Spec",
    "UNet",
    "VaeSpec",
    "binarise",
    "composite",
    "fold_positional",
    "get_spec",
    "timestep_embedding",
    "timesteps_for",
]
