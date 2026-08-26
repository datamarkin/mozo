# SPDX-License-Identifier: Apache-2.0
"""The seam: an image and a mask in, the masked thing gone.

Everything the other modules compute is joined here, and this is the only file in the package that
holds an opinion about *how* the model should be run rather than what it computes.

**The answer is a sample, not an estimate.** Two draws are involved -- the autoencoder's stochastic
encode and the initial noise -- and both come from a :class:`torch.Generator` that this class owns.
Upstream reaches for ``torch.manual_seed``, which is a process-wide write; a library may not do
that, so the generator is threaded explicitly instead. The draws happen in upstream's order and at
upstream's shapes, which is what makes the two reproduce each other bit for bit from the same seed:

1. encode the clean image, 2. encode the masked image, 3. the initial noise, 4. the offset.

**Twenty steps runs nineteen.** See :mod:`~.scheduler`. The default is kept at upstream's so the
two are comparable, and the number actually run is reported by :meth:`steps_for`.

**Guidance ships at 2.0**, which is the value upstream's own README passes. Its pipeline signature
says 4.5 and its argparse says 2.5; three defaults for one knob, and this picks the one the
documented invocation uses.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import get_spec
from .image import as_tensor, binarise, composite, dilate, to_pixels
from .network import UNet
from .scheduler import DDIM, timesteps_for
from .vae import AutoencoderKL

__all__ = ["Predictor"]

#: Upstream's README invocation. Not its pipeline default (4.5) and not its argparse default (2.5).
GUIDANCE = 2.0
#: Upstream's ``--noise-offset``. Perturbs the initial latent once, per channel.
NOISE_OFFSET = 0.0357
#: Upstream's ``strength``. Below one, so one timestep is trimmed -- see :mod:`~.scheduler`.
STRENGTH = 0.99


class Predictor:
    """Moebius, loaded and ready to remove things.

    Args:
        checkpoint: The UNet's ``.pth``.
        vae_checkpoint: The autoencoder's ``.pth``. A separate file because it is a separate work
            from a separate repository -- see ``PROVENANCE.md``.
        variant: ``"general"`` or ``"places2"``.
        device: Anything torch accepts.
    """

    def __init__(self, checkpoint, vae_checkpoint, variant: str = "general",
                 device: str = "cpu") -> None:
        self.spec = get_spec(variant)
        self.device = torch.device(device)

        self.unet = _built(UNet, self.spec, _read(checkpoint, unwrap="diff_model."), self.device)
        self.vae = _built(AutoencoderKL, self.spec.vae, _read(vae_checkpoint), self.device)

        self.scheduler = DDIM()
        # Ten conditional and ten unconditional rows, projected. Fixed for the life of the model:
        # there is no id to pass, so there is none to pass wrongly.
        self._conditioning = self.unet.conditioning(1).detach()

    def steps_for(self, steps: int) -> int:
        """How many denoising steps *steps* actually runs. Not *steps*.

        Examples:
            >>> Predictor.steps_for(None, 20)   # doctest: +SKIP
            19
        """
        return len(timesteps_for(self.scheduler.schedule(steps), steps, STRENGTH))

    @torch.no_grad()
    def sample(self, image: np.ndarray, mask: np.ndarray, *, seed: int = 0, steps: int = 20,
               guidance: float = GUIDANCE, noise_offset: float = NOISE_OFFSET) -> np.ndarray:
        """Run the model at its native size. *image* must be 512x512 RGB; *mask* 512x512 binary.

        Returns the decoder's full frame, uncomposited -- every pixel is the model's. Callers
        almost always want :meth:`predict`, which puts the untouched pixels back.
        """
        side = self.spec.image_size
        if image.shape[:2] != (side, side):
            raise ValueError(
                f"Moebius runs at {side}x{side} and nothing else -- its positional table is stored "
                f"per latent cell, so there is no other shape to resize to. Got {image.shape[:2]}. "
                "Use predict(), which resizes and composites for you.")

        pixels, binary, masked = as_tensor(image, mask)
        pixels, binary, masked = (t.to(self.device) for t in (pixels, binary, masked))

        generator = torch.Generator(device="cpu").manual_seed(seed)
        latent = self.vae.scale(self.vae.encode(pixels).sample(generator).to(self.device))
        masked_latent = self.vae.scale(self.vae.encode(masked).sample(generator).to(self.device))

        noise = torch.randn(latent.shape, generator=generator, dtype=latent.dtype).to(self.device)
        if noise_offset:
            offset = torch.randn((latent.shape[0], latent.shape[1], 1, 1), generator=generator,
                                 dtype=latent.dtype).to(self.device)
            noise = noise + noise_offset * offset

        schedule = timesteps_for(self.scheduler.schedule(steps), steps, STRENGTH)
        latent = self.scheduler.add_noise(latent, noise, schedule[:1])

        # Nearest, not bilinear: the mask channel is meant to stay in {0, 1}.
        latent_side = side // self.spec.vae.downsample
        mask_latent = F.interpolate(binary, size=(latent_side, latent_side))

        for timestep in schedule:
            noise_pred = self._predict_noise(latent, mask_latent, masked_latent,
                                             timestep, guidance)
            latent = self.scheduler.step(noise_pred, int(timestep), latent, steps)

        return to_pixels(self.vae.decode(latent))

    def _predict_noise(self, latent: torch.Tensor, mask: torch.Tensor, masked: torch.Tensor,
                       timestep: torch.Tensor, guidance: float) -> torch.Tensor:
        """One classifier-free-guided noise prediction.

        The batch is doubled **unconditional first**, which is the order
        :attr:`Spec.conditioning_ids` returns and the order ``chunk`` reads back. Reversed, the
        guidance points away from the conditioning and still produces a coherent picture.
        """
        doubled = torch.cat([torch.cat([latent, mask, masked], dim=1)] * 2)
        predicted = self.unet(doubled, timestep.to(self.device), self._conditioning)
        uncond, cond = predicted.chunk(2)
        return uncond + guidance * (cond - uncond)

    def predict(self, image: np.ndarray, mask: np.ndarray, *, seed: int = 0, steps: int = 20,
                guidance: float = GUIDANCE, dilate_pixels: int = 0,
                feather: int = 3) -> np.ndarray:
        """Remove whatever *mask* selects, at the caller's own resolution.

        *image* is ``(H, W, 3)`` uint8 RGB; *mask* is ``(H, W)``, anything :func:`~.image.binarise`
        accepts. Returns ``(H, W, 3)`` uint8 with the masked region replaced.

        **Every pixel the feathered mask does not reach comes back byte-identical.** Said
        precisely rather than loosely, because *feather* spreads the blend outward: at the default
        radius of 3 the seam extends about 8 px past the selection, and only at ``feather=0`` is
        the untouched region exactly "outside the mask". Beyond that band the caller's own bytes
        are returned, not a value that rounded back to them.

        An empty mask returns the input unchanged without running the model -- there is nothing to
        remove, and nineteen denoising steps to answer that would be nineteen too many.

        Several disjoint regions are removed in one pass over their union, which is what the
        9-channel conditioning expects. One pass per region would cost n times as much and would
        do it worse, since each pass would be blind to the others' holes.

        **Anything other than 512x512 is resized for the model and composited back at full
        resolution**, and the resize is a *squash* rather than a fit -- the model has one shape and
        it is square, so a 16:9 frame reaches it distorted and is unsquashed afterwards. Detail
        inside the hole is limited by 512 either way, which on a large photograph is visible.
        Cropping a 512 window around the mask would fix both and is the intended improvement; it is
        not this landing.

        Raises:
            ValueError: If the selection does not survive the downsample to 512. A mask a few
                pixels wide on a large photograph can vanish entirely, and the failure is silent
                and total: the model is asked to remove nothing, the composite then blends a
                blurred reconstruction over the region, and the caller is handed the object still
                there and a success. Refusing is the only honest answer, and the message says what
                to do about it.
        """
        binary = binarise(np.asarray(mask))
        if binary.shape != image.shape[:2]:
            raise ValueError(f"mask {binary.shape} does not cover image {image.shape[:2]}")
        binary = dilate(binary, dilate_pixels)
        if not binary.any():
            return image.copy()

        side = self.spec.image_size
        if image.shape[:2] == (side, side):
            generated = self.sample(image, binary, seed=seed, steps=steps, guidance=guidance)
        else:
            small = _resize(image, side)
            small_mask = binarise(_resize(binary * 255, side, mask=True))
            if not small_mask.any():
                raise ValueError(
                    f"the selection does not survive the resize to {side}x{side}: it covers "
                    f"{int(binary.sum())} pixels of a {image.shape[1]}x{image.shape[0]} frame and "
                    "nothing at the model's own size. Grow it with dilate=, or crop the region "
                    "you care about and pass that instead.")
            generated = _resize(self.sample(small, small_mask, seed=seed, steps=steps,
                                            guidance=guidance),
                                image.shape[1], height=image.shape[0])

        return composite(image, generated, binary, feather=feather)


def _built(cls, spec, state: dict, device) -> "torch.nn.Module":
    """Build *cls* and fill it, without initialising 310M parameters that are about to be replaced.

    Constructing on the meta device allocates shapes and no storage, so ``kaiming_uniform_`` never
    runs; ``assign=True`` then hands the loaded tensors straight in. Together that is about 0.9 s
    of startup that was being spent generating random numbers and immediately overwriting them.

    Safe because the load is strict: a parameter or buffer missing from the checkpoint would stay
    on the meta device and fail confusingly later, and ``strict=True`` refuses before that -- which
    is the same guarantee the eager path had, arrived at by the same check.
    """
    with torch.device("meta"):
        model = cls(spec)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval().to(device)


def _read(path, unwrap: str = "") -> dict:
    """A checkpoint as fp32 tensors. ``weights_only`` because these files are pickles.

    *unwrap* strips a prefix. The published denoiser is upstream's byte stream placed unchanged,
    and upstream wraps its UNet in a ``RemovalModel`` that owns the conditioning table -- so every
    key arrives under ``diff_model.`` except ``embedding_layer.weight``. mozo folds the table into
    the UNet, which is what makes the ids un-passable, so the prefix comes off here rather than the
    published bytes being repacked to suit it. Repacking would make the NOTICE's "no tensor is
    altered, renamed, cast or dropped" false for the sake of nine characters.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    return {(key[len(unwrap):] if unwrap and key.startswith(unwrap) else key): value.float()
            for key, value in state.items()}


def _resize(array: np.ndarray, width: int, height: int | None = None,
            mask: bool = False) -> np.ndarray:
    """LANCZOS for pixels, nearest for masks -- a smooth filter ramps a binary edge."""
    mode = "L" if array.ndim == 2 else "RGB"
    resample = Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
    resized = Image.fromarray(array.astype(np.uint8), mode).resize(
        (width, height if height is not None else width), resample=resample)
    return np.asarray(resized)
