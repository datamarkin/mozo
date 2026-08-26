# SPDX-License-Identifier: Apache-2.0
"""DDIM, at the one configuration Moebius was trained under.

A hundred lines of arithmetic on a ``(B, 4, 64, 64)`` tensor. It is written out rather than
imported because importing ``diffusers`` to obtain one scheduler would drag in the dependency tree,
the import-time side effects and the version churn that :mod:`mozo.vendors` exists to avoid -- and
because this is the piece the export path has to keep in Python, so mozo has to own it.

The schedule is Stable Diffusion's: ``beta_start=0.00085``, ``beta_end=0.012``, ``scaled_linear``
over 1000 training steps. "Scaled linear" interpolates the **square roots** of the endpoints and
squares the result, which is not the same curve as interpolating the endpoints directly, and
nothing raises if you take the obvious reading.

**Two things here are silently off-by-one and both are upstream's behaviour.**

*A run of twenty steps runs nineteen.* Upstream's pipeline passes ``strength=0.99``, and
:func:`timesteps_for` reproduces the consequence: ``int(20 * 0.99) == 19``, so one step is trimmed
off the front. Asking for twenty and running twenty produces a different image and no error.

*The last step reads a different table.* ``prev_timestep`` goes negative at the end, and
``alpha_prod_t_prev`` falls back to :attr:`final_alpha_cumprod` rather than indexing
``alphas_cumprod``. It is a distinct branch, and it is the one a gate has to cover separately --
see ``tools/verify/moebius.py``.
"""

from __future__ import annotations

import torch

__all__ = ["DDIM", "timesteps_for"]


def timesteps_for(schedule: torch.Tensor, steps: int, strength: float) -> torch.Tensor:
    """The timesteps a run of *steps* at *strength* actually visits.

    Reproduces upstream's ``get_timesteps``: it keeps the last ``int(steps * strength)`` entries,
    so the default ``strength=0.99`` trims exactly one from the front of a twenty-step schedule.

    Examples:
        >>> import torch
        >>> full = torch.arange(950, -1, -50)
        >>> len(full), len(timesteps_for(full, 20, 0.99))
        (20, 19)
        >>> timesteps_for(full, 20, 0.99)[0].item()
        900
        >>> len(timesteps_for(full, 20, 1.0))
        20
    """
    keep = min(int(steps * strength), steps)
    return schedule[steps - keep:]


class DDIM:
    """The deterministic sampler, at ``eta = 0``.

    Args:
        num_train_timesteps: Length of the training schedule. 1000.
        beta_start: First beta, before scaling.
        beta_end: Last beta, before scaling.

    Examples:
        >>> ddim = DDIM()
        >>> ddim.alphas_cumprod.shape
        torch.Size([1000])
        >>> float(ddim.final_alpha_cumprod)
        1.0
        >>> ddim.schedule(20).tolist()[:3]
        [950, 900, 850]
    """

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.00085,
                 beta_end: float = 0.012) -> None:
        self.num_train_timesteps = num_train_timesteps
        # "scaled_linear": interpolate the square roots, then square. Not linear in beta.
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps,
                               dtype=torch.float32) ** 2
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        #: What the final step uses in place of an out-of-range lookup. One, because upstream
        #: leaves ``set_alpha_to_one`` at its default -- *not* ``alphas_cumprod[0]``.
        self.final_alpha_cumprod = torch.tensor(1.0)

    def schedule(self, steps: int) -> torch.Tensor:
        """The full descending timestep schedule for *steps* inference steps."""
        ratio = self.num_train_timesteps // steps
        return (torch.arange(steps) * ratio).flip(0).to(torch.int64)

    def add_noise(self, latent: torch.Tensor, noise: torch.Tensor,
                  timestep: torch.Tensor) -> torch.Tensor:
        """Take *latent* to the noise level of *timestep*."""
        alpha = self.alphas_cumprod[timestep].to(latent.device, latent.dtype)
        while alpha.dim() < latent.dim():
            alpha = alpha.unsqueeze(-1)
        return alpha.sqrt() * latent + (1.0 - alpha).sqrt() * noise

    def step(self, noise_pred: torch.Tensor, timestep: int, sample: torch.Tensor,
             steps: int) -> torch.Tensor:
        """One denoising step: ``x_t`` and a predicted noise in, ``x_{t-1}`` out.

        *steps* is the inference-step count the schedule was built with, because the stride
        between adjacent timesteps is what says where ``t - 1`` actually is.
        """
        previous = int(timestep) - self.num_train_timesteps // steps
        alpha = self.alphas_cumprod[int(timestep)].to(sample.device, sample.dtype)
        # The final-step branch. ``previous`` is negative once, at the end of the run.
        alpha_prev = (self.alphas_cumprod[previous] if previous >= 0
                      else self.final_alpha_cumprod).to(sample.device, sample.dtype)

        original = (sample - (1.0 - alpha).sqrt() * noise_pred) / alpha.sqrt()
        direction = (1.0 - alpha_prev).sqrt() * noise_pred
        return alpha_prev.sqrt() * original + direction
