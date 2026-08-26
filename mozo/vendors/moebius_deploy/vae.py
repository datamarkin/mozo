# SPDX-License-Identifier: Apache-2.0
"""The autoencoder, extracted: pixels to latents and back.

Moebius never sees a pixel. It denoises a 64x64x4 latent, and this is the pair of networks that
puts an image into that space and takes one back out -- ``sdvae_f8d4`` in upstream's config, which
is Stable Diffusion XL's ``AutoencoderKL`` fine-tuned and republished by the PixelHacker authors.

Three things about it are worth knowing before reading the code, because each one is a silent
wrong answer rather than an exception:

**The encoder is stochastic.** ``encode`` returns a mean and a log-variance, and upstream samples
from that distribution rather than taking the mode. That is why :meth:`AutoencoderKL.encode`
returns the parameters and makes the caller draw: the draw needs a generator, and where the
randomness comes from is a decision the pipeline has to own rather than inherit. See
:class:`Gaussian`.

**Every ``GroupNorm`` here uses ``eps=1e-6``, not PyTorch's ``1e-5``.** ``diffusers`` passes
``resnet_eps=1e-6`` throughout the VAE. Nothing raises if you take the default.

**The latents are scaled.** ``encode`` multiplies by ``scaling_factor`` and ``decode`` divides.
That constant, ``0.13025``, belongs to the diffusion model that was trained against these latents,
not to the autoencoder -- skip it and the denoiser sees inputs roughly seven times too large.

The downsample is padded asymmetrically -- ``(0, 1, 0, 1)`` before a stride-2 convolution with no
padding of its own -- which is not what ``Conv2d(..., stride=2, padding=1)`` computes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["AutoencoderKL", "Gaussian"]

#: Every ``GroupNorm`` in the autoencoder. Upstream's ``resnet_eps``, and not PyTorch's default.
NORM_EPS = 1e-6


class Gaussian:
    """The distribution :meth:`AutoencoderKL.encode` describes.

    Not a tensor, because the choice between sampling it and taking its mode changes what the
    whole pipeline guarantees, and a caller that is handed a tensor has already had that choice
    made for it. Upstream samples. :meth:`mode` exists for anyone who wants a deterministic answer
    and is prepared to say that is what they took.

    Args:
        parameters: ``(B, 8, H, W)`` -- the mean and log-variance the encoder wrote, concatenated.
    """

    def __init__(self, parameters: torch.Tensor) -> None:
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        #: Clamped exactly as upstream clamps it. Left un-clamped, ``exp`` overflows to ``inf``
        #: on a checkpoint stored in half precision.
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw one latent.

        The generator is threaded rather than taken from the global RNG, so that running a
        workflow cannot change what an unrelated part of the process draws next.

        **The draw happens on the CPU and is moved**, whatever device the latent lives on. Two
        reasons, and the second is the one that matters. A CPU generator cannot seed a CUDA or MPS
        draw at all -- torch raises -- so this is required rather than chosen. And drawing on the
        CPU everywhere makes a seed mean the same picture on every device: seed 0 on a Mac and
        seed 0 on a CUDA box give the same sample, which is not true of a per-device generator.
        """
        noise = torch.randn(self.mean.shape, generator=generator, dtype=self.mean.dtype)
        return self.mean + self.std * noise.to(self.mean.device)

    def mode(self) -> torch.Tensor:
        """The distribution's peak. Deterministic, and **not** what upstream runs."""
        return self.mean


class Resnet(nn.Module):
    """Two convolutions and a skip, at one width.

    ``diffusers`` divides the sum by ``output_scale_factor``, which is ``1.0`` here. That division
    is kept because it is exact in IEEE 754 and dropping it would be a divergence chosen for
    tidiness rather than for a reason.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=NORM_EPS)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=NORM_EPS)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                              if in_channels != out_channels else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + hidden


class Attention(nn.Module):
    """Single-head self-attention over the bottleneck, one per mid block.

    One head, because ``diffusers`` builds this with ``attention_head_dim=channels``. The
    ``GroupNorm`` runs on the channel-first view and the projections on the sequence-first one,
    which is why the transposes look redundant and are not.
    """

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(groups, channels, eps=NORM_EPS)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        hidden = x.view(batch, channels, height * width).transpose(1, 2)
        hidden = self.group_norm(hidden.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden).unsqueeze(1)
        key = self.to_k(hidden).unsqueeze(1)
        value = self.to_v(hidden).unsqueeze(1)

        hidden = F.scaled_dot_product_attention(query, key, value)
        hidden = hidden.squeeze(1)
        hidden = self.to_out[0](hidden)
        return x + hidden.transpose(-1, -2).reshape(batch, channels, height, width)


class MidBlock(nn.Module):
    """Resnet, attention, resnet. Identical in the encoder and the decoder."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([Attention(channels, groups)])
        self.resnets = nn.ModuleList([Resnet(channels, channels, groups) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class Downsample(nn.Module):
    """Halve both sides.

    The pad is ``(0, 1, 0, 1)`` -- right and bottom only -- and the convolution then has no
    padding of its own. ``Conv2d(3, stride=2, padding=1)`` pads all four sides and is a different
    function; it produces the same shape, which is what makes the mistake survive.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class Upsample(nn.Module):
    """Double both sides: nearest-neighbour, then a 3x3 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class DownBlock(nn.Module):
    """One encoder level: *layers* resnets, then an optional halving."""

    def __init__(self, in_channels: int, out_channels: int, layers: int, groups: int,
                 downsample: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [Resnet(in_channels if i == 0 else out_channels, out_channels, groups)
             for i in range(layers)])
        self.downsamplers = nn.ModuleList([Downsample(out_channels)]) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
        return x


class UpBlock(nn.Module):
    """One decoder level: *layers* resnets, then an optional doubling."""

    def __init__(self, in_channels: int, out_channels: int, layers: int, groups: int,
                 upsample: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [Resnet(in_channels if i == 0 else out_channels, out_channels, groups)
             for i in range(layers)])
        self.upsamplers = nn.ModuleList([Upsample(out_channels)]) if upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x


class Encoder(nn.Module):
    """Pixels to distribution parameters."""

    def __init__(self, spec) -> None:
        super().__init__()
        widths = spec.block_out_channels
        groups = spec.norm_num_groups
        self.conv_in = nn.Conv2d(spec.in_channels, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([
            DownBlock(widths[max(i - 1, 0)], widths[i], spec.layers_per_block, groups,
                      downsample=i < len(widths) - 1)
            for i in range(len(widths))])
        self.mid_block = MidBlock(widths[-1], groups)
        self.conv_norm_out = nn.GroupNorm(groups, widths[-1], eps=NORM_EPS)
        # Twice the latent width: a mean and a log-variance, concatenated on the channel axis.
        self.conv_out = nn.Conv2d(widths[-1], 2 * spec.latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class Decoder(nn.Module):
    """Latents to pixels."""

    def __init__(self, spec) -> None:
        super().__init__()
        widths = tuple(reversed(spec.block_out_channels))
        groups = spec.norm_num_groups
        self.conv_in = nn.Conv2d(spec.latent_channels, widths[0], 3, padding=1)
        self.mid_block = MidBlock(widths[0], groups)
        # One resnet more per level than the encoder uses -- upstream's ``layers_per_block + 1``.
        self.up_blocks = nn.ModuleList([
            UpBlock(widths[max(i - 1, 0)], widths[i], spec.layers_per_block + 1, groups,
                    upsample=i < len(widths) - 1)
            for i in range(len(widths))])
        self.conv_norm_out = nn.GroupNorm(groups, widths[-1], eps=NORM_EPS)
        self.conv_out = nn.Conv2d(widths[-1], spec.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(F.silu(self.conv_norm_out(x)))


class AutoencoderKL(nn.Module):
    """The pair, plus the two 1x1 convolutions that sit either side of the latent.

    Args:
        spec: A :class:`~.config.VaeSpec`.

    Examples:
        >>> from mozo.vendors.moebius_deploy.config import VaeSpec
        >>> vae = AutoencoderKL(VaeSpec()).eval()
        >>> sum(p.numel() for p in vae.parameters()) / 1e6
        83.65...
    """

    def __init__(self, spec) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = Encoder(spec)
        self.decoder = Decoder(spec)
        self.quant_conv = nn.Conv2d(2 * spec.latent_channels, 2 * spec.latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(spec.latent_channels, spec.latent_channels, 1)

    def encode(self, x: torch.Tensor) -> Gaussian:
        """Describe *x* in latent space.

        Returns the distribution, not a latent: see :class:`Gaussian` for why the draw is the
        caller's. The ``scaling_factor`` is **not** applied here -- it belongs to the sample, and
        applying it to a mean and a log-variance would scale them differently.
        """
        return Gaussian(self.quant_conv(self.encoder(x)))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Turn a scaled latent back into pixels in ``[-1, 1]``.

        Undoes ``scaling_factor`` on the way in, so callers hand back exactly what they were given
        by :meth:`scale`.
        """
        return self.decoder(self.post_quant_conv(latent / self.spec.scaling_factor))

    def scale(self, latent: torch.Tensor) -> torch.Tensor:
        """Put a drawn latent into the units the diffusion model was trained on."""
        return latent * self.spec.scaling_factor
