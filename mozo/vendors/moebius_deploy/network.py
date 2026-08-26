# SPDX-License-Identifier: Apache-2.0
"""The denoiser: nine latent channels in, four channels of predicted noise out.

A UNet, but not the one Stable Diffusion ships. Three things were taken out or swapped, and each is
visible in the published tensors rather than inferred from the paper:

**There is no mid block.** Upstream's class name -- ``..._prune_down_mid_up_block_8x8`` -- is
describing what was removed, and the checkpoint contains no ``mid_block.*`` key at all. Three levels
down, three levels up, and the bottleneck is simply where they meet.

**Every convolution that would be 3x3 is depthwise separable.** ``conv_in``, ``conv_out`` and both
convolutions in every residual block are timm's ``DepthwiseSeparableConv``, which is most of how a
226M-parameter model covers the ground an 11.9B one does.

**Both attentions are λ layers** -- see :mod:`~.attention`. Upstream builds ordinary
``diffusers`` attention and then *overwrites* ``attn1`` and ``attn2`` after construction, so the
quadratic modules are created and immediately discarded. Nothing here builds them.

One inherited quirk is reproduced rather than fixed: ``in_channels`` is nine, ordered **noisy
latent (4), mask (1), masked-image latent (4)**. The order is not recorded anywhere in upstream's
config -- it lives in the line that concatenates them -- and getting it wrong conditions the model
on its own noise while raising nothing.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .attention import CrossLambda, DepthwiseSeparableConv, MixFFN, SelfLambda

__all__ = ["UNet", "timestep_embedding"]


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep features, at ``diffusers``' ``flip_sin_to_cos=True, freq_shift=0``.

    The flip is not decoration: it puts cosine in the first half and sine in the second, which is
    the opposite of the ordering most references use, and a model trained one way reads the other
    as a different timestep entirely.
    """
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32,
                                                    device=timesteps.device)
    emb = timesteps[:, None].float() * torch.exp(exponent / half)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TimestepEmbedding(nn.Module):
    """Two linear layers with a SiLU between them."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, out_dim)
        self.linear_2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class ResnetBlock(nn.Module):
    """A residual block whose two convolutions are depthwise separable.

    The timestep is added between them, after its own SiLU -- ``diffusers``' ``"default"``
    time-embedding norm, which is a shift and not a scale-and-shift.
    """

    def __init__(self, in_channels: int, out_channels: int, temb_channels: int,
                 groups: int = 32, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.conv_shortcut = (nn.Conv2d(in_channels, out_channels, 1)
                              if in_channels != out_channels else None)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_emb_proj(F.silu(temb))[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x.contiguous())
        return x + hidden


class TransformerBlock(nn.Module):
    """Self-λ, cross-λ, MixFFN -- each pre-normed and residual.

    The shape of ``diffusers``' ``BasicTransformerBlock`` with both attentions replaced. The
    ``LayerNorm`` epsilon is torch's ``1e-5`` here, unlike the ``GroupNorm``s around it.
    """

    def __init__(self, channels: int, dim_k: int, heads: int, cross_dim: int, positions: int,
                 sequence: int, kernel: int, ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn1 = SelfLambda(channels, dim_k, heads, kernel=kernel)
        self.norm2 = nn.LayerNorm(channels)
        self.attn2 = CrossLambda(channels, dim_k, heads, cross_dim, positions, sequence)
        self.norm3 = nn.LayerNorm(channels)
        self.ff = MixFFN(channels, ratio)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor,
                height: int, width: int) -> torch.Tensor:
        x = self.attn1(self.norm1(x), height, width) + x
        x = self.attn2(self.norm2(x), conditioning, height, width) + x
        return self.ff(self.norm3(x), height, width) + x


class Transformer2D(nn.Module):
    """A stack of :class:`TransformerBlock`, wrapped in the projections that enter and leave 2-D.

    ``proj_in`` and ``proj_out`` are 1x1 convolutions rather than linears -- upstream leaves
    ``use_linear_projection`` at ``False`` -- and the whole thing is residual around its input.
    """

    def __init__(self, channels: int, dim_k: int, heads: int, cross_dim: int, positions: int,
                 sequence: int, kernel: int, ratio: float, layers: int = 1,
                 groups: int = 32) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(channels, dim_k, heads, cross_dim, positions, sequence, kernel, ratio)
            for _ in range(layers)])
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        hidden = self.proj_in(self.norm(x))
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        for block in self.transformer_blocks:
            hidden = block(hidden, conditioning, height, width)
        hidden = hidden.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        return self.proj_out(hidden) + x


class Downsample(nn.Module):
    """Halve both sides. Padded 1 on every side, unlike the autoencoder's."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Double both sides: nearest-neighbour, then a 3x3 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class DownBlock(nn.Module):
    """One level on the way down: resnet, attention, resnet, attention, then maybe a halving.

    Every intermediate is kept -- including the downsampled one -- because the matching up level
    consumes them in reverse.
    """

    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, layers: int,
                 dim_k: int, heads: int, cross_dim: int, positions: int, sequence: int,
                 kernel: int, ratio: float, groups: int, eps: float, downsample: bool) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels, temb_channels,
                        groups, eps) for i in range(layers)])
        self.attentions = nn.ModuleList([
            Transformer2D(out_channels, dim_k, heads, cross_dim, positions, sequence, kernel,
                          ratio, groups=groups) for _ in range(layers)])
        self.downsamplers = nn.ModuleList([Downsample(out_channels)]) if downsample else None

    def forward(self, x: torch.Tensor, temb: torch.Tensor,
                conditioning: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        skips: tuple[torch.Tensor, ...] = ()
        for resnet, attention in zip(self.resnets, self.attentions):
            x = attention(resnet(x, temb), conditioning)
            skips = skips + (x,)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            skips = skips + (x,)
        return x, skips


class UpBlock(nn.Module):
    """One level on the way up. Consumes one skip per layer, most recent first."""

    def __init__(self, in_channels: int, out_channels: int, prev_channels: int,
                 temb_channels: int, layers: int, dim_k: int, heads: int, cross_dim: int,
                 positions: int, sequence: int, kernel: int, ratio: float, groups: int,
                 eps: float, upsample: bool) -> None:
        super().__init__()
        resnets = []
        for i in range(layers):
            skip_channels = in_channels if i == layers - 1 else out_channels
            resnet_in = prev_channels if i == 0 else out_channels
            resnets.append(ResnetBlock(resnet_in + skip_channels, out_channels, temb_channels,
                                       groups, eps))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList([
            Transformer2D(out_channels, dim_k, heads, cross_dim, positions, sequence, kernel,
                          ratio, groups=groups) for _ in range(layers)])
        self.upsamplers = nn.ModuleList([Upsample(out_channels)]) if upsample else None

    def forward(self, x: torch.Tensor, skips: tuple[torch.Tensor, ...], temb: torch.Tensor,
                conditioning: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        for resnet, attention in zip(self.resnets, self.attentions):
            x = torch.cat([x, skips[-1]], dim=1)
            skips = skips[:-1]
            x = attention(resnet(x, temb), conditioning)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x, skips


class UNet(nn.Module):
    """The denoiser, plus the conditioning table and projection it is always called with.

    Upstream keeps ``nn.Embedding(20, 3072)`` outside the UNet, in a wrapper that looks the ids up
    and passes the result in. It is folded in here because the ids are **fixed at construction** --
    ten conditional and ten unconditional, the same for every image and every step -- so there is
    no id for a caller to pass, and therefore none to pass wrongly. :meth:`conditioning` builds the
    tensor once; it never varies.

    Args:
        spec: A :class:`~.config.Spec`.

    Examples:
        >>> from mozo.vendors.moebius_deploy.config import get_spec
        >>> net = UNet(get_spec("general")).eval()
        >>> sum(p.numel() for p in net.parameters()) / 1e6
        226.19...
    """

    def __init__(self, spec) -> None:
        super().__init__()
        self.spec = spec
        widths = spec.block_out_channels
        temb = spec.time_embed_dim
        sequence = spec.num_embeddings // 2
        down_sides, up_sides = spec.latent_sides()
        shared = dict(heads=spec.heads, cross_dim=spec.cross_attention_dim, sequence=sequence,
                      kernel=spec.local_kernel, ratio=spec.mix_mlp_ratio,
                      groups=spec.norm_num_groups, eps=spec.norm_eps)

        self.embedding_layer = nn.Embedding(spec.num_embeddings, spec.encoder_hid_dim)
        self.encoder_hid_proj = nn.Linear(spec.encoder_hid_dim, spec.cross_attention_dim)
        self.conv_in = DepthwiseSeparableConv(spec.in_channels, widths[0])
        self.time_embedding = TimestepEmbedding(widths[0], temb)

        self.down_blocks = nn.ModuleList([
            DownBlock(widths[max(i - 1, 0)], widths[i], temb, spec.layers_per_block,
                      dim_k=spec.head_dim(widths[i]), positions=down_sides[i] ** 2,
                      downsample=i < len(widths) - 1, **shared)
            for i in range(len(widths))])

        reversed_widths = tuple(reversed(widths))
        self.up_blocks = nn.ModuleList([
            UpBlock(in_channels=reversed_widths[min(i + 1, len(widths) - 1)],
                    out_channels=reversed_widths[i],
                    prev_channels=reversed_widths[max(i - 1, 0)], temb_channels=temb,
                    layers=spec.layers_per_block + 1, dim_k=spec.head_dim(reversed_widths[i]),
                    positions=up_sides[i] ** 2, upsample=i < len(widths) - 1, **shared)
            for i in range(len(widths))])

        self.conv_norm_out = nn.GroupNorm(spec.norm_num_groups, widths[0], eps=spec.norm_eps)
        self.conv_out = DepthwiseSeparableConv(widths[0], spec.out_channels)

    def conditioning(self, batch: int) -> torch.Tensor:
        """The Latent Categories Guidance tensor, unconditional rows first.

        ``(2 * batch, 10, cross_attention_dim)`` -- built to be concatenated against a
        classifier-free-guidance batch that is itself unconditional-first.
        """
        uncond, cond = self.spec.conditioning_ids
        ids = torch.tensor([list(uncond)] * batch + [list(cond)] * batch,
                           dtype=torch.long, device=self.embedding_layer.weight.device)
        return self.encoder_hid_proj(self.embedding_layer(ids))

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor,
                conditioning: torch.Tensor) -> torch.Tensor:
        """*sample* is ``(B, 9, 64, 64)``; the answer is ``(B, 4, 64, 64)``."""
        temb = self.time_embedding(
            timestep_embedding(timestep.expand(sample.shape[0]), self.spec.block_out_channels[0]))

        hidden = self.conv_in(sample)
        skips: tuple[torch.Tensor, ...] = (hidden,)
        for block in self.down_blocks:
            hidden, produced = block(hidden, temb, conditioning)
            skips = skips + produced

        for block in self.up_blocks:
            hidden, skips = block(hidden, skips, temb, conditioning)

        return self.conv_out(F.silu(self.conv_norm_out(hidden)))
