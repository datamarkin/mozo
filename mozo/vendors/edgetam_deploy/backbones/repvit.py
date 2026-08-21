# SPDX-License-Identifier: Apache-2.0
"""EdgeTAM's image trunk: RepViT-M1, a mobile CNN built out of reparameterisable blocks.

Derived from ``timm/models/repvit.py`` (Apache-2.0, Hugging Face), reduced to the one variant
EdgeTAM trains and to inference. Upstream reaches it through ``timm.create_model``; both of
EdgeTAM's own entry points do the same -- the original repo calls ``create_model(pretrained=True)``,
which downloads ImageNet weights at construction, and ``transformers`` falls back to
``AutoConfig.from_pretrained("timm/repvit_m1.dist_in1k")``. Neither is available to a vendor here:
a vendor imports the standard library, torch and itself, and it does not reach the network to be
built. So the modules are carried rather than depended on, the way every other package under
``mozo/vendors`` carries its architecture.

**The module names are not a choice.** EdgeTAM's checkpoint stores this trunk under timm's own
names, so reproducing them is what makes the load strict with nothing renamed::

    image_encoder.trunk.body.stem.conv1.c.weight
    image_encoder.trunk.body.stages_0.blocks.0.token_mixer.conv.bn.weight
    image_encoder.trunk.body.stages_1.downsample.pre_block.channel_mixer.conv1.c.weight

Two of those levels come from wrappers rather than from RepViT itself, and both are reproduced
here instead of being stripped at load. ``body`` is EdgeTAM's ``TimmBackbone``, which holds the
network it wraps under that attribute. ``stages_0`` rather than ``stages.0`` is timm's
``FeatureListNet``, which flattens a feature-extracting model into a ``ModuleDict`` and joins the
path with underscores. :class:`RepViT` is that flattened form directly, which is why its stages
live in a ``ModuleDict`` and not an ``nn.Sequential``.

The geometry is timm's ``repvit_m1`` with ``legacy=True``, and the checkpoint agrees on every
count: widths ``(48, 96, 192, 384)``, depths ``(2, 2, 14, 2)``, squeeze-excite on alternating
blocks starting with the first, and the legacy branch's ``conv1`` carrying a batch norm rather
than being a bare convolution.

Left behind: the classifier head and its distillation twin, the model registry and the other
seven variants, ``forward_intermediates`` and its index arithmetic, gradient checkpointing, the
``device``/``dtype`` construction kwargs, and every ``fuse()``. Fusing folds each block's
branches into a single convolution for inference; it changes the weights the module holds, and
this package's whole claim is that it runs the published ones unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["RepViT"]

#: ``repvit_m1``: stage widths, blocks per stage, and the hidden-width multiplier of every
#: channel-mixing MLP. Written out rather than passed in -- EdgeTAM publishes one checkpoint, so
#: a second set of numbers here would be a variant nobody can load.
WIDTHS = (48, 96, 192, 384)
DEPTHS = (2, 2, 14, 2)
MLP_RATIO = 2
KERNEL_SIZE = 3


def _divisible(value: float, divisor: int = 8) -> int:
    """Round a channel count to a multiple of *divisor*, timm's ``make_divisible``.

    Called with ``round_limit=0`` upstream, which disables the guard against rounding a width
    down by more than a tenth -- so that branch is not carried here. On this trunk's four widths
    it returns 16, 24, 48 and 96, which is what the checkpoint's squeeze-excite layers hold.
    """
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class ConvNorm(nn.Sequential):
    """A convolution and its batch norm, named ``c`` and ``bn`` as the checkpoint spells them.

    Args:
        in_dim: Input channels.
        out_dim: Output channels.
        kernel: Square kernel side.
        stride: Convolution stride.
        pad: Symmetric padding.
        groups: Convolution groups; equal to *in_dim* makes it depthwise.
        norm_init: What the batch norm's scale starts at. Zero on the second projection of a
            residual MLP, so the branch begins as a no-op.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel: int = 1,
        stride: int = 1,
        pad: int = 0,
        groups: int = 1,
        norm_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.add_module(
            "c", nn.Conv2d(in_dim, out_dim, kernel, stride, pad, groups=groups, bias=False)
        )
        self.add_module("bn", nn.BatchNorm2d(out_dim))
        # Only ever observed through a loaded checkpoint, so these decide nothing about the
        # published numbers. Kept because a module that initialises differently from the code it
        # was copied from is no longer diffable against it.
        nn.init.constant_(self.bn.weight, norm_init)
        nn.init.constant_(self.bn.bias, 0)


class SqueezeExcite(nn.Module):
    """Per-channel gating from the channel means, timm's ``SEModule`` at RepViT's settings.

    Upstream's version is configurable across the activation, the gate, an optional norm and an
    optional max-pool branch. RepViT names none of those, so what is left is the original
    squeeze-and-excite: average over space, two 1x1 projections through a ReLU, a sigmoid, and a
    multiply.

    Args:
        channels: Channels in and out.
        ratio: Bottleneck width as a fraction of *channels*, before rounding.
    """

    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden = _divisible(channels * ratio)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        gate = x.mean((2, 3), keepdim=True)
        gate = self.fc2(F.relu(self.fc1(gate)))
        return x * torch.sigmoid(gate)


class RepVggDw(nn.Module):
    """The token mixer: a depthwise 3x3, a depthwise 1x1 and the identity, summed.

    This is the "reparameterisable" half of RepViT. At training time the three branches are
    separate; upstream can fold them into one convolution for inference. mozo runs them
    unfolded, because folding rewrites the published weights -- see the module docstring.

    Args:
        dim: Channels, which is also the group count.
        kernel: Side of the larger branch's kernel.
    """

    def __init__(self, dim: int, kernel: int = KERNEL_SIZE) -> None:
        super().__init__()
        self.conv = ConvNorm(dim, dim, kernel, 1, (kernel - 1) // 2, groups=dim)
        # timm's ``legacy=True`` branch, which is the one EdgeTAM's weights were trained under:
        # the 1x1 carries its own batch norm and the block-level one is dropped. The other
        # branch has a bare ``nn.Conv2d`` here and a ``BatchNorm2d`` around the sum, and would
        # leave ``conv1.bn.*`` with nowhere to load.
        self.conv1 = ConvNorm(dim, dim, 1, 1, 0, groups=dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x) + self.conv1(x) + x


class RepVitMlp(nn.Module):
    """The channel mixer: expand, activate, project back.

    Args:
        dim: Channels in and out.
        hidden: Width between the two projections.
    """

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.conv1 = ConvNorm(dim, hidden, 1, 1, 0)
        self.act = nn.GELU()
        self.conv2 = ConvNorm(hidden, dim, 1, 1, 0, norm_init=0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.act(self.conv1(x)))


class RepViTBlock(nn.Module):
    """Token mixing, optional gating, then channel mixing with a residual.

    Args:
        dim: Channels in and out.
        use_se: Whether this block gates. Alternates through a stage, starting on.
    """

    def __init__(self, dim: int, use_se: bool) -> None:
        super().__init__()
        self.token_mixer = RepVggDw(dim)
        self.se = SqueezeExcite(dim) if use_se else nn.Identity()
        self.channel_mixer = RepVitMlp(dim, dim * MLP_RATIO)

    def forward(self, x: Tensor) -> Tensor:
        x = self.se(self.token_mixer(x))
        return x + self.channel_mixer(x)


class RepVitStem(nn.Module):
    """Two strided 3x3 convolutions, taking the image to a quarter of its side.

    Args:
        in_chans: Image channels.
        out_dim: Width the first stage runs at.
    """

    def __init__(self, in_chans: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = ConvNorm(in_chans, out_dim // 2, 3, 2, 1)
        self.act1 = nn.GELU()
        self.conv2 = ConvNorm(out_dim // 2, out_dim, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.act1(self.conv1(x)))


class RepVitDownsample(nn.Module):
    """Halve the resolution and change the width, between one stage and the next.

    Args:
        in_dim: Width arriving.
        out_dim: Width leaving.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.pre_block = RepViTBlock(in_dim, use_se=False)
        self.spatial_downsample = ConvNorm(
            in_dim, in_dim, KERNEL_SIZE, 2, (KERNEL_SIZE - 1) // 2, groups=in_dim
        )
        self.channel_downsample = ConvNorm(in_dim, out_dim)
        self.ffn = RepVitMlp(out_dim, out_dim * MLP_RATIO)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_downsample(self.spatial_downsample(self.pre_block(x)))
        return x + self.ffn(x)


class RepVitStage(nn.Module):
    """One resolution: an optional downsample, then *depth* blocks.

    Args:
        in_dim: Width arriving.
        out_dim: Width this stage runs at.
        depth: How many blocks.
        downsample: Whether to halve the resolution first. False only on the first stage, which
            takes the stem's output at the width it already has.
    """

    def __init__(self, in_dim: int, out_dim: int, depth: int, downsample: bool = True) -> None:
        super().__init__()
        self.downsample = RepVitDownsample(in_dim, out_dim) if downsample else nn.Identity()
        # Gating alternates block by block, starting on -- which is why the checkpoint carries
        # ``se.*`` for the even blocks of every stage and nothing for the odd ones.
        self.blocks = nn.Sequential(
            *(RepViTBlock(out_dim, use_se=index % 2 == 0) for index in range(depth))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.downsample(x))


class RepViT(nn.Module):
    """RepViT-M1 as a feature pyramid: an image in, one map per stage out.

    The four maps come out at strides 4, 8, 16 and 32 -- for EdgeTAM's 1024-pixel square that is
    256, 128, 64 and 32 -- and are handed to the neck fine-to-coarse, which is the order it
    indexes them in.

    Args:
        in_chans: Image channels.

    Attributes:
        channel_list: Output widths coarsest-first, which is the order the neck's convolutions
            are built in. :class:`~.image_encoder.ImageEncoder` asserts the two agree, so this is
            the trunk's half of that contract.
    """

    def __init__(self, in_chans: int = 3) -> None:
        super().__init__()
        stages: dict[str, nn.Module] = {"stem": RepVitStem(in_chans, WIDTHS[0])}
        for index, (width, depth) in enumerate(zip(WIDTHS, DEPTHS)):
            stages[f"stages_{index}"] = RepVitStage(
                WIDTHS[index - 1] if index else width, width, depth, downsample=index > 0
            )
        # A ``ModuleDict`` rather than a ``Sequential`` because the checkpoint's names are
        # ``stages_0`` and not ``stages.0``: see the module docstring. Insertion order is the
        # execution order, and the stem is the first entry rather than a separate attribute for
        # the same reason -- that is how timm's flattened form spells it.
        self.body = nn.ModuleDict(stages)
        self.channel_list = list(WIDTHS[::-1])

    def forward(self, x: Tensor) -> list[Tensor]:
        """Run the trunk and return every stage's output.

        Args:
            x: ``(B, 3, H, W)`` normalised image batch.

        Returns:
            Four maps, finest first, at strides 4, 8, 16 and 32.
        """
        maps = []
        for name, stage in self.body.items():
            x = stage(x)
            if name != "stem":
                maps.append(x)
        return maps
