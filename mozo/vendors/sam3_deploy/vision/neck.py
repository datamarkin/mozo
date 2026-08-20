# SPDX-License-Identifier: Apache-2.0
"""The FPN that turns the trunk's single 72x72 grid into a pyramid.

Derived from ``transformers/models/sam3/modeling_sam3.py`` (Apache-2.0), with one structural
addition that implementation does not carry.

**The neck is dual.** Upstream's ``Sam3DualViTDetNeck`` holds two conv stacks of identical shape
and different weights, and runs both over the *same* trunk output: one feeds the concept head
(text and exemplar prompts), the other feeds the click head that the checkpoint's ``tracker.``
weights drive. ``transformers`` implements only the concept half, so only one stack appears there.
Both are built here because the checkpoint has weights for both and a strict load must find
somewhere to put them. What they do *not* do is serve one encode between them: the two heads
preprocess an image differently, so each runs its own trunk pass and reads its own stack. See
:meth:`~..predictor.Segmenter.encode_click`.

The position encoding is a function of shape alone -- not of the image, and not of which stack
produced the map -- so it lives in :mod:`..position`, which memoises it. Only the coarsest
level's is returned, because that is the only one anything downstream reads: the grounding stage
attends over the 72x72 grid and the mask head consumes the finer levels as features, not as
positions. Building the other two costs 34 ms and holds 106 MB for the life of the process.
"""

from __future__ import annotations

from torch import Tensor, nn

from ..config import Spec
from ..position import SinePositionEmbedding

__all__ = ["FpnLevel", "Neck"]


class FpnLevel(nn.Module):
    """One pyramid level: resample the trunk's grid, then project it to the FPN width.

    Args:
        channels: Trunk width in.
        hidden: FPN width out.
        scale: Resolution multiplier. Only 4.0, 2.0, 1.0 and 0.5 exist in this model, and each
            builds a different resampler -- transposed convolutions up, max pooling down.
    """

    def __init__(self, channels: int, hidden: int, scale: float):
        super().__init__()
        layers: list[nn.Module] = []
        if scale == 4.0:
            layers = [
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2),
                nn.GELU(),
                nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size=2, stride=2),
            ]
            resampled = channels // 4
        elif scale == 2.0:
            layers = [nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2)]
            resampled = channels // 2
        elif scale == 1.0:
            resampled = channels
        elif scale == 0.5:
            layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            resampled = channels
        else:
            raise ValueError(f"unsupported FPN scale factor {scale}")

        self.scale_layers = nn.ModuleList(layers)
        self.proj1 = nn.Conv2d(resampled, hidden, kernel_size=1)
        self.proj2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.scale_layers:
            x = layer(x)
        return self.proj2(self.proj1(x))


class Neck(nn.Module):
    """The dual FPN. One trunk output in, two pyramids plus their shared position encoding out.

    Args:
        spec: The vision geometry.
    """

    def __init__(self, spec: Spec):
        super().__init__()
        self.position_encoding = SinePositionEmbedding(features=spec.fpn_hidden // 2)
        channels = spec.trunk.hidden
        self.levels = nn.ModuleList(
            FpnLevel(channels, spec.fpn_hidden, scale) for scale in spec.scale_factors
        )
        self.click_levels = nn.ModuleList(
            FpnLevel(channels, spec.fpn_hidden, scale) for scale in spec.scale_factors
        )
        # Upstream builds every level and then discards the lowest-resolution ones. The modules
        # are still constructed, because the checkpoint has weights for them and a strict load
        # must find somewhere to put them -- but running them is pure waste, so ``forward`` stops
        # here. Skipping the 36x36 level saves roughly 2.2 GFLOP per image across both stacks.
        self.keep = len(spec.scale_factors) - spec.scalp

    #: The stacks :meth:`forward` can build, in the order they are defined.
    STACKS = ("concept", "click")

    def forward(
        self, x: Tensor, stacks: tuple[str, ...] = STACKS
    ) -> dict[str, list[Tensor] | Tensor]:
        """Run the requested stacks over ``x``, for the levels that survive the scalp.

        Both stacks are built and loaded, but a caller rarely wants both at once: the two heads
        preprocess an image differently, so each encodes separately and reads one stack. Running
        the other and discarding it costs 62 ms on an M-series GPU and a transient 111 MB, which
        is the same kind of waste the scalp above exists to avoid.

        Args:
            x: ``(B, hidden, grid, grid)`` from the trunk.
            stacks: Which pyramids to build. Defaults to both.

        Returns:
            The requested pyramids -- coarsest last -- and ``positions``, the coarsest level's
            position encoding. ``positions`` is a function of shape, device and dtype alone, so
            it does not depend on which stack was asked for.

        Raises:
            ValueError: If ``stacks`` names nothing, or names something this neck does not have.
        """
        unknown = set(stacks) - set(self.STACKS)
        if unknown or not stacks:
            raise ValueError(f"stacks must be a non-empty subset of {self.STACKS}, got {stacks!r}")

        built = {
            name: [level(x) for level in levels[: self.keep]]
            for name, levels in (("concept", self.levels), ("click", self.click_levels))
            if name in stacks
        }
        coarse = next(iter(built.values()))[-1]
        positions = self.position_encoding(
            coarse.shape[-2], coarse.shape[-1], coarse.device, coarse.dtype
        )
        return {**built, "positions": positions}
