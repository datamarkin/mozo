# SPDX-License-Identifier: Apache-2.0
"""The trunk and the feature-pyramid neck that sits on it.

Taken verbatim from ``sam2/modeling/backbones/image_encoder.py`` in ``facebookresearch/EdgeTAM``,
which is byte-identical to SAM 2's own file. The whole difference between the two models on this
path is which trunk gets passed in -- SAM 2 hands it a Hiera, EdgeTAM a RepViT -- and the neck
adapts by taking its widths as an argument.

``scalp`` discards the coarsest level after the neck has run. EdgeTAM sets it to 1, so the neck
sees four maps and three come out: the trunk's stride-32 output contributes to the top-down
pathway and is then dropped.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FpnNeck", "ImageEncoder"]


class ImageEncoder(nn.Module):
    """A trunk, a neck, and the rule for how many of the neck's levels survive.

    Args:
        trunk: The backbone. Must expose ``channel_list`` coarsest-first.
        neck: The feature pyramid.
        scalp: How many of the coarsest levels to discard after the neck.
    """

    def __init__(self, trunk: nn.Module, neck: nn.Module, scalp: int = 0) -> None:
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"Channel dims of trunk and neck do not match. Trunk: {self.trunk.channel_list}, "
            f"neck: {self.neck.backbone_channel_list}"
        )

    def forward(self, sample: torch.Tensor) -> dict:
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        return {"vision_features": src, "vision_pos_enc": pos, "backbone_fpn": features}


class FpnNeck(nn.Module):
    """A feature pyramid with lateral 1x1 projections and a partial top-down pathway.

    A modified variant of FPN: there is no output convolution, and which levels receive top-down
    features is configurable rather than being all of them.

    Args:
        position_encoding: Applied to each output level.
        d_model: Width every level is projected to.
        backbone_channel_list: Trunk output widths, coarsest first.
        kernel_size: Lateral convolution kernel.
        stride: Lateral convolution stride.
        padding: Lateral convolution padding.
        fpn_interp_model: How top-down features are upsampled.
        fuse_type: ``sum`` or ``avg`` for combining lateral and top-down.
        fpn_top_down_levels: Which levels get top-down features. EdgeTAM uses ``[2, 3]``, so
            levels 0 and 1 carry lateral features only.
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )
            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]) -> tuple[list, list]:
        """Project and fuse the trunk's maps.

        Args:
            xs: The trunk's outputs, finest first.

        Returns:
            One list of fused maps and one of their position encodings, both finest first.
        """
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass, in top-down order (from low to high resolution)
        prev_features = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos
