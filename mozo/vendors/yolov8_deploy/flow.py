# SPDX-License-Identifier: Apache-2.0
"""Dataflow of the composite blocks — the one thing a checkpoint does not record.

A checkpoint stores which children a block owns and every scalar those children need, but not the order
in which the children feed each other. Each function below supplies exactly that for one recorded class
name, reading split widths, residual flags and concat axes from the recorded attributes rather than
inferring them from tensor shapes.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn


class Block(nn.Module):
    """A composite module: its recorded children, its recorded scalars, and its dataflow."""

    def __init__(self, kind: str, flow: Callable, children: dict[str, nn.Module], attributes: dict[str, Any]):
        super().__init__()
        self.kind = kind
        self.flow = flow
        self.attributes = attributes
        for name, child in children.items():
            self.add_module(name, child)

    def forward(self, x):
        return self.flow(self, x)

    def extra_repr(self) -> str:
        return self.kind


def chain(block: Block, x: torch.Tensor) -> torch.Tensor:
    """Apply every child in its recorded order (convolution, optional norm, activation)."""
    for child in block.children():
        x = child(x)
    return x


def bottleneck(block: Block, x: torch.Tensor) -> torch.Tensor:
    """Two convolutions, added back onto the input when the recorded residual flag is set."""
    y = block.cv2(block.cv1(x))
    return x + y if block.attributes["add"] else y


def c2f(block: Block, x: torch.Tensor) -> torch.Tensor:
    """Split the entry projection into two halves of the recorded width, chain the bottlenecks over the
    second half, and merge every intermediate result."""
    parts = list(block.cv1(x).split(block.attributes["c"], 1))
    for unit in block.m:
        parts.append(unit(parts[-1]))
    return block.cv2(torch.cat(parts, 1))


def sppf(block: Block, x: torch.Tensor) -> torch.Tensor:
    """Pool the reduced feature map three times in succession and merge all four scales."""
    scales = [block.cv1(x)]
    for _ in range(3):
        scales.append(block.m(scales[-1]))
    return block.cv2(torch.cat(scales, 1))


def concat(block: Block, inputs: list[torch.Tensor]) -> torch.Tensor:
    """Join the incoming feature maps along the recorded axis."""
    return torch.cat(inputs, block.attributes["d"])


def distribution(block: Block, x: torch.Tensor) -> torch.Tensor:
    """Collapse the per-side bin distribution to a distance, weighting bins by the recorded weights."""
    batch, _, anchors = x.shape
    bins = block.attributes["c1"]
    spread = x.reshape(batch, 4, bins, anchors).permute(0, 2, 1, 3).softmax(1)
    return block.conv(spread).reshape(batch, 4, anchors)


_GRIDS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def anchor_grid(shapes: tuple, strides: tuple, dtype: torch.dtype, device: torch.device) -> tuple:
    """Return cell centres in feature units and the matching per-anchor stride, for all levels at once."""
    key = (shapes, strides, dtype, device)
    grid = _GRIDS.get(key)
    if grid is None:
        centres, scales = [], []
        for (rows, columns), step in zip(shapes, strides):
            y = torch.arange(rows, dtype=dtype, device=device) + 0.5
            x = torch.arange(columns, dtype=dtype, device=device) + 0.5
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            centres.append(torch.stack((grid_x.reshape(-1), grid_y.reshape(-1))))
            scales.append(torch.full((1, rows * columns), step, dtype=dtype, device=device))
        grid = (torch.cat(centres, 1)[None], torch.cat(scales, 1)[None])
        _GRIDS[key] = grid
    return grid


def detect(block: Block, features: list[torch.Tensor]) -> torch.Tensor:
    """Run both head branches per level and decode to boxes in letterboxed pixels plus class scores."""
    levels = [torch.cat((box(f), labels(f)), 1) for box, labels, f in zip(block.cv2, block.cv3, features)]
    flat = torch.cat([level.flatten(2) for level in levels], 2)
    sides, scores = flat.split((block.attributes["no"] - block.attributes["nc"], block.attributes["nc"]), 1)
    shapes = tuple((level.shape[2], level.shape[3]) for level in levels)
    centres, scales = anchor_grid(shapes, block.attributes["stride"], flat.dtype, flat.device)
    left_top, right_bottom = block.dfl(sides).chunk(2, 1)
    upper_left, lower_right = centres - left_top, centres + right_bottom
    boxes = torch.cat(((upper_left + lower_right) / 2, lower_right - upper_left), 1) * scales
    return torch.cat((boxes, scores.sigmoid()), 1)


#: Recorded class name -> its dataflow. A composite outside this table stops the build.
DATAFLOW: dict[str, Callable] = {
    "Conv": chain,
    "Bottleneck": bottleneck,
    "C2f": c2f,
    "SPPF": sppf,
    "Concat": concat,
    "DFL": distribution,
    "Detect": detect,
}
