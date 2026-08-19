# SPDX-License-Identifier: Apache-2.0
"""How each composite block routes tensors through its children.

This is the one thing a checkpoint does not record, so it is the one thing written by hand.
Every width, repeat count, residual flag, split point and attention shape used below is read
from the checkpoint; only the wiring lives here.
"""

from __future__ import annotations

from functools import cache

import torch


def chain(block, x):
    """Feed the tensor through the recorded children in order."""
    for child in block.children():
        x = child(x)
    return x


def bottleneck(block, x):
    """Two convolutions, with the residual connection the ``add`` flag records."""
    y = block.cv2(block.cv1(x))
    return x + y if block.value("add") else y


def c3k2(block, x):
    """Split into two halves of the recorded width, then grow one branch per unit in ``m``."""
    width = block.value("c")
    parts = list(block.cv1(x).split((width, width), dim=1))
    for unit in block.m:
        parts.append(unit(parts[-1]))
    return block.cv2(torch.cat(parts, 1))


def c3k(block, x):
    """Run the unit stack on one projection and concatenate it with the untouched other."""
    return block.cv3(torch.cat((block.m(block.cv1(x)), block.cv2(x)), 1))


def sppf(block, x):
    """Pool the projection three times in a row and concatenate the four receptive fields."""
    stages = [block.cv1(x)]
    for _ in range(3):
        stages.append(block.m(stages[-1]))
    return block.cv2(torch.cat(stages, 1))


def c2psa(block, x):
    """Send half the recorded width through the attention stack and keep the other half."""
    width = block.value("c")
    kept, attended = block.cv1(x).split((width, width), dim=1)
    return block.cv2(torch.cat((kept, block.m(attended)), 1))


def psablock(block, x):
    """Attention then feed-forward, each residual when the recorded ``add`` flag says so."""
    residual = block.value("add")
    x = x + block.attn(x) if residual else block.attn(x)
    return x + block.ffn(x) if residual else block.ffn(x)


def attention(block, x):
    """Multi-head self attention over the spatial grid, with a depthwise positional term.

    Head count, query/key width, value width and the softmax scale are all recorded, so nothing
    is inferred from the channel count.
    """
    heads, key_dim, head_dim = block.value("num_heads"), block.value("key_dim"), block.value("head_dim")
    batch, _, height, width = x.shape
    positions = height * width
    packed = block.qkv(x).view(batch, heads, 2 * key_dim + head_dim, positions)
    queries, keys, values = packed.split((key_dim, key_dim, head_dim), dim=2)
    weights = torch.softmax(queries.transpose(-2, -1) @ keys * block.value("scale"), dim=-1)
    grid = (batch, heads * head_dim, height, width)
    mixed = (values @ weights.transpose(-2, -1)).reshape(grid)
    return block.proj(mixed + block.pe(values.reshape(grid)))


def concat(block, xs):
    """Join the incoming tensors along the recorded axis."""
    return torch.cat(xs, block.value("d"))


def dfl(block, x):
    """Turn each box side's distribution over bins into a single distance.

    The expectation is taken by the recorded 1x1 convolution, whose weights are the bin indices.
    """
    batch, _, anchors = x.shape
    bins = block.value("c1")
    distribution = x.view(batch, 4, bins, anchors).transpose(1, 2).softmax(1)
    return block.conv(distribution).view(batch, 4, anchors)


@cache
def anchor_grid(shapes, strides, dtype, device):
    """Cell centres and their strides for one set of feature-map shapes, cached per grid."""
    centres, scales = [], []
    for (rows, cols), stride in zip(shapes, strides):
        y, x = torch.meshgrid(
            torch.arange(rows, dtype=dtype, device=device) + 0.5,
            torch.arange(cols, dtype=dtype, device=device) + 0.5,
            indexing="ij",
        )
        centres.append(torch.stack((x.reshape(-1), y.reshape(-1))))
        scales.append(torch.full((1, rows * cols), stride, dtype=dtype, device=device))
    return torch.cat(centres, 1), torch.cat(scales, 1)


def detect(block, features):
    """Decode the per-level box and class branches into boxes and class scores.

    Output is ``(batch, 4 + nc, anchors)``: centre-form boxes in input-image pixels followed by
    per-class probabilities.
    """
    outputs = [torch.cat((box(f), cls(f)), 1) for box, cls, f in zip(block.cv2, block.cv3, features)]
    flat = torch.cat([out.flatten(2) for out in outputs], 2)
    distances, scores = flat.split((4 * block.value("reg_max"), block.value("nc")), 1)
    shapes = tuple((int(out.shape[-2]), int(out.shape[-1])) for out in outputs)
    centres, strides = anchor_grid(shapes, block.value("strides"), flat.dtype, flat.device)
    top_left, bottom_right = block.dfl(distances).chunk(2, 1)
    corner_min, corner_max = centres - top_left, centres + bottom_right
    boxes = torch.cat(((corner_min + corner_max) / 2, corner_max - corner_min), 1) * strides
    return torch.cat((boxes, scores.sigmoid()), 1)


DATAFLOW = {
    "Conv": chain,
    "DWConv": chain,
    "Bottleneck": bottleneck,
    "C3k2": c3k2,
    "C3k": c3k,
    "SPPF": sppf,
    "C2PSA": c2psa,
    "PSABlock": psablock,
    "Attention": attention,
    "Concat": concat,
    "DFL": dfl,
    "Detect": detect,
}
