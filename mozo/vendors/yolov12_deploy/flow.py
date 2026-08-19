# SPDX-License-Identifier: Apache-2.0
"""Dataflow of the composite blocks — the only logic a checkpoint does not record.

A checkpoint records every module's class name, its children and each leaf's complete
hyper-parameters, but not how a composite block routes tensors through its children.
Each function below supplies exactly that for one recorded class name, reading the
scalars the checkpoint stored alongside the children.

``SCALARS`` declares, per class, which recorded scalars its dataflow reads; the builder
copies those onto the block and nothing else.
"""

from __future__ import annotations

import functools

import torch


def _chain(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Conv / DWConv: the recorded children applied in order (convolution, norm, activation)."""
    for child in block.children():
        x = child(x)
    return x


def _concat(block: torch.nn.Module, inputs: list) -> torch.Tensor:
    """Concat: join the incoming feature maps along the recorded axis."""
    return torch.cat(inputs, block.d)


def _bottleneck(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Bottleneck: two convolutions, with a residual connection when the block records one."""
    y = block.cv2(block.cv1(x))
    return x + y if block.add else y


def _c3k(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """C3k: a bottleneck stack on one branch, a plain projection on the other, then a merge."""
    return block.cv3(torch.cat((block.m(block.cv1(x)), block.cv2(x)), 1))


def _c3k2(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """C3k2: split the entry projection in two, chain the sub-blocks, concatenate every stage."""
    stages = list(block.cv1(x).split(block.c, dim=1))
    for sub in block.m:
        stages.append(sub(stages[-1]))
    return block.cv2(torch.cat(stages, 1))


def _a2c2f(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """A2C2f: chain the sub-blocks off the entry projection and concatenate every stage.

    The larger variants record a per-channel ``gamma`` on this block and the smaller ones do not.
    Where it is present it scales a residual branch: the block's input is added back to its output,
    weighted per channel.

    That is the one piece of dataflow here inferred rather than read, so it was tested against its
    alternatives on ``large`` and ``xlarge`` rather than assumed. Ignoring ``gamma`` gives 300
    detections at confidence 1.000, naming ovens, toilets and teddy bears in a photograph of a
    desk. Using it as a plain output scale, without the residual, gives zero detections. Adding
    the residual gives 14 detections at a top confidence of 0.956 and a mean of 0.80 -- which is
    what the same photograph yields on ``medium``, the largest variant that records no ``gamma``
    at all and therefore needs no interpretation.

    The shapes agree independently: it is recorded only on blocks whose input and output channel
    counts are equal, which is the only place a residual can be formed, and it is shaped exactly
    like those channels.
    """
    stages = [block.cv1(x)]
    for sub in block.m:
        stages.append(sub(stages[-1]))
    out = block.cv2(torch.cat(stages, 1))
    gamma = getattr(block, "gamma", None)
    return out if gamma is None else x + gamma.view(1, -1, 1, 1) * out


def _ablock(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """ABlock: residual area attention followed by a residual channel MLP."""
    x = x + block.attn(x)
    return x + block.mlp(x)


def _to_maps(t: torch.Tensor, batch: int, channels: int, height: int, width: int) -> torch.Tensor:
    """Fold (group, head, token, head_dim) back into a (batch, channel, height, width) map.

    One permute and one reshape. Transposing to (batch, tokens, channels) and back cost two full
    copies of the feature map -- the intermediate reshape forces one because its source is
    strided, and the ``contiguous()`` that followed forces another. That was 4.9 MB per forward on
    nano and 49 MB on xlarge, for 16 to 32 calls each. ``reshape`` on the permuted view produces
    the contiguous result the convolution below wants, so the copy that remains is the one the
    old comment was paying for on purpose.
    """
    groups = t.shape[0] // batch
    return (t.view(batch, groups, t.shape[1], t.shape[2], t.shape[3])
             .permute(0, 2, 4, 1, 3)
             .reshape(batch, channels, height, width))


def _area_attention(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """AAttn: attention within equal contiguous groups of tokens, plus a positional value path."""
    batch, channels, height, width = x.shape
    tokens = height * width
    if tokens % block.area:
        raise ValueError(
            f"area attention splits the token sequence into {block.area} equal groups, but a "
            f"{height}x{width} feature map has {tokens} tokens, which {block.area} does not divide"
        )
    groups = batch * block.area
    span = tokens // block.area
    # Transposing to (batch, token, channel) first is what makes the next reshape a pure
    # regrouping of the token axis, so every group is a contiguous run of tokens.
    qkv = block.qkv(x).flatten(2).transpose(1, 2).reshape(groups, span, 3 * channels)
    # Each head owns a contiguous slice of the projection's channels, holding its query, key
    # and value in that order.
    query, key, value = qkv.view(groups, span, block.num_heads, 3 * block.head_dim).split(block.head_dim, dim=3)
    query, key, value = (t.transpose(1, 2) for t in (query, key, value))
    weights = (query @ key.transpose(-2, -1) * block.head_dim**-0.5).softmax(-1)
    attended = _to_maps(weights @ value, batch, channels, height, width)
    return block.proj(attended + block.pe(_to_maps(value, batch, channels, height, width)))


def _dfl(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """DFL: read each box side's bin distribution as a single distance.

    The recorded 1x1 convolution holds the bin indices, so taking the expectation over the
    softmaxed bins is just that convolution applied across the bin axis.
    """
    batch, _, anchors = x.shape
    distribution = x.view(batch, 4, block.c1, anchors).transpose(1, 2).softmax(1)
    return block.conv(distribution).view(batch, 4, anchors)


@functools.lru_cache(maxsize=4)
def _anchor_grid(shapes: tuple, strides: tuple, dtype: torch.dtype, device: torch.device) -> tuple:
    """Cell centres and their strides for every detection level, laid out row by row."""
    centres, scales = [], []
    for (height, width), stride in zip(shapes, strides):
        columns = torch.arange(width, dtype=dtype, device=device) + 0.5
        rows = torch.arange(height, dtype=dtype, device=device) + 0.5
        grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
        centres.append(torch.stack((grid_x.flatten(), grid_y.flatten())))
        scales.append(torch.full((1, height * width), stride, dtype=dtype, device=device))
    return torch.cat(centres, 1), torch.cat(scales, 1)


def _detect(block: torch.nn.Module, features: list) -> torch.Tensor:
    """Detect: per level a box branch and a class branch, decoded into boxes over the whole image.

    The head also carries ``anchors``, ``strides`` and ``shape`` attributes left over from
    training. They describe whatever batch ran last and are wrong for any other input size, so
    the grid is built here from the feature maps in hand and the recorded per-level strides.
    """
    # The two branches are concatenated separately rather than merged per level and split again.
    # Merging built a (batch, 4*reg_max + nc, anchors) tensor -- 4.8 MB on nano -- only for the
    # next line to tear it back into the two halves it was made of.
    distances = torch.cat([box(f).flatten(2) for f, box in zip(features, block.cv2)], 2)
    class_scores = torch.cat([scores(f).flatten(2) for f, scores in zip(features, block.cv3)], 2)
    shapes = tuple((int(f.shape[2]), int(f.shape[3])) for f in features)
    centres, strides = _anchor_grid(shapes, block.stride, distances.dtype, distances.device)
    top_left, bottom_right = block.dfl(distances).chunk(2, 1)
    corner1, corner2 = centres - top_left, centres + bottom_right
    boxes = torch.cat(((corner1 + corner2) * 0.5, corner2 - corner1), 1) * strides
    return torch.cat((boxes, class_scores.sigmoid()), 1)


DATAFLOW = {
    "Conv": _chain,
    "DWConv": _chain,
    "Concat": _concat,
    "Bottleneck": _bottleneck,
    "C3k": _c3k,
    "C3k2": _c3k2,
    "A2C2f": _a2c2f,
    "ABlock": _ablock,
    "AAttn": _area_attention,
    "DFL": _dfl,
    "Detect": _detect,
}

#: Tensors a block owns itself rather than delegating to a child, by recorded class name.
#: Declared rather than derived from whatever the checkpoint happens to record, because the
#: builder's "records unusable" check is what surfaced ``gamma`` in the first place -- deriving
#: this from the file would register every recorded tensor and make that check vacuous, so the
#: next unrecognised parameter would load silently and never be read.
TENSORS = {
    "A2C2f": ("gamma",),
}

SCALARS = {
    "Concat": ("d",),
    "Bottleneck": ("add",),
    "C3k2": ("c",),
    "AAttn": ("area", "num_heads", "head_dim"),
    "DFL": ("c1",),
    "Detect": ("nc", "no", "stride"),
}
