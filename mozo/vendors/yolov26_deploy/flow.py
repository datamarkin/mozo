# SPDX-License-Identifier: Apache-2.0
"""How each composite block routes tensors through its children.

This is the one thing a checkpoint does not record, so it is the one thing written by hand. Every
width, repeat count, split point and attention shape used below is read from the checkpoint; only
the wiring lives here.

The head is where this family departs from its siblings. It is trained to fire once per object, so
only the ``one2one_*`` branches are evaluated and there is no non-maximum suppression anywhere in
the package. The decode and the top-k that turn those branches into a detection list live on the
network itself, in :mod:`~mozo.vendors.yolov26_deploy.model`, because they need the anchor grid.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F


def _chain(block, x):
    """Convolution units: every recorded child applied in order (conv, norm, activation)."""
    for child in block.children():
        x = child(x)
    return x


def _bottleneck(block, x):
    """Two convolutions, optionally added back onto the input."""
    y = block.cv2(block.cv1(x))
    return x + y if block.rec("add") else y


def _csp_split(block, x):
    """Split the entry projection into the two recorded halves: kept, and processed."""
    width = block.rec("c")
    return block.cv1(x).split((width, width), 1)


def _c3k2(block, x):
    """Both halves of the split, plus the output of every unit in the chain, all concatenated."""
    kept, running = _csp_split(block, x)
    parts = [kept, running]
    for unit in block.m:
        running = unit(running)
        parts.append(running)
    return block.cv2(torch.cat(parts, 1))


def _c3k(block, x):
    """Chain over one branch, bare projection on the other, concatenated and projected out."""
    return block.cv3(torch.cat((block.m(block.cv1(x)), block.cv2(x)), 1))


def _sppf(block, x):
    """Pool the same map repeatedly; each successive pooling widens the receptive field."""
    pooled = [block.cv1(x)]
    for _ in range(block.rec("n")):
        pooled.append(block.m(pooled[-1]))
    y = block.cv2(torch.cat(pooled, 1))
    return x + y if block.rec("add") else y


def _c2psa(block, x):
    """Attention applied to one half of the split, the other half carried through untouched."""
    kept, running = _csp_split(block, x)
    return block.cv2(torch.cat((kept, block.m(running)), 1))


def _psablock(block, x):
    """Attention then feed-forward, each optionally residual."""
    residual = block.rec("add")
    x = x + block.attn(x) if residual else block.attn(x)
    return x + block.ffn(x) if residual else block.ffn(x)


def _attention(block, x):
    """Multi-head self-attention over the flattened map, plus a depthwise positional term."""
    heads, key_dim, head_dim = block.rec("num_heads"), block.rec("key_dim"), block.rec("head_dim")
    batch, channels, height, width = x.shape
    qkv = block.qkv(x).view(batch, heads, 2 * key_dim + head_dim, height * width)
    q, k, v = qkv.split((key_dim, key_dim, head_dim), 2)
    weights = torch.softmax(q.transpose(-2, -1) @ k * block.rec("scale"), -1)
    mixed = (v @ weights.transpose(-2, -1)).reshape(batch, channels, height, width)
    return block.proj(mixed + block.pe(v.reshape(batch, channels, height, width)))


def _concat(block, xs):
    return torch.cat(xs, block.rec("d"))


def _proto(block, x):
    """The mask prototype stack: a convolution, a *learned* upsample, then two more.

    ``upsample`` is a ``ConvTranspose2d``, not an interpolation -- see ``build._convtranspose2d``.

    A helper rather than a ``DATAFLOW`` row: no checkpoint this package serves records a bare
    ``Proto``, and a table whose rule is "an error, never a guess" should not claim a class it has
    never been handed. ``Proto26`` inherits these four children and calls this directly.
    """
    return block.cv3(block.cv2(block.upsample(block.cv1(x))))


def _proto26(block, feats):
    """YOLO26's prototypes, refined from all three levels rather than from the finest alone.

    The two coarser maps are projected to the finest one's width by a 1x1 convolution, resized up
    by nearest neighbour and summed. Base ``Proto`` takes a single tensor and this takes the list,
    which is why building one where the checkpoint records the other drops the refinement, raises
    nothing, and still produces plausible masks.

    ``semseg`` is deliberately not evaluated. It is a training-time semantic-segmentation head --
    upstream's own ``fuse()`` deletes it before inference and its ``forward`` returns it only while
    training -- so it is built for the strict load and never run.
    """
    feat = feats[0]
    for refine, level in zip(block.feat_refine, feats[1:]):
        feat = feat + F.interpolate(refine(level), size=feat.shape[2:], mode="nearest")
    return _proto(block, block.feat_fuse(feat))


def _segment26(block, feats):
    """Per anchor: box distances, class logits and mask coefficients. Plus the prototypes.

    The same one-to-one branches ``_detect`` evaluates, with ``one2one_cv4`` alongside them, so the
    channel order is ``[4 distances, nc classes, nm coefficients]`` -- which is the order
    ``network._decode`` splits and the order upstream's own ``postprocess`` expects.

    Returns the raw grid *and* the prototypes, because the coefficients are meaningless without
    them and nothing downstream should have to reach back into the head to find them.
    """
    levels = block.rec("nl")
    if len(feats) != levels:
        raise ValueError(f"segmentation head recorded nl={levels} but received {len(feats)} feature maps")
    outputs = [
        torch.cat((box(feat).flatten(2), cls(feat).flatten(2), mask(feat).flatten(2)), 1)
        for feat, box, cls, mask in
        zip(feats, block.one2one_cv2, block.one2one_cv3, block.one2one_cv4)
    ]
    return torch.cat(outputs, 2), block.proto(feats)


def _detect(block, feats):
    """Per level, box distances and class logits side by side; levels flattened and joined.

    Only the ``one2one_*`` branches run. The ``cv2``/``cv3`` pair is the training-time
    one-to-many assignment head, and ``dfl`` is an identity because this model regresses box
    distances with a single bin; both are built so the weight load is complete, and neither is
    ever evaluated.
    """
    levels = block.rec("nl")
    if len(feats) != levels:
        raise ValueError(f"detection head recorded nl={levels} but received {len(feats)} feature maps")
    # Flattened before the per-level concatenation rather than after: joining the two branches at
    # full spatial rank builds a (1, 84, H, W) intermediate that the outer concatenation then
    # copies again.
    outputs = [
        torch.cat((box(feat).flatten(2), cls(feat).flatten(2)), 1)
        for feat, box, cls in zip(feats, block.one2one_cv2, block.one2one_cv3)
    ]
    return torch.cat(outputs, 2)


DATAFLOW: dict[str, Callable[[Any, Any], Any]] = {
    "Conv": _chain,
    "DWConv": _chain,
    "Bottleneck": _bottleneck,
    "C3k2": _c3k2,
    "C3k": _c3k,
    "SPPF": _sppf,
    "C2PSA": _c2psa,
    "PSABlock": _psablock,
    "Attention": _attention,
    "Concat": _concat,
    "Detect": _detect,
    "Segment26": _segment26,
    "Proto26": _proto26,
}
