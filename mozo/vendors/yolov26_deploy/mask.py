# SPDX-License-Identifier: Apache-2.0
"""Turning mask coefficients and prototypes into one boolean mask per detection.

The head returns ``nm`` coefficients per detection and a single ``(nm, H/4, W/4)`` stack of
prototypes shared by all of them. A mask is the linear combination: coefficients times prototypes,
resized back to the source image, cropped to its own box, thresholded at zero.

**This reproduces upstream's ``retina_masks`` path, not its default one, and that is a deliberate
divergence.** By default the published predictor returns masks at the *network's* resolution --
640x640 letterboxed -- while scaling the boxes back to source pixels, so the two do not describe
the same coordinate system and the caller is expected to know. mozo returns boxes in source pixels,
so a mask that did not match them would be unusable. Upstream's own ``process_mask_native`` is the
path that produces both in source pixels, and it is what this module follows step for step.

Three details here are load-bearing and none of them is the obvious reading:

**The threshold is on logits, at zero, and it comes last.** There is no sigmoid anywhere. Sigmoid
then thresholding at 0.5 picks the same side of the same boundary, but only if nothing in between
is non-monotonic -- and the bilinear resize in the middle is exactly that, so the two disagree
along every edge.

**The padding is removed at prototype resolution, before the resize**, by a crop whose bounds are
rounded with an asymmetric nudge (``round(pad - 0.1)`` against ``round(pad + 0.1)``). Deriving the
crop after the resize, or rounding it symmetrically, moves the mask by a pixel or two against the
box it belongs to.

**The box crop runs in source pixels**, on the already-scaled box -- unlike the default path, which
crops in network pixels against a box scaled by the prototype ratio.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["assemble", "crop"]

#: Below this many masks, upstream takes a per-mask loop instead of a vectorised comparison. The
#: two are *not* the same computation -- see :func:`crop` -- so the count decides the arithmetic
#: and the threshold has to be carried rather than chosen.
LOOP_BELOW = 50


def crop(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Zero everything outside each mask's own box, in that mask's coordinate system.

    Args:
        masks: ``(n, h, w)`` mask logits.
        boxes: ``(n, 4)`` ``x1, y1, x2, y2`` in the same pixels as *masks*.

    Returns:
        *masks*, with everything outside each box set to zero.

    Note:
        Upstream carries two implementations and picks between them on ``n < 50 and not cuda``,
        described there as a speed choice. They do not agree: the loop rounds each edge to a whole
        pixel and the vectorised form compares against the unrounded float, so a box edge at
        ``x2 = 10.4`` clears column 10 in one and keeps it in the other. Both are reproduced, and
        the same condition selects between them, because a mask that changes with the number of
        detections in the picture is upstream's behaviour rather than a defect to tidy away.
    """
    if masks.shape[0] < LOOP_BELOW and not masks.is_cuda:
        for index, (x1, y1, x2, y2) in enumerate(boxes.round().int()):
            masks[index, :y1] = 0
            masks[index, y2:] = 0
            masks[index, :, :x1] = 0
            masks[index, :, x2:] = 0
        return masks
    _, height, width = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    columns = torch.arange(width, device=masks.device, dtype=x1.dtype)[None, None, :]
    rows = torch.arange(height, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((columns >= x1) * (columns < x2) * (rows >= y1) * (rows < y2))


def _unpadded(masks: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Cut the letterbox border off *masks* and resize what is left to *shape*.

    The border is derived from the two shapes rather than from the letterbox that produced it, so
    this needs nothing carried forward from preprocessing -- which is also what makes the
    asymmetric rounding below the only thing standing between the mask and a one-pixel shift.
    """
    padded_h, padded_w = masks.shape[1:]
    height, width = shape
    if (padded_h, padded_w) == (height, width):
        return masks

    gain = min(padded_h / height, padded_w / width)
    pad_w = (padded_w - width * gain) / 2
    pad_h = (padded_h - height * gain) / 2
    # The nudges are upstream's, and they are not symmetric: the top edge rounds a tenth of a pixel
    # down and the bottom edge a tenth up, so a border of exactly half a pixel resolves inwards on
    # both sides rather than landing on whichever way the tie breaks.
    top, left = round(pad_h - 0.1), round(pad_w - 0.1)
    bottom, right = padded_h - round(pad_h + 0.1), padded_w - round(pad_w + 0.1)
    # The batch axis is added and dropped here rather than by the caller: it exists only because
    # ``interpolate`` wants NCHW, which is this function's business and nobody else's.
    return F.interpolate(masks[None, ..., top:bottom, left:right].float(), shape,
                         mode="bilinear")[0]


def assemble(protos: torch.Tensor, coefficients: torch.Tensor, boxes: torch.Tensor,
             shape: tuple[int, int]) -> torch.Tensor:
    """Build one boolean mask per detection, in the source image's pixels.

    Args:
        protos: ``(nm, mh, mw)`` prototypes, as the head returns them for one image.
        coefficients: ``(n, nm)`` coefficients, one row per surviving detection.
        boxes: ``(n, 4)`` boxes already mapped back to source pixels.
        shape: The source image's ``(height, width)``.

    Returns:
        ``(n, height, width)`` boolean masks, aligned with *boxes*.
    """
    channels, mask_h, mask_w = protos.shape
    if not len(coefficients):
        # ``interpolate`` refuses a zero-channel tensor, so a threshold nothing clears would raise
        # from inside the resize rather than returning nothing. A detection variant answers an
        # empty picture with an empty result, and so does this.
        return torch.zeros((0, *shape), dtype=torch.bool)

    masks = (coefficients @ protos.float().view(channels, -1)).view(-1, mask_h, mask_w)
    # Thresholded before the crop rather than after. ``crop`` only ever writes zero and zero fails
    # ``> 0``, so the two orders give bit-identical masks -- but this one hands ``crop`` a boolean
    # an eighth of the size and lets the full-resolution float go before the crop instead of after
    # it, which at 300 detections is the difference between 2.9 GB alive and 0.7.
    #
    # ``gt_`` is in-place, so on a float tensor it writes 1.0 and 0.0 and keeps the dtype --
    # upstream casts that to ``byte`` and mozo to ``bool``. The comparison is the same one; only
    # the container differs, and a mask is a yes-or-no per pixel.
    return crop(_unpadded(masks, shape).gt_(0.0).bool(), boxes)
