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
path that produces both in source pixels, and this module reproduces what it does.

**Which upstream, though, is a decision rather than a measurement.** Nothing about mask assembly is
stored in a checkpoint -- post-processing is not learned -- and it has moved: `ultralytics` 8.3.63
crops with a single vectorised comparison and unpads with ``int(pad)``, where 8.4.0 has two crop
branches and rounds with an asymmetric nudge. This package reproduces **8.4.0**, the release the
checkpoints mozo publishes are taken from, and PROVENANCE says so.

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

#: At or above this many masks, upstream compares each box edge as it stands; below it, and off
#: the GPU, it rounds the edge to a whole pixel first. The two do not agree, so the count decides
#: the arithmetic and the threshold has to be carried rather than chosen. See :func:`crop`.
ROUNDS_BELOW = 50


def crop(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Zero everything outside each mask's own box, in that mask's coordinate system.

    Args:
        masks: ``(n, h, w)`` masks, modified in place.
        boxes: ``(n, 4)`` ``left, top, right, bottom`` in the same pixels as *masks*, and inside
            the image -- :func:`~mozo.vendors.yolov11_deploy.image.to_original` has already
            clipped them to it.

    Returns:
        *masks*, with everything outside each box set to zero.

    Note:
        Upstream carries two implementations of this and picks between them on the mask count,
        describing the choice as one of speed. It is not only that. Below the threshold it rounds
        each edge to a whole pixel and above it compares the edge as it stands, so a box ending at
        ``right = 10.4`` clears column 10 in one and keeps it in the other, and a picture with 49
        objects in it is cropped by a different rule than the same picture with 50.

        That rounding is the entire difference between them, so it is the entire difference here:
        one comparison, against edges that were rounded first when the count says upstream would
        have rounded them. Reproduced rather than tidied away, because it is upstream's behaviour
        and these masks are checked against it.

        Each axis is applied on its own and in place. Testing a pixel against all four edges at
        once would broadcast a row band against a column band and build the full ``(n, h, w)``
        result three times over -- gigabytes of temporaries at the detection cap, to produce an
        answer that is already there to be written into.
    """
    if masks.shape[0] < ROUNDS_BELOW and not masks.is_cuda:
        boxes = boxes.round()

    height, width = masks.shape[1:]
    left, top, right, bottom = (edge[:, None] for edge in boxes.unbind(1))
    span_x = torch.arange(width, device=masks.device, dtype=boxes.dtype)
    span_y = torch.arange(height, device=masks.device, dtype=boxes.dtype)
    within_x = (span_x >= left) & (span_x < right)
    within_y = (span_y >= top) & (span_y < bottom)
    return masks.mul_(within_x[:, None, :]).mul_(within_y[:, :, None])


def unpad(masks: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Cut the letterbox border off *masks* and resize what is left to *shape*.

    The border is derived from the two shapes rather than from the letterbox that produced it, so
    this needs nothing carried forward from preprocessing -- and it is derived at *prototype*
    resolution, a quarter of the network's, which is what makes the rounding below load-bearing on
    images whose padding would look harmless at full size.
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
    return F.interpolate(masks[None, ..., top:bottom, left:right].float(), shape, mode="bilinear")[0]


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

    # One mask per detection: its coefficients weight the shared prototype stack, which is
    # flattened so the whole combination is a single matrix multiply and then folded back to the
    # prototype grid.
    masks = (coefficients @ protos.reshape(channels, -1)).unflatten(1, (mask_h, mask_w))
    # Thresholded before the crop rather than after. ``crop`` only ever writes zero and zero fails
    # ``> 0``, so the two orders give bit-identical masks -- but this one hands ``crop`` a boolean
    # an eighth of the size and lets the full-resolution float go before the crop instead of after
    # it. At the detection cap of 300, on a 1281x1920 photograph, that is 2.9 GB alive against
    # 0.7.
    #
    # ``gt`` rather than ``gt_(0.0).bool()``: thresholding in place on a float tensor writes 1.0
    # and 0.0 across the whole 2.9 GB, and the cast then reads all of it back to produce the
    # 0.7 GB answer the plain comparison writes in one pass. Bit-identical, four times quicker,
    # and it leaves :func:`unpad`'s early-return case unmutated. Upstream casts to ``byte`` and
    # mozo to ``bool``; the comparison is the same one, and a mask is a yes-or-no per pixel.
    return crop(unpad(masks, shape).gt(0.0), boxes)
