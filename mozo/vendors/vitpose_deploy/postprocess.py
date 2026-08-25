# SPDX-License-Identifier: Apache-2.0
"""Heatmaps to joint positions, in the frame's own pixels.

Replaces ``VitPoseImageProcessor.post_process_pose_estimation``. Three steps: find each channel's
peak, refine it below the cell, and map it back through the crop's affine.

**The refinement is what makes this worth doing carefully.** A heatmap cell is four crop pixels
across, and the crop is itself a downscale of the person -- so a raw argmax quantises a wrist to
something like twenty pixels of the original frame. DARK (Zhang et al., CVPR 2020) recovers the
sub-cell position by treating the blurred log-heatmap as a quadratic near its peak and stepping to
that quadratic's own maximum. It is one Newton step, and it is the difference between joints that
snap to a grid and joints that track.

**The blur is written here rather than called.** Upstream reaches for
``scipy.ndimage.gaussian_filter``; the same reasoning as in :mod:`~.image` applies, and the same
test holds the two together. One thing has to be right for that: SciPy's ``mode="reflect"``
repeats the edge sample (``b a | a b c``) where ``torch``'s reflect padding skips it
(``b a | a b c`` versus ``c b | a b c``). At sigma 0.8 the neighbour carries about a fifth of the
weight, so the difference is not a rounding error -- it is 0.58 on a heatmap whose values run to 3.
:func:`blur` pads with NumPy's ``symmetric``, which is SciPy's ``reflect``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .image import NORMALIZE_FACTOR

__all__ = ["KERNEL", "SIGMA", "blur", "peaks", "refine", "to_frame", "to_keypoints"]

#: Gaussian kernel width for the DARK modulation, as an odd number of taps. Upstream's
#: ``kernel_size`` default. Only its radius is used; the standard deviation is separate.
KERNEL = 11

#: Standard deviation of that kernel. Fixed upstream, and not derived from :data:`KERNEL` -- at
#: this sigma the outermost taps are about 3e-9, so the kernel is wider than it needs to be.
SIGMA = 0.8


def peaks(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The brightest cell of each channel, and how bright it was.

    Args:
        heatmaps: ``(N, K, H, W)``.

    Returns:
        ``(coordinates, scores)`` -- ``(N, K, 2)`` integer cell positions as ``(column, row)``,
        and ``(N, K)`` peak values. A channel whose peak is not positive is returned at ``-1``,
        which upstream uses to mean "nothing here"; :func:`to_keypoints` leaves it alone.
    """
    count, joints, _, width = heatmaps.shape
    flat = heatmaps.reshape(count, joints, -1)
    index = flat.argmax(axis=2)
    scores = flat.max(axis=2)
    coordinates = np.stack([index % width, index // width], axis=-1).astype(np.float32)
    return np.where(scores[..., None] > 0.0, coordinates, -1), scores


def blur(heatmaps: np.ndarray, kernel: int = KERNEL, sigma: float = SIGMA) -> np.ndarray:
    """Gaussian-smooth each channel, matching ``scipy.ndimage.gaussian_filter``.

    Separable, computed in float64, with the edge sample repeated -- see the module docstring for
    why that last part is not a detail.
    """
    radius = (kernel - 1) // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    taps = np.exp(-(offsets ** 2) / (2 * sigma * sigma))
    taps /= taps.sum()

    padded = np.pad(np.asarray(heatmaps, dtype=np.float64),
                    ((0, 0), (0, 0), (radius, radius), (radius, radius)), mode="symmetric")
    count, joints = padded.shape[:2]
    batch = torch.from_numpy(padded).reshape(count * joints, 1, *padded.shape[2:])
    weights = torch.from_numpy(taps)
    batch = F.conv2d(batch, weights.view(1, 1, -1, 1))
    batch = F.conv2d(batch, weights.view(1, 1, 1, -1))
    return batch.reshape(count, joints, *batch.shape[2:]).numpy()


def refine(coordinates: np.ndarray, heatmaps: np.ndarray) -> np.ndarray:
    """Move each peak to the maximum of the local quadratic. DARK's modulation step.

    Args:
        coordinates: ``(N, K, 2)`` integer cell positions from :func:`peaks`.
        heatmaps: ``(N, K, H, W)``, unblurred.

    Returns:
        ``(N, K, 2)`` refined positions, still in heatmap cells.
    """
    count, joints, height, width = heatmaps.shape
    # Back to float32 before the log, because that is the width upstream's blur returns and the
    # Newton step below divides by second differences -- where a 1e-7 disagreement does not stay
    # 1e-7. Faithfulness beats the extra digits here.
    field = np.log(np.clip(blur(heatmaps).astype(np.float32), 0.001, 50))
    padded = np.pad(field, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").reshape(-1)

    # One flat index per joint into the padded field, offset by which heatmap it belongs to.
    stride = (width + 2) * (height + 2)
    index = coordinates[..., 0] + 1 + (coordinates[..., 1] + 1) * (width + 2)
    index = index + stride * np.arange(count * joints).reshape(count, joints)
    index = index.astype(np.int64).reshape(-1, 1)

    here = padded[index]
    right, left = padded[index + 1], padded[index - 1]
    below, above = padded[index + width + 2], padded[index - width - 2]
    below_right = padded[index + width + 3]
    above_left = padded[index - width - 3]

    gradient = np.concatenate([0.5 * (right - left), 0.5 * (below - above)], axis=1)
    dxx = right - 2 * here + left
    dyy = below - 2 * here + above
    dxy = 0.5 * (below_right - right - below + 2 * here - left - above + above_left)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(count, joints, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))

    step = np.einsum("ijmn,ijnk->ijmk", hessian, gradient.reshape(count, joints, 2, 1))
    return coordinates - step.squeeze(-1)


def to_frame(coordinates: np.ndarray, center: np.ndarray, scale: np.ndarray,
             heatmap: tuple[int, int]) -> np.ndarray:
    """Map heatmap-cell positions back into the frame the box came from.

    The inverse of :func:`~.image.warp_matrix`, written as a scale and a shift because the warp
    has no rotation. ``heatmap`` is ``(rows, columns)``, and the divisor is one less than each --
    the same "unbiased" convention the forward matrix uses, for the same reason.

    Elementwise throughout, so it takes one person's ``(K, 2)`` against a ``(2,)`` centre or a
    whole batch's ``(N, K, 2)`` against ``(N, 1, 2)``.
    """
    rows, columns = heatmap
    size = scale * NORMALIZE_FACTOR
    # Indexed on the last axis, not the first: *scale* may be one person's ``(2,)`` or a batch's
    # ``(N, 1, 2)``, and ``size[0]`` means the x component in the first and the first person in
    # the second.
    per_cell = np.stack([size[..., 0] / (columns - 1.0), size[..., 1] / (rows - 1.0)], axis=-1)
    return coordinates * per_cell + center - size * 0.5


def to_keypoints(heatmaps: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Heatmaps to ``(N, K, 3)`` joints as ``(x, y, confidence)`` in the frame's pixels.

    Args:
        heatmaps: ``(N, K, H, W)`` as the model emitted them.
        centers: ``(N, 2)`` from :func:`~.image.preprocess`.
        scales: ``(N, 2)`` from :func:`~.image.preprocess`.

    Returns:
        ``(N, K, 3)`` float32. The confidence is the heatmap peak, not a probability: it is not
        calibrated and does not sum to anything. A joint the model cannot see comes back with a
        confidence near zero and a position that means nothing, so filter on it before reading a
        coordinate.
    """
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    coordinates, scores = peaks(heatmaps)
    refined = refine(coordinates, heatmaps)
    positions = to_frame(refined, centers[:, None, :], scales[:, None, :],
                         (heatmaps.shape[2], heatmaps.shape[3]))
    return np.concatenate([positions, scores[..., None]], axis=-1).astype(np.float32)
