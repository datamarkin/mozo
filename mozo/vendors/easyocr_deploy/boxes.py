"""From two heatmaps to a list of lines to read.

The detector emits per-pixel scores, not boxes. Turning them into boxes is three steps, and all
three are OpenCV rather than torch:

1. :func:`quads` thresholds the region and affinity maps, takes connected components of their
   union, and fits a rotated rectangle to each. One component is one *word*.
2. :func:`rescale` undoes the half-resolution heatmap and the input resize.
3. :func:`group` merges words that sit on the same line, and splits the result by whether a line
   is level enough to slice out of the page or has to be warped upright.

Step 3 is the one that decides what the recogniser is given, so its five thresholds change the
text that comes back, not just the rectangles around it.
"""

from __future__ import annotations

__all__ = ["TEXT_THRESHOLD", "LINK_THRESHOLD", "LOW_TEXT", "MIN_SIZE", "SLOPE_THS",
           "YCENTER_THS", "HEIGHT_THS", "WIDTH_THS", "ADD_MARGIN", "quads", "rescale", "group"]

import math

import cv2
import numpy as np

#: A component has to peak above this on the region map to be a word at all.
TEXT_THRESHOLD = 0.7
#: Above this on the affinity map, two neighbouring characters belong to the same word.
LINK_THRESHOLD = 0.4
#: The region map is binarised at this, below :data:`TEXT_THRESHOLD`, so a word's faint edges
#: stay attached to its confident middle.
LOW_TEXT = 0.4

#: Lines whose longer side does not exceed this many pixels are dropped before recognition.
MIN_SIZE = 20

#: How far from level a line may sit before it stops being a plain slice of the page and gets a
#: perspective warp instead.
SLOPE_THS = 0.1
#: Vertical gap, as a fraction of running mean height, at which a new line starts.
YCENTER_THS = 0.5
#: Height disagreement, as a fraction of running mean height, that stops two words merging.
HEIGHT_THS = 0.5
#: Horizontal gap, as a multiple of word height, that stops two words merging.
WIDTH_THS = 0.5
#: Margin grown around every merged line, as a fraction of its shorter side.
ADD_MARGIN = 0.1

#: The heatmaps come out at half the input's resolution.
_HEATMAP_STRIDE = 2

#: Components smaller than this many pixels are noise.
_MIN_COMPONENT_AREA = 10


def quads(region: np.ndarray, affinity: np.ndarray) -> list[np.ndarray]:
    """Rotated rectangles, one per word, in heatmap coordinates.

    The union of "inside a character" and "between two characters" is what makes a word one
    connected component; the link pixels are then erased again before the box is fitted, so a
    word's box covers its characters rather than the gaps as well.
    """
    _, text_score = cv2.threshold(region, LOW_TEXT, 1, 0)
    _, link_score = cv2.threshold(affinity, LINK_THRESHOLD, 1, 0)
    combined = np.clip(text_score + link_score, 0, 1).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=4)
    height, width = region.shape
    found = []

    # Loop-invariant: which pixels are link and not text. Upstream rebuilds it per component.
    link_only = np.logical_and(link_score == 1, text_score == 0)

    for k in range(1, count):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < _MIN_COMPONENT_AREA:
            continue

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        # Dilate by an amount that scales with how solid the component is: a dense blob grows
        # more than a sparse one, so thin strokes are not swallowed by their own bounding box.
        steps = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, sy = max(x - steps, 0), max(y - steps, 0)
        ex, ey = min(x + w + steps + 1, width), min(y + h + steps + 1, height)

        # Everything below happens inside this window. It is the component's bounding box grown
        # by ``steps``, and the dilation's kernel reaches at most ``(1 + steps) // 2``, so no
        # pixel of this component can land outside it. Upstream allocates and scans a
        # full-heatmap array per component instead, which costs O(words x page) -- 1,053 ms on a
        # dense page against 25 ms here, for the identical points.
        window = (slice(sy, ey), slice(sx, ex))
        mask = labels[window] == k
        if np.max(region[window][mask]) < TEXT_THRESHOLD:
            continue

        segment = np.zeros((ey - sy, ex - sx), dtype=np.uint8)
        segment[mask] = 255
        segment[link_only[window]] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + steps, 1 + steps))
        segment = cv2.dilate(segment, kernel)

        # Row-major order is unchanged by scanning the window instead of the page, so the point
        # cloud reaching minAreaRect is identical once the window's origin is added back.
        points = np.roll(np.array(np.where(segment != 0)), 1, axis=0).transpose().reshape(-1, 2)
        points = points + np.array([sx, sy])
        box = cv2.boxPoints(cv2.minAreaRect(points))

        # A near-square rotated rectangle is ambiguous -- minAreaRect can return it rotated 45
        # degrees, as a diamond. Squaring it up to the point cloud's own extent is upstream's
        # fix and it only fires when the two side lengths agree to within a tenth.
        side_a, side_b = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        if abs(1 - max(side_a, side_b) / (min(side_a, side_b) + 1e-5)) <= 0.1:
            left, right = points[:, 0].min(), points[:, 0].max()
            top, bottom = points[:, 1].min(), points[:, 1].max()
            box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]],
                           dtype=np.float32)

        # Rotate the corner list so it starts at the top-left one and runs clockwise, which is
        # the order every consumer downstream assumes.
        box = np.roll(box, 4 - box.sum(axis=1).argmin(), 0)
        found.append(np.array(box))

    return found


def rescale(boxes: list[np.ndarray], ratio: float) -> list[np.ndarray]:
    """Heatmap coordinates back to the original image's.

    Two factors, not one: the heatmap is half the network input, and the network input was
    itself the image scaled by ``ratio``.

    The result is cast to ``int32``, truncating rather than rounding. That is upstream's cast
    and :func:`group` reads the truncated values, so it is part of where the boxes land.
    """
    scale = (1.0 / ratio) * _HEATMAP_STRIDE
    return [np.array(box * scale).astype(np.int32).reshape(-1) for box in boxes]


def _horizontal_entry(poly: np.ndarray) -> list:
    """``[x_min, x_max, y_min, y_max, y_centre, height]`` for a level word."""
    xs, ys = poly[0::2], poly[1::2]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    return [x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min]


def _free_entry(poly: np.ndarray) -> list:
    """A tilted word's four corners, pushed outward along its own axes by the margin."""
    height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
    width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])
    # 1.44 is upstream's; it is the margin fraction scaled so a diagonal push covers the same
    # ground as the axis-aligned one applied to level lines.
    margin = int(1.44 * ADD_MARGIN * min(width, height))

    theta13 = abs(np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))))
    theta24 = abs(np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))))
    dx13, dy13 = np.cos(theta13) * margin, np.sin(theta13) * margin
    dx24, dy24 = np.cos(theta24) * margin, np.sin(theta24) * margin
    return [[poly[0] - dx13, poly[1] - dy13], [poly[2] + dx24, poly[3] - dy24],
            [poly[4] + dx13, poly[5] + dy13], [poly[6] - dx24, poly[7] + dy24]]


def _with_margin(x_min: float, x_max: float, y_min: float, y_max: float) -> list:
    """Grow a box by :data:`ADD_MARGIN` of its shorter side, the same on all four edges."""
    margin = int(ADD_MARGIN * min(x_max - x_min, y_max - y_min))
    return [x_min - margin, x_max + margin, y_min - margin, y_max + margin]


def group(polys: list[np.ndarray]) -> tuple[list, list]:
    """Words to lines. Returns ``(horizontal, free)``.

    Level words are merged into lines and returned as ``[x_min, x_max, y_min, y_max]``; tilted
    ones are returned as four corners each and are never merged, because merging two quads that
    do not share an axis has no obvious answer.
    """
    horizontal, free = [], []
    for poly in polys:
        slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
        if max(abs(slope_up), abs(slope_down)) < SLOPE_THS:
            horizontal.append(_horizontal_entry(poly))
        else:
            free.append(_free_entry(poly))

    horizontal.sort(key=lambda item: item[4])

    # Break the vertical run into lines: a word whose centre has drifted more than half the
    # running mean height away from the line so far starts a new one.
    lines, current, heights, centres = [], [], [], []
    for word in horizontal:
        if current and abs(np.mean(centres) - word[4]) >= YCENTER_THS * np.mean(heights):
            lines.append(current)
            current, heights, centres = [], [], []
        current.append(word)
        heights.append(word[5])
        centres.append(word[4])
    lines.append(current)

    merged = []
    for line in lines:
        if len(line) == 1:
            box = line[0]
            # The one-word case measures its margin against the box's *height* rather than its
            # shorter side. Upstream does this and the two disagree for a short, tall word.
            margin = int(ADD_MARGIN * min(box[1] - box[0], box[5]))
            merged.append([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])
            continue

        # Left to right, merging while the words keep a comparable height and stay close enough
        # horizontally. A gap wider than WIDTH_THS times the word's height ends the run.
        line.sort(key=lambda item: item[0])
        runs, run, heights, right_edge = [], [], [], None
        for word in line:
            if run:
                near = (word[0] - right_edge) < WIDTH_THS * (word[3] - word[2])
                alike = abs(np.mean(heights) - word[5]) < HEIGHT_THS * np.mean(heights)
                if not (near and alike):
                    runs.append(run)
                    run, heights = [], []
            run.append(word)
            heights.append(word[5])
            right_edge = word[1]
        if run:
            runs.append(run)

        for words in runs:
            merged.append(_with_margin(
                min(w[0] for w in words), max(w[1] for w in words),
                min(w[2] for w in words), max(w[3] for w in words)))

    keep = [box for box in merged if max(box[1] - box[0], box[3] - box[2]) > MIN_SIZE]
    free = [quad for quad in free
            if max(_spread(quad, 0), _spread(quad, 1)) > MIN_SIZE]
    return keep, free


def _spread(quad: list, axis: int) -> float:
    values = [corner[axis] for corner in quad]
    return max(values) - min(values)
