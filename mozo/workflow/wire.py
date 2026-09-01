"""What a value looks like once it has left the process.

A node's output is a numpy array, a ``pixelflow`` detections table, a float matrix. None of those
travel over a socket, and turning them into something that does is its own job -- one that belongs
to the **port types**, not to the transport that happens to be carrying them. Which is the whole
reason this is a module rather than four helpers at the bottom of :mod:`mozo.workflow.api`.

**By the declaration, never by the shape of the value.** The shapes collide: a depth map and an
embedding are both two-dimensional float arrays, so guessing from the array would have sent an
embedding as a min-max normalised 16-bit PNG the moment a node produced one, and said nothing.
The port already says which is which, and a node cannot produce a value on a port it did not
declare -- so reading the declaration is not a heuristic, it is the answer.

**It owns the envelope and delegates every byte format.** An image's bytes are
:func:`mozo.image.encode_image`'s, a depth map's are :func:`mozo.depth.encode`'s, and what is added
here is only the JSON around them. So nothing in the workflow package, and nothing above it, is
where a second answer about a format could appear.

It reaches for those two and for :mod:`mozo.workflow.node`, whose port types it dispatches on. Not
the graph, not the registry, not the HTTP layer -- a second transport, a websocket or a live
preview cheaper than the real thing, is then a caller of this rather than a second opinion. Two
opinions is how one of them goes stale.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

import numpy as np

from pixelflow import transforms

from ..depth import encode as encode_depth
from ..image import as_rgb, encode_image
from .node import NodeSpec, PortType

__all__ = ["as_json", "preview", "serialise", "thumbnail"]

#: How tall a preview is. Width follows the aspect ratio.
#:
#: 360 rather than something smaller, which is the opposite of what you would guess: OpenCV sizes
#: its resize threads by the *destination* area while ``INTER_AREA``'s work is set by the source,
#: so a smaller preview is the same work on fewer threads. Measured, 1920x1281 down to 270 costs
#: 7.0 ms and down to 360 costs 5.9 ms -- bigger and cheaper. Against the canvas's 3.6 MB PNG this
#: is around a hundredth of the bytes.
#:
#: **The cost is the source's size, not this one's**, and on a 12 MP camera photograph the resize
#: is 18 ms whatever height is asked for. That belongs to ``pf.transforms.resize`` rather than
#: here; what bounds it here is :data:`PREVIEW_EVERY`.
PREVIEW_HEIGHT = 360

#: The floor between previews, in seconds. Shared by every transport that sends them, because it
#: is one measured trade-off -- what a glance is worth against what it costs -- and not a property
#: of HTTP or of a terminal.
#:
#: By the clock rather than by a count of items: every tenth item is fifty previews a second on a
#: graph that runs fast and one every twenty seconds on a graph that runs slow, where an interval
#: reads the same on both and needs no tuning per workflow.
PREVIEW_EVERY = 0.2


def serialise(spec: NodeSpec, value: Any) -> Any:
    """One node's output as JSON, by the port types *spec* declared.

    Args:
        spec: The node that produced *value*. Its ports are what say how to send it.
        value: What the node returned -- one thing, or a tuple of them in declared order.

    Returns:
        JSON-ready: one value where the node has one output, a list where it has several, and
        ``None`` where it has none, which is what a sink is.

    Which value belongs to which port is :meth:`~mozo.workflow.node.NodeSpec.paired`'s to say.
    """
    paired = spec.paired(value)
    if not paired:
        return None
    if len(paired) == 1:
        return as_json(paired[0][0].type, paired[0][1])
    return [as_json(port.type, part) for port, part in paired]


def as_json(port: PortType, value: Any) -> Any:
    """One value travelling on a port of type *port*.

    A list is a batch -- one wire carrying many -- and every item on it has the same port type.

    Raises:
        TypeError: For a port type nothing here knows how to send.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [as_json(port, item) for item in value]

    if port in (PortType.DETECTIONS, PortType.CLASSIFICATIONS):
        return value.to_dict()
    if port is PortType.IMAGE:
        return _data_uri(encode_image(value), "image/png")
    if port is PortType.DEPTH:
        png, low, high = encode_depth(value)
        # The endpoints travel with the pixels rather than in a header, because here there is no
        # header to put them in -- and a depth map without them is a picture, not a measurement.
        # Same encoding as /predict, from the same function.
        return {"depth": _data_uri(png, "image/png"), "min": low, "max": high}
    if port is PortType.EMBEDDING:
        return np.asarray(value).tolist()

    raise TypeError(f"no way to send a {port.value} as JSON")


def preview(spec: NodeSpec, value: Any) -> Optional[str]:
    """A thumbnail of what *spec* produced, or None where it produced nothing to look at.

    The cheap counterpart of :func:`serialise`, and it pairs the value the same way -- through
    :meth:`~mozo.workflow.node.NodeSpec.paired`, which is the one place that knows a single output
    is the value and several are a tuple in declared order. That rule had two homes once, the
    second being a transport deciding it again; this is the same function resisting the same drift.

    None rather than a refusal for a node with no picture: watching a detector's output is a
    reasonable thing to ask for, and not a reason to stop a run that is otherwise fine.
    """
    for port, part in spec.paired(value):
        if port.type is PortType.IMAGE and part is not None:
            # A preview is JPEG, which has no alpha to show a cut-out's transparency in, so the
            # colour channels are what gets glanced at. The result keeps the opacity; this is the
            # fiftieth of two hundred thousand frames and is discarded before the next arrives.
            #
            # Resized first and narrowed second: cv2 resizes four channels as happily as three,
            # and dropping alpha off a 360px thumbnail rather than off a 4K frame is the whole
            # difference between 0.01 ms and 37 ms of pure memcpy per preview.
            return _data_uri(
                encode_image(as_rgb(transforms.resize(part, height=PREVIEW_HEIGHT)), ".jpg"),
                "image/jpeg")
    return None


def thumbnail(image: np.ndarray, height: int = PREVIEW_HEIGHT) -> str:
    """A cheap look at *image*, for watching a run rather than reading its result.

    The same value as :func:`as_json` would send on an image port, smaller and lossy. Both live
    here so there is one place that knows how an image leaves the process; a preview encoder beside
    the transport that wanted it would be the second opinion this module exists to prevent.

    **JPEG and small, deliberately.** :func:`as_json` sends lossless PNG because an annotated
    result is thin lines and mask edges, which is what JPEG is worst at, and a smeared result is
    worse than a large response. A preview is not a result -- it is a glance at the fiftieth of
    two hundred thousand frames, discarded before the next one arrives -- so the trade goes the
    other way, and the run is what the viewer is reading, not this.
    """
    return _data_uri(encode_image(transforms.resize(image, height=height), ".jpg"), "image/jpeg")


def _data_uri(payload: bytes, media_type: str) -> str:
    """Encoded bytes as something an ``<img src>`` can take.

    The type is an argument rather than a literal so that a second encoding -- a JPEG thumbnail for
    a run in progress -- is a different value passed in here, not a second copy of this function
    beside it.
    """
    return f"data:{media_type};base64,{base64.b64encode(payload).decode()}"
