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
from typing import Any

import numpy as np

from ..depth import encode as encode_depth
from ..image import encode_image
from .node import NodeSpec, PortType

__all__ = ["as_json", "serialise"]


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


def _data_uri(payload: bytes, media_type: str) -> str:
    """Encoded bytes as something an ``<img src>`` can take.

    The type is an argument rather than a literal so that a second encoding -- a JPEG thumbnail for
    a run in progress -- is a different value passed in here, not a second copy of this function
    beside it.
    """
    return f"data:{media_type};base64,{base64.b64encode(payload).decode()}"
