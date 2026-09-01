"""Which nodes exist.

One dictionary and the decorator that fills it. A node registers itself at import, so the set of
nodes is the set of modules :mod:`mozo.workflow.nodes` imports -- there is no list of node names
anywhere for that set to disagree with.

The catalogue the editor reads is generated from the same dictionary. Nothing here is written down
twice, which is the point of reading nodes off their signatures in the first place.
"""

from __future__ import annotations

from typing import Callable, Sequence

from .node import NodeSpec

__all__ = ["catalogue", "get", "names", "node", "source"]

#: Node name -> spec, in registration order. Insertion order is the editor's palette order, so it
#: follows the order node modules declare their nodes in rather than an alphabetical accident.
_NODES: dict[str, NodeSpec] = {}


def source(*, category: str = "Input", outputs: Sequence[str] | None = None) -> Callable:
    """Register the decorated generator as a source: the node a run's items come from.

    A source is the one node that is not called once per item, because it is where the items come
    from. It is asked once for an iterator, and what it yields becomes the run -- one image for a
    file, two hundred thousand for a video, an unbounded stream for a camera. So a run is one pass
    over whatever the source yields, and how many items there are stops being something the caller
    has to know and starts being something the workflow says.

    It yields rather than returning a list, and that is not a style: a list of a video's frames is
    the video in memory, which is the one thing a source exists not to do.

    Where it declares :class:`~mozo.workflow.node.Context` in its signature, it should say what the
    run is -- rate, size, how many to expect -- before it yields the first item. That is the only
    moment those facts can be settled, and the only place that knows them.

    Args:
        category: How the editor groups this node. ``"Input"`` unless there is a reason.
        outputs: A name for the output port, where the port type's own name is not specific
            enough. A source has exactly one, since what it yields is the item.
    """
    return node(category=category, outputs=outputs, produces_many=True)


def node(*, category: str, outputs: Sequence[str] | None = None,
         ordered: bool = False, exclusive: bool = False,
         produces_many: bool = False, alpha: bool = False) -> Callable:
    """Register the decorated function as a node.

    Args:
        category: How the editor groups this node -- ``"Annotate"``, ``"Transform"``, ``"Model"``.
        outputs: Names for the output ports, in order, where the port types' own names are not
            specific enough -- two images out of one node, say.
        ordered: Set it where the node's calls are a sequence rather than a set -- a video writer,
            a tracker, a running total. See :attr:`~mozo.workflow.node.NodeSpec.ordered`; the node
            then runs one item at a time, in arrival order, however many workers were asked for.
        exclusive: Set it where the node holds a model, a device, or any other single resource, so
            only one item may be inside it at a time. Implied by *ordered*, and by asking for a
            :class:`~mozo.workflow.node.State`.
        produces_many: Set by :func:`source` rather than by hand. Says the node is asked once for
            an iterator whose yields are the run's items, instead of being called once per item.
        alpha: Set it where the node can take an image that carries an alpha channel. An image
            port carries three channels or four, and a node is handed three unless this says
            otherwise -- see :attr:`~mozo.workflow.node.NodeSpec.alpha`. One node sets it.

    Returns:
        The function, unchanged. A node is an ordinary function and stays callable as one, which
        is what lets it be tested without a graph around it.
    """
    def register(function: Callable) -> Callable:
        spec = NodeSpec.from_function(function, category, outputs, ordered=ordered,
                                      exclusive=exclusive, produces_many=produces_many,
                                      alpha=alpha)
        if spec.name in _NODES:
            raise ValueError(
                f"a node called {spec.name!r} is already registered, from "
                f"{_NODES[spec.name].run.__module__}")
        _NODES[spec.name] = spec
        return function
    return register


def get(name: str) -> NodeSpec:
    """The spec called *name*, or a message naming what there is."""
    try:
        return _NODES[name]
    except KeyError:
        raise KeyError(f"unknown node {name!r}. Known: {sorted(_NODES)}") from None


def names() -> tuple:
    """Every registered node name, in registration order."""
    return tuple(_NODES)


def catalogue() -> list:
    """Every node, as the editor reads it."""
    return [spec.to_dict() for spec in _NODES.values()]
