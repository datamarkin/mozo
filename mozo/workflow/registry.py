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

__all__ = ["catalogue", "get", "names", "node"]

#: Node name -> spec, in registration order. Insertion order is the editor's palette order, so it
#: follows the order node modules declare their nodes in rather than an alphabetical accident.
_NODES: dict[str, NodeSpec] = {}


def node(*, category: str, outputs: Sequence[str] | None = None) -> Callable:
    """Register the decorated function as a node.

    Args:
        category: How the editor groups this node -- ``"Annotate"``, ``"Transform"``, ``"Model"``.
        outputs: Names for the output ports, in order, where the port types' own names are not
            specific enough -- two images out of one node, say.

    Returns:
        The function, unchanged. A node is an ordinary function and stays callable as one, which
        is what lets it be tested without a graph around it.
    """
    def register(function: Callable) -> Callable:
        spec = NodeSpec.from_function(function, category, outputs)
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
