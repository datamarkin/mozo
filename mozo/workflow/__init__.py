"""The workflow runtime: run a graph of nodes over an image.

mozo runs models. This runs sequences of them, with the image handling, annotation and geometry
between them, from a graph an editor can draw and a file you can keep in version control::

    from mozo.workflow import Workflow

    Workflow.load("blur_faces.json").run(image="street.jpg")

**This subpackage depends on mozo. mozo does not depend on it.** The one exception is the router
``mozo.server`` mounts, which is the only line anywhere in ``mozo/`` that names this package.
Deleting this directory leaves a working model zoo -- ``tests/workflow/test_isolation.py`` is what
keeps that true rather than merely intended.

Everything a node author needs is exported here: :func:`node` to declare one, and the annotations
that say what travels along its ports.
"""

from __future__ import annotations

from .graph import Event, Workflow
from .node import Classifications, Color, Depth, Detections, Embedding, Image, NodeSpec, PortType
from .registry import catalogue, get, names, node

# Last, and for its side effect: a node registers itself by being declared, so importing the
# runtime has to import the nodes or ``Workflow.load`` would know no node types. It comes after
# the imports above because the nodes are declared against them.
from . import nodes  # noqa: E402,F401

#: What a node author and a caller need. ``Port``, ``Parameter`` and ``Connection`` are the
#: engine's own records -- reachable from ``mozo.workflow.node`` for anyone introspecting, but not
#: part of the surface this module documents.
__all__ = [
    "Classifications", "Color", "Depth", "Detections", "Embedding", "Event", "Image", "NodeSpec",
    "PortType", "Workflow", "catalogue", "get", "names", "node",
]
