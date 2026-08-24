"""The workflow runtime: run a graph of nodes over an image.

mozo runs models. This runs sequences of them, with the image handling, annotation and geometry
between them, from a graph an editor can draw and a file you can keep in version control::

**This subpackage depends on mozo. mozo does not depend on it.** The one exception is the router
``mozo.server`` mounts, which is the only line anywhere in ``mozo/`` that names this package.
Deleting this directory leaves a working model zoo -- ``tests/workflow/test_isolation.py`` is what
keeps that true rather than merely intended.

Everything a node author needs is exported here: :func:`node` to declare one, and the annotations
that say what travels along its ports.
"""

from __future__ import annotations

from .node import Classifications, Color, Depth, Detections, Embedding, Image, NodeSpec, PortType
from .registry import catalogue, get, names, node

#: What a node author and a caller need. ``Port``, ``Parameter`` and ``Connection`` are the
#: engine's own records -- reachable from ``mozo.workflow.node`` for anyone introspecting, but not
#: part of the surface this module documents.
__all__ = [
    "Classifications", "Color", "Depth", "Detections", "Embedding", "Image", "NodeSpec",
    "PortType", "catalogue", "get", "names", "node",
]
