"""Nodes to build test workflows out of.

The engine is tested against nodes that do arithmetic rather than against real ones, because what
is under test is wiring: order, types, batching, failure. A node that loads a checkpoint would make
every one of these tests slow and none of them sharper.

They are registered once, at import, and stay registered for the session -- the registry is a
module-level dictionary and a node is meant to register itself by being imported. Nothing here
un-registers anything, so a test that inspects the catalogue sees a stable vocabulary.

Deliberately not a ``conftest.py``. Without an ``__init__.py``, pytest imports every conftest by
its basename, so a second one here would occupy ``sys.modules["conftest"]`` and answer the
``from conftest import ...`` that ``tests/test_deployment.py`` and its neighbours already use.
The collision is silent and order-dependent -- it cost one confusing failure to find.
"""

from __future__ import annotations

import random
import time

import numpy as np
import pixelflow as pf

from mozo.workflow import Classifications, Depth, Detections, Embedding, Image, node

RECORD: list = []

#: What an ordered sink was handed, in the order it was handed it.
SEQUENCE: list = []

#: Every item that reached ``hesitate``, so a test can count what is alive.
STARTED: list = []


@node(category="Test")
def make(width: int = 2) -> Image:
    """Produce an image of a stated width, filled with that width."""
    RECORD.append(("make", width))
    return np.full((1, width, 3), width, dtype=np.uint8)


@node(category="Test")
def several(count: int = 2) -> Image:
    """Produce several images at once, so batching has something to fan out over."""
    return [np.full((1, index + 1, 3), index + 1, dtype=np.uint8) for index in range(count)]


@node(category="Test")
def brighten(image: Image, by: int = 1) -> Image:
    """Add a constant to every pixel."""
    RECORD.append(("brighten", int(image[0, 0, 0])))
    return image + by


@node(category="Test")
def widen(image: Image, times: int = 2) -> Image:
    """Repeat an image sideways."""
    return np.tile(image, (1, times, 1))


@node(category="Test")
def combine(left: Image, right: Image) -> Image:
    """Join two images side by side."""
    return np.concatenate([left, right], axis=1)


@node(category="Test")
def detect(image: Image) -> Detections:
    """Produce one detection per pixel column, so the count is readable."""
    count = image.shape[1]
    return pf.detections.from_arrays(
        boxes=np.array([[0.0, 0.0, 1.0, 1.0]] * count, dtype=np.float32),
        scores=np.ones(count, dtype=np.float32),
        class_ids=np.zeros(count, dtype=int),
    )


@node(category="Test")
def split(image: Image) -> tuple[Image, Detections]:
    """Produce two things at once, the way a detection-aware transform does."""
    RECORD.append(("split", image.shape[1]))
    return image + 1, detect(image)


@node(category="Test")
def fake_depth(image: Image) -> Depth:
    """Produce a depth map: a float array, which is not a picture."""
    return np.linspace(0.5, 40.0, image.shape[1], dtype=np.float32)[None, :].repeat(
        image.shape[0], 0)


@node(category="Test")
def fake_scores(image: Image) -> Classifications:
    """Produce classifications, which have no boxes at all."""
    return pf.from_scores(np.array([0.7, 0.3], np.float32), labels=["cat", "dog"])


@node(category="Test")
def fake_embedding(image: Image) -> Embedding:
    """Produce an embedding: the other two-dimensional float array, and not a depth map.

    Its whole reason for existing is that a serialiser guessing from shape would send this as a
    16-bit PNG of normalised depth and say nothing about it.
    """
    return np.arange(8, dtype=np.float32).reshape(2, 4)


@node(category="Test")
def measure(image: Image) -> None:
    """Consume an image and produce nothing, the way a node that writes a file does."""
    RECORD.append(("measure", image.shape[1]))


@node(category="Test")
def explode(image: Image) -> Image:
    """Fail, so failure has something to happen to."""
    raise RuntimeError("as promised")


def shipped(*, without: tuple = ()) -> tuple:
    """The nodes mozo ships, told apart from the test nodes above by where they were declared.

    Args:
        without: Module basenames under ``mozo.workflow.nodes`` to leave out, e.g. ``("model",)``
            for the sweeps that must run without weights.

    One statement of "a shipped node is one declared under ``mozo.workflow.nodes``", because both
    sweeps need it and a rename that narrowed one of them to nothing would leave only that sweep's
    own vacuity guard to notice.
    """
    from mozo.workflow import get, names

    excluded = {f"mozo.workflow.nodes.{name}" for name in without}
    return tuple(name for name in names()
                 if (module := get(name).run.__module__).startswith("mozo.workflow.nodes.")
                 and module not in excluded)


@node(category="Test")
def dawdle(image: Image, ms: int = 0) -> Image:
    """Pass an image through after an unpredictable pause.

    Exists so a two-branch graph actually races. A join that pairs values by the order they turned
    up rather than by the port they arrived on is correct whenever the branches happen to finish in
    declaration order, so a test whose branches are steady agrees with the bug.
    """
    time.sleep(random.uniform(0.0, ms / 1000.0))
    return image


@node(category="Test")
def explode_on(image: Image, on: int = -1) -> Image:
    """Fail for one item and pass the rest through.

    :func:`explode` fails on every item, which ends the whole run at once -- and a run that has
    ended stops its stages, so a sink on another branch never runs for a reason that has nothing
    to do with what is being tested. One failure among many keeps the run alive around it.

    The parameter is ``on`` rather than ``width`` because ``make`` already has a ``width``, and a
    run binds its items to a parameter by name: two nodes with the same one is an ambiguity
    ``run_many`` refuses rather than guesses at.
    """
    if int(image[0, 0, 0]) == on:
        raise RuntimeError(f"item {on}, as promised")
    return image


@node(category="Test")
def linger(image: Image, ms: int = 0) -> Image:
    """Pass an image through after a pause of a stated length.

    The steady counterpart to :func:`dawdle`, which pauses for a *random* fraction of its argument
    because the join tests need branches that finish in an unpredictable order. A test about what
    happens to an item after one of its branches fails needs the opposite: the other branch still
    inside this node when the failure lands, every time, or the test passes for the wrong reason
    on a fast machine.
    """
    time.sleep(ms / 1000.0)
    return image


@node(category="Test", ordered=True)
def append_ordered(image: Image) -> None:
    """A sink whose calls are a sequence, the way a video writer's are.

    Records the value it was handed. The order of this list *is* what the node produced, which is
    why the node declares itself ordered: a video file assembled out of order is not a slower
    video, it is a wrong one.
    """
    SEQUENCE.append(int(image[0, 0, 0]))


@node(category="Test")
def hesitate(image: Image, slow: int = -1) -> Image:
    """Record that this item started, in :data:`STARTED`, and dawdle on one of them.

    One slow item among fast ones is what tells an admission cap from bounded queues: without a
    cap, everything behind the straggler runs to completion and waits, finished, for its turn.
    """
    STARTED.append(int(image[0, 0, 0]))
    if int(image[0, 0, 0]) == slow:
        time.sleep(0.4)
    return image
