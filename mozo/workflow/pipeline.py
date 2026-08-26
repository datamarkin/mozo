# SPDX-License-Identifier: Apache-2.0
"""Run a workflow as a staged pipeline: one queue and one worker set per node.

The serial engine in :mod:`mozo.workflow.graph` runs every node of one item before it starts the
next, so the slowest node is idle for everything the other nodes are doing. Here each node is a
**stage** -- a bounded inbound queue and one or more workers -- so while the model works on item 5,
decoding is already on item 6 and drawing is still finishing item 4. Throughput stops being the sum
of the node costs and becomes the cost of the slowest one.

**A parcel knows which item it belongs to and which port it feeds.** That pairing is the whole
correctness argument, and one half of it is not enough: a sequence number alone tells a two-input
node that both its values belong to item 5, but not which is the image and which is the detections.
Two branches racing means the answer differs per item, so the arguments would be assembled
correctly for some items and swapped for others -- silently, and only under load. The port name is
what makes a join deterministic, so it travels with every value.

**Pipelining costs memory that the serial engine does not spend.** ``graph.py`` drops each value
the moment its last reader has taken it; a pipeline cannot, because the point is to have several
items alive at once. An admission semaphore holds that number at ``nodes x DEPTH x workers``, so
on a five-node graph at ``workers=4`` about forty 4K frames can exist, near 1 GiB. That ceiling is
enforced rather than hoped for -- see :func:`run_pipelined` for why bounded queues alone did not
reach it -- but it is the reason ``workers`` is two by default rather than the core count.

Nothing here changes a node. :meth:`~mozo.workflow.graph.Workflow.run` and
:meth:`~mozo.workflow.graph.Workflow.stream` are untouched and single-threaded; ``run_many``
defaults to two workers and takes ``workers=1`` to go back to the plain serial loop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable, Iterator, Optional

from .graph import Event, deliver

__all__ = ["run_pipelined"]

#: Items a stage may hold waiting, per worker. Small on purpose -- see the note on memory above.
DEPTH = 2

_STOP = object()
#: The port a source node with no inputs is woken on. Stripped before the node is called, so a
#: node never sees it. Sources other than the one bound to each item still have to run once per
#: item, and they have no input to arrive on.
_TRIGGER = "\0trigger"


@dataclass(frozen=True)
class Parcel:
    """One value, travelling to one input port of one node, on behalf of one item.

    Args:
        seq: Which item. Assigned in arrival order and never reused.
        port: Which of the receiving node's inputs this fills. Carried rather than inferred
            because a node's inputs arrive from different branches at different times.
        value: The value itself, or a failed :class:`~mozo.workflow.graph.Event` if something
            upstream broke.
    """

    seq: int
    port: str
    value: Any


def _failed(value: Any) -> bool:
    """Is this a failure travelling in place of the value it would have been?"""
    return isinstance(value, Event) and value.status == "failed"


class Stage:
    """One node, its inbound queue, and the workers that drain it."""

    def __init__(self, step, ports: frozenset, width: int, settings: dict,
                 results: Queue, measure: bool = False) -> None:
        self.step = step
        #: Every argument that must arrive before this node can run. Input ports, plus the
        #: parameter the run binds each item to when this is the node the items enter by.
        self.ports = ports
        self.width = width
        #: Parameters and overrides, settled once. Only the per-item values still arrive.
        self.settings = settings
        self.results = results
        self.measure = measure
        self.queue: Queue = Queue(maxsize=DEPTH * width)
        #: ``{output port: [(stage, that stage's input port)]}``.
        self.sends: dict[str, list] = {}
        #: ``{seq: {port: value}}`` -- arguments collected so far, per item in flight.
        self.waiting: dict[int, dict] = {}
        self.lock = threading.Lock()
        #: ``{seq: arguments}`` held back until this node's turn comes, for an ordered node.
        self.held: dict[int, dict] = {}
        self.turn = 0
        self.stopping = False
        self.queue_peak = 0
        #: Nanoseconds spent inside the node itself, summed over every worker. Against the run's
        #: wall time this says whether the stage was kept fed -- which is the only way to tell a
        #: saturated bottleneck from a starved one without guessing. Only collected when a caller
        #: asked for stats: it costs a queue lock per parcel, on the same mutex producers block on.
        self.busy_ns = 0

    def offer(self, parcel: Parcel) -> None:
        """Hand *parcel* to this stage. Blocks while the queue is full, which is the backpressure."""
        self.queue.put(parcel)

    def _send(self, seq: int, outputs: dict) -> None:
        for port, value in outputs.items():
            for stage, target_port in self.sends.get(port, ()):
                stage.offer(Parcel(seq, target_port, value))

    def _fail(self, seq: int, event: Event) -> None:
        """Send *event* on to everything downstream, so no stage waits for a value never coming."""
        for targets in self.sends.values():
            for stage, target_port in targets:
                stage.offer(Parcel(seq, target_port, event))
        self.results.put((seq, None, event))

    def _work(self) -> None:
        while True:
            parcel = self.queue.get()
            # Stopping drops what is queued rather than working through it. Without this, a run
            # ended early -- a failure budget, a caller who stopped iterating -- still pays for
            # every parcel already in flight, model inference included, to produce answers that
            # nobody will read.
            if parcel is _STOP:
                return
            if self.stopping:
                continue            # drop what is queued rather than work through it
            if self.measure:
                self.queue_peak = max(self.queue_peak, self.queue.qsize())

            with self.lock:
                slots = self.waiting.setdefault(parcel.seq, {})
                slots[parcel.port] = parcel.value        # by port, never by arrival order
                if len(slots) < len(self.ports):
                    continue
                arguments = self.waiting.pop(parcel.seq)

            for seq, ready in self._due(parcel.seq, arguments):
                self._run(seq, ready)

    def _due(self, seq: int, arguments: dict) -> list:
        """Which items may run now, oldest first.

        Everything but an ordered node runs its own item straight away. An ordered node is handed
        items in sequence, so this holds one back until its turn comes. Failures take their turn
        too: skipping one would leave the counter waiting on a number that is never coming, and
        the node would stop for the rest of the run.
        """
        if not self.step.spec.ordered:
            return [(seq, arguments)]
        with self.lock:
            self.held[seq] = arguments
            due = []
            while self.turn in self.held:
                due.append((self.turn, self.held.pop(self.turn)))
                self.turn += 1
            return due

    def _run(self, seq: int, arguments: dict) -> None:
        """Call the node for one item and send what it produced onward."""
        broken = next((v for v in arguments.values() if _failed(v)), None)
        if broken is not None:
            self._fail(seq, broken)                      # already broken upstream; do not run
            return

        arguments.pop(_TRIGGER, None)
        started = time.perf_counter_ns() if self.measure else 0
        outputs, failure = self.step.call({**self.settings, **arguments})
        if self.measure:
            self.busy_ns += time.perf_counter_ns() - started
        if failure is not None:
            self._fail(seq, failure)
            return

        self.results.put((seq, self.step.id, self.step.spec.result(outputs)))
        self._send(seq, outputs)

    def start(self) -> None:
        for _ in range(self.width):
            threading.Thread(target=self._work, daemon=True,
                             name=f"stage-{self.step.id}").start()

    def stop(self) -> None:
        self.stopping = True
        for _ in range(self.width):
            self.queue.put(_STOP)


def _width(spec, workers: int, model_workers: int = 1) -> int:
    """How many items may be inside this node at once.

An **ordered** node gets one, whatever was asked for -- four threads taking turns through a
    sequence is the same one-at-a-time with more machinery. An **exclusive** node gets
    *model_workers*: it holds a model or some other single resource and says so itself, so a node
    declared in a user's own file is sized correctly rather than by where its file happens to live.

    *model_workers* defaults to one. **One is the safe answer, not the right
    one**: whether a second concurrent inference helps is a property of the model and the device
    together, and the two disagree sharply. Measured, same machine, same 1281x1920 photograph:

    ==================  =========  =========================================
    model               width 1    widened
    ==================  =========  =========================================
    ``sam3``            1.00x      1.02x at 2 -- it already fills the device
    ``yolov26/nano``    1.41x      1.70x at 4 -- it does not
    ==================  =========  =========================================

    So there is no number this function could pick that would be right on the next machine, and it
    does not try. It defaults to the value that cannot exhaust memory and lets a caller who has
    measured say otherwise -- with :func:`run_pipelined`'s ``stats`` to measure against.
    """
    if spec.ordered:
        return 1
    return max(1, model_workers if spec.exclusive else workers)


def _report(stages: dict, elapsed: float) -> dict:
    """What each stage did, so a caller can see whether the hardware was kept busy.

    ``busy_s`` is time spent inside the node, summed over that stage's workers. **``saturation`` is
    the number that answers "is this thing at 100%"**: 1.0 means every worker of that stage was
    inside the node for the whole run, so it is the limit; a low figure means the stage spent the
    run waiting for work.

    ``queue_peak`` says which side of a stage the waiting happened on. A stage that is both
    saturated and has a full queue in front of it is the bottleneck, and everything upstream of it
    is being throttled by it -- which is the one fact worth knowing before changing any width.

    There is no advice here and no tuning. The right width depends on the model and the device
    together, so this reports what happened and leaves the decision where the measurement is.
    """
    return {
        "elapsed_s": elapsed,
        "stages": {
            node_id: {
                "width": stage.width,
                "busy_s": stage.busy_ns / 1e9,
                "saturation": (min(1.0, stage.busy_ns / 1e9 / (elapsed * stage.width))
                               if elapsed else 0.0),
                "queue_peak": stage.queue_peak,
                "queue_size": stage.queue.maxsize,
            }
            for node_id, stage in stages.items()
        },
    }


def _build(workflow, settings: dict, over: str, workers: int, model_workers: int,
           results: Queue, measure: bool) -> tuple:
    """Wire one stage per node. Returns ``(stages, the stages each item is fed into)``.

    Every node with no inputs is a source and must be woken once per item, not only the one the
    items bind to -- a graph may legally have several, since ``_check_wiring`` requires only that
    declared input ports be fed and ``_sorted`` seeds on all zero-indegree nodes. The node that
    takes *over* is found the way the serial path finds it, by asking which settings entry holds
    it, rather than by assuming it is the topological root.
    """
    sources = [node for node in workflow.order if not workflow.incoming[node]]
    bound = next((node for node, values in settings.items() if over in values), sources[0])
    #: Which port each source is woken on -- the one the items bind to, or a bare trigger.
    entry = {node: (over if node == bound else _TRIGGER) for node in sources}

    stages = {}
    for node_id in workflow.order:
        step = workflow.steps[node_id]
        ports = set(workflow.incoming[node_id])
        if node_id in entry:
            ports.add(entry[node_id])
        arguments = step.arguments(settings)
        arguments.pop(over, None)                        # supplied per item, not once
        stages[node_id] = Stage(step, frozenset(ports), _width(step.spec, workers, model_workers),
                                arguments, results, measure)

    for node_id, wires in workflow.incoming.items():
        for target_port, (source_id, source_port) in wires.items():
            stages[source_id].sends.setdefault(source_port, []).append(
                (stages[node_id], target_port))

    return stages, [(stages[node], port) for node, port in entry.items()]


def run_pipelined(workflow, items, over: str, settings: dict, workers: int,
                  on_error: Optional[Callable] = None, model_workers: int = 1,
                  stats: Optional[dict] = None) -> Iterator[tuple]:
    """Run *items* through *workflow* as a staged pipeline, yielding ``(item, results)`` in order.

    Ordering is not a convenience here. Wide stages finish out of order by construction -- with two
    four-wide stages, measured, 172 of 200 items came out of place -- and ``run_many`` promises
    arrival order, so results are held by sequence number and released in sequence.

    Nothing is kept for an item that has been handed over. ``sources``, ``gathered`` and ``ready``
    are all popped as an item leaves, whether it succeeded or failed.

    That is necessary and was not sufficient. **Bounded queues alone do not bound what is held**,
    because backpressure only reaches back from a queue that is full, and the last stage never
    fills one -- it hands its answers to the caller. So one slow item let every item behind it run
    to completion and wait, finished, for its turn: measured at 712 MiB for 60 frames and 1425 MiB
    for 120, peak memory tracking the length of the source rather than the depth of the queues.
    An admission semaphore is what actually holds the line -- an item takes a permit on the way in
    and returns it on the way out, so the number alive is the number of permits whatever any stage
    is doing. It cannot deadlock: finishing the oldest item never needs a permit that a younger
    one is holding.
    """
    results: Queue = Queue()
    #: How many items may exist at once, which is the ceiling this module's header quotes.
    inflight = threading.Semaphore(len(workflow.order) * DEPTH * max(1, workers))
    measure = stats is not None
    stages, entries = _build(workflow, settings, over, workers, model_workers, results, measure)
    began = time.perf_counter()
    for stage in stages.values():
        stage.start()

    wanted = len(workflow.order)
    sources: dict = {}          # seq -> the item, until it is handed over
    gathered: dict = {}         # seq -> {node id: output}, until the item is complete
    ready: dict = {}            # seq -> results, or the failed Event, awaiting its turn
    emit = 0                    # the next sequence number to hand over
    expected: Optional[int] = None

    def feed() -> None:
        count = 0
        try:
            for index, item in enumerate(items):
                inflight.acquire()
                sources[index] = item
                for stage, port in entries:
                    stage.offer(Parcel(index, port, item))
                count = index + 1
        finally:
            results.put((None, None, count))            # the source is exhausted, and how many

    threading.Thread(target=feed, daemon=True, name="pipeline-feed").start()

    def release() -> Iterator:
        """Hand over every item whose turn has come, oldest first."""
        nonlocal emit
        while emit in ready:
            outcome, item = ready.pop(emit), sources.pop(emit)
            emit += 1
            inflight.release()
            failure = outcome if _failed(outcome) else None
            yield from deliver(item, None if failure else outcome, failure, on_error)

    try:
        while True:
            seq, node_id, value = results.get()
            if seq is None:
                expected = value
            elif seq >= emit and seq not in ready:      # anything older is already handed over
                if node_id is None:
                    ready[seq] = value                  # the failed Event
                    gathered.pop(seq, None)
                else:
                    outputs = gathered.setdefault(seq, {})
                    outputs[node_id] = value
                    if len(outputs) == wanted:
                        ready[seq] = gathered.pop(seq)
                yield from release()

            if expected is not None and emit == expected:
                break
    finally:
        for stage in stages.values():
            stage.stop()
        if stats is not None:
            stats.clear()
            stats.update(_report(stages, time.perf_counter() - began))
