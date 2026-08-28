"""A workflow: nodes, the connections between them, and what happens when you run it.

A workflow either is valid or does not exist. Constructing one checks that every connection names
ports that exist, that the types on both ends agree, that every input is fed exactly once, and that
the graph is acyclic. Anything wrong raises here, naming the node -- so ``/workflow/validate`` is
just construction, and a workflow that loaded will not fail for a structural reason halfway through
a run with a model already on the GPU.

Running is one loop, in :meth:`Workflow.stream`. :meth:`Workflow.run` drains it. The implementation
this replaces had two loops that had drifted apart, one of them with error handling the other
lacked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Iterator, Optional

from . import registry
from .node import Connection, Context, NodeSpec, State

__all__ = ["Event", "Workflow"]

#: A source that yielded nothing is not a source that yielded None.
_NOTHING = object()


def _forget(states: list) -> None:
    """End the run for every node that kept something, releasing what each of them opened.

    Every node is closed even where one of them raises, and the first failure is re-raised after:
    a run whose video writer was never released because an unrelated sink threw has produced a file
    that will not play, which is a worse answer than the error that caused it.
    """
    failures = []
    for state in states:
        try:
            state.close()
        except Exception as error:  # noqa: BLE001 -- this node's failure, not the other nodes'
            failures.append(error)
    if failures:
        raise failures[0]


def _closing(events: Iterator, states: list) -> Iterator:
    """*events*, with the run's state released when it ends -- or when the caller walks away."""
    try:
        yield from events
    finally:
        _forget(states)


@dataclass(frozen=True)
class Step:
    """One node in one workflow: which kind it is, and what its parameters are set to."""

    id: str
    spec: NodeSpec
    parameters: dict
    #: Where the editor drew it. Meaningless to execution, carried so that loading and saving a
    #: workflow does not silently rearrange someone's canvas.
    position: dict = field(default_factory=dict)

    def arguments(self, settings: dict) -> dict:
        """This node's saved parameters, with the run's overrides on top.

        The one statement of how a parameter is settled, because both engines settle them: the
        serial loop calls this per item and the pipeline calls it once at build time. Written twice
        it would be a rule that can differ by worker count, which is the single thing running
        concurrently is not allowed to change -- and this module has already paid for two loops
        that drifted apart once.
        """
        return {**self.parameters, **settings.get(self.id, {})}

    def call(self, arguments: dict) -> tuple:
        """Run the node, as ``(what it produced, the Event that says why it did not)``.

        The failure message is what a caller finally reads -- ``deliver`` puts it in the
        ``RuntimeError`` -- so it is built once here rather than once per engine. Written twice,
        a run's failures would read differently depending on how many workers were asked for,
        which is the single thing ``workers`` is not allowed to change.
        """
        try:
            return self.spec(**arguments), None
        except Exception as error:      # noqa: BLE001 -- one item's failure, not the run's
            return None, Event(self.id, "failed", error=f"{self.spec.name}: {error}")

    def first(self, arguments: dict) -> tuple:
        """A source's first item, as this node's output, for a run of one item.

        :meth:`Workflow.run` and :meth:`Workflow.stream` are one pass over one item, so a source
        that would yield two hundred thousand frames is asked for one and closed. That is what
        makes a workflow with a video in it something the editor can show: without it the source
        returned its generator, the generator travelled down the graph as though it were an image,
        and the failure surfaced two nodes later reading ``OpenCV(-5:Bad argument)`` -- an error
        naming the wrong node and blaming the wrong library.

        Closed in a ``finally`` rather than left to be collected, because the thing it holds is an
        open file handle on a video, and a preview must not keep one.
        """
        produced = None
        try:
            produced = self.spec.run(**arguments)
            item = next(produced, _NOTHING)
            if item is _NOTHING:
                raise ValueError("produced nothing to run on")
            return {self.spec.outputs[0].name: item}, None
        except Exception as error:      # noqa: BLE001 -- this node's failure, reported as one
            return None, Event(self.id, "failed", error=f"{self.spec.name}: {error}")
        finally:
            if produced is not None:
                produced.close()

    def to_dict(self) -> dict:
        """Serialise to the editor's node format."""
        return {
            "id": self.id,
            "type": self.spec.name,
            "position": self.position,
            "data": {"parameters": self.parameters},
        }

    @classmethod
    def from_dict(cls, data: dict) -> Step:
        """Read one node, resolving its type against the registry.

        Owns both halves of its wire format, the way :class:`~mozo.workflow.node.Connection` owns
        both halves of an edge's. Adding a field to the editor's node record is then one change
        here rather than two literals in the graph that have to be kept in step.
        """
        if "id" not in data or "type" not in data:
            raise ValueError(f"node needs an id and a type: {data}")
        return cls(
            id=data["id"],
            spec=registry.get(data["type"]),
            parameters=dict(data.get("data", {}).get("parameters", {})),
            position=dict(data.get("position", {})),
        )


@dataclass(frozen=True)
class Event:
    """One thing that happened during a run."""

    node: str
    status: str  # "running" | "completed" | "failed"
    output: Any = None
    error: Optional[str] = None


def _named(item) -> str:
    """*item* in a failure message, short enough to read.

    ``run_many`` brings paths, which name themselves. :meth:`Workflow.process` brings whatever the
    source yielded, and for a video or a folder that is the frame -- so the message used to be the
    ``repr`` of a 720x1280 array, printed once per failure, burying the reason it was raised for.
    """
    if isinstance(item, (str, bytes, Path)):
        return repr(item)
    shape = getattr(item, "shape", None)
    return f"a {shape[1]}x{shape[0]} image" if shape and len(shape) == 3 else type(item).__name__


def deliver(item, outcome, failure: Optional[Event], on_error: Optional[Any]) -> Iterator:
    """Hand one finished item to the caller, the one way both engines hand it over.

    Serial and pipelined must agree here or ``workers`` would change what a failure looks like.
    It lives beside :class:`Event` rather than in either engine, so the serial path does not have
    to import the concurrent one to give an answer back.
    """
    if failure is None:
        yield item, outcome
    elif on_error is None:
        raise RuntimeError(f"{_named(item)}: {failure.error}")
    else:
        on_error(item, failure)


class Workflow:
    """A directed acyclic graph of nodes.

    Examples:
        >>> workflow = Workflow.load("blur_faces.json")     # doctest: +SKIP
        >>> results = workflow.run(image="street.jpg")      # doctest: +SKIP
    """

    def __init__(self, steps: list, connections: list) -> None:
        self.steps = {step.id: step for step in steps}
        if len(self.steps) != len(steps):
            raise ValueError("two nodes share an id")
        self.connections = list(connections)

        # Settled at construction, because validity depends on them: checking the wiring is what
        # produces the inbound map, and computing the order is how a cycle is found. Everything
        # else about the graph -- :attr:`terminals`, :attr:`parameters` -- is a question you can
        # ask of it, and stays a property.
        #: Node id -> ``{input port: the wire feeding it}``.
        self.incoming = self._check_wiring()
        #: Execution order.
        self.order = self._sorted()
        #: How many inputs each wire feeds, so a value can be dropped once nothing else wants it.
        self.readers: dict = {}
        for wires in self.incoming.values():
            for wire in wires.values():
                self.readers[wire] = self.readers.get(wire, 0) + 1

    # --- Loading and saving ---

    @classmethod
    def load(cls, path: str | Path) -> Workflow:
        """Read a workflow from a JSON file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no workflow at {path}")
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_dict(cls, data: dict) -> Workflow:
        """Read a workflow from the editor's ``{nodes, edges}`` format."""
        return cls(
            [Step.from_dict(node) for node in data.get("nodes", [])],
            [Connection.from_dict(edge) for edge in data.get("edges", [])],
        )

    def to_dict(self) -> dict:
        """Write the workflow back out in the format it was read from."""
        return {
            "nodes": [step.to_dict() for step in self.steps.values()],
            "edges": [connection.to_dict() for connection in self.connections],
        }

    def save(self, path: str | Path) -> None:
        """Write the workflow to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # --- Shape ---

    @property
    def terminals(self) -> tuple:
        """Ids of the nodes nothing reads from -- where a run's answers end up."""
        feeding = {connection.source for connection in self.connections}
        return tuple(step_id for step_id in self.order if step_id not in feeding)

    @property
    def source(self) -> Optional[str]:
        """The node this run's items come from, or None where the caller supplies them.

        One at most, for now. Two sources is two runs interleaved -- each with its own rate, its
        own size, its own count -- and :class:`~mozo.workflow.node.Context` would have to answer
        "which one" before it could answer anything. That is the multi-camera design, and refusing
        it here says so plainly rather than picking one source's facts and calling them the run's.
        """
        found = [step_id for step_id, step in self.steps.items() if step.spec.produces_many]
        if len(found) > 1:
            raise ValueError(
                f"{len(found)} sources in one workflow: {found}. A run is one pass over one "
                f"source, so two would be two runs and the facts of neither.")
        return found[0] if found else None

    @property
    def file_parameter(self) -> str:
        """The parameter a file chosen somewhere else is the value of.

        A question about the graph's shape, like :attr:`source` and :attr:`parameters`, and asked
        here so that every way in gets the same answer. The three that need it -- the HTTP upload,
        the command line's ``--file``, and :meth:`run_many`'s default -- each used to name a
        parameter as a string literal instead, which meant each knew one node's vocabulary. Renaming
        that parameter broke them one at a time, and the command line was still broken after the
        other two were fixed.

        Derived from what the parameter *is* rather than what it is called:
        :data:`~mozo.workflow.node.Source` is the annotation for a value a person cannot type,
        which is why it puts a file picker on the node. An input node added tomorrow, calling its
        parameter anything at all, is reachable from all three without any of them changing.

        Raises:
            ValueError: If there is not exactly one. Both other answers are wrong in a way that
                would be silent -- nowhere to put the file means it was chosen for nothing, and
                two places means choosing one of them on the caller's behalf.
        """
        named = sorted({parameter.name
                        for step in self.steps.values()
                        for parameter in step.spec.parameters
                        if parameter.kind == "source"})
        if not named:
            raise ValueError(
                "no node in this workflow reads a file, so there is nothing for one to be. "
                "Add an input node, or set the value by name.")
        if len(named) > 1:
            raise ValueError(
                f"{len(named)} parameters take a file: {named}. One file cannot say which it is "
                f"-- set them by name instead.")
        return named[0]

    @property
    def parameters(self) -> dict:
        """Every parameter in the workflow, as ``name -> [node id, ...]``.

        What :meth:`run` accepts as overrides, and what it consults to reject an ambiguous one.
        """
        found: dict = {}
        for step in self.steps.values():
            for parameter in step.spec.parameters:
                found.setdefault(parameter.name, []).append(step.id)
        return found

    # --- Running ---

    def run(self, **overrides) -> dict:
        """Run the workflow and return every node's output, keyed by node id.

        Args:
            **overrides: Parameter values to use instead of the ones saved in the workflow, by
                parameter name. A name that more than one node uses is refused rather than guessed
                at, naming the nodes that have it.

        Returns:
            ``{node id: output}``. Every node, not only the terminal ones -- an intermediate result
            is often the one worth looking at, and hiding it would mean re-running to see it.
            :attr:`terminals` names the ends.

            A node that failed is absent, along with everything downstream of it. Use
            :meth:`stream` where the reason matters: it reports the failure and names the node.
        """
        settings = self._resolve(overrides)
        states = self._remember(settings)
        try:
            return self._drain(settings)[0]
        finally:
            _forget(states)

    def run_many(self, items, *, over: Optional[str] = None, on_error=None, workers: int = 2,
                 model_workers: int = 1, stats: Optional[dict] = None,
                 **overrides) -> Iterator[tuple]:
        """Run the workflow once per item, yielding ``(item, results)`` as each finishes.

        A generator rather than a list, which is the whole reason this works at any size: what a
        run holds is bounded by how many items may be alive at once, not by how many there are, so
        a directory of a million costs what a handful costs. ``items`` is consumed lazily, so it
        may be a generator itself and may never end -- though above ``workers=1`` it is read ahead
        of the results by a bounded amount, since a stage with nothing queued has nothing to
        overlap.

        The overrides are settled **once**, before the first item, because their shape does not
        change -- only the one value does. Resolving per item would rebuild :attr:`parameters` a
        million times for a million items, which measured at a tenth of the whole run.

        Args:
            items: What to run on, one at a time. Anything :meth:`run` accepts for *over* -- paths,
                bytes, arrays -- and any iterable of them.
            over: The parameter each item is bound to. Left unset it is
                :attr:`file_parameter` -- the one that takes a file, whatever the node calls it --
                so a batch of paths needs no name at all and a node named differently still works.
            on_error: Called as ``on_error(item, event)`` with the failing :class:`Event`, which
                skips that item and continues. Left unset, a failure raises and the run stops.
                A corrupt file in a million should not end a six-hour run, but silently dropping
                it is worse than stopping, so the caller says where the failure goes. Raising from
                inside the callback stops the run, whatever *workers* is.
            workers: How many items may be in flight at once. **Two by default**, which runs the
                workflow as a staged pipeline: every node gets its own queue and its own workers,
                so the model works on one item while decoding is already on the next and drawing is
                still finishing the last. See :mod:`mozo.workflow.pipeline`.

                Two rather than more because two was the peak on every real workflow measured --
                four was equal or slower, and on a graph with two model stages it fell from 1.91x
                to 1.40x. Two rather than one because one leaves the bottleneck waiting for its
                own inputs. ``workers=1`` still runs the plain serial loop, unchanged, for a
                caller who wants no threads at all.

                The threads are made per call, so ``run_many`` over a single item pays for a pool
                it cannot use -- measured at 18 us serial against 436 us at two workers on a
                three-node graph of trivial nodes. Against real node work that is noise; against a
                camera loop calling ``run_many([frame])`` per frame it is the whole cost, and such
                a caller wants one ``run_many`` over the frames instead.

                **A node holding a model is pinned to one item regardless**, because a second
                concurrent inference doubles activation memory, and running out of memory ends a
                run rather than slowing it.

                This changes how long a run takes and nothing else: same items, same results, same
                order, and failures reach *on_error* the same way.

        Yields:
            ``(item, results)`` in the order *items* arrived, where *results* is exactly what
            :meth:`run` returns. An item whose graph failed part way is not yielded at all, rather
            than yielded with the nodes that did complete: a caller iterating results would have
            no way to tell a partial answer from a whole one, and half a workflow's output is the
            kind of wrong that looks right.

        Raises:
            KeyError: On the first item, if *over* or an override names no parameter or an
                ambiguous one, or if *over* is also given as an override -- every item would then
                run on that fixed value instead of itself. A workflow that cannot be run does not
                become a million failed items.

        Examples:
            >>> for path, results in workflow.run_many(paths):   # doctest: +SKIP
            ...     save(results["annotate"], path)
        """
        over = self.file_parameter if over is None else over
        if over in overrides:
            raise KeyError(
                f"{over!r} is both the parameter each item binds to and an override, so every "
                f"item would run on {overrides[over]!r} rather than on itself. Pass one or "
                f"the other.")
        settings = self._resolve({over: None, **overrides})
        states = self._remember(settings)
        try:
            yield from self._many(items, over, settings, on_error, workers, model_workers, stats)
        finally:
            # Reached however the run ended: the last item, an exception, or a caller who stopped
            # iterating -- the case that matters most, because that is the one where a half-written
            # video would otherwise be left with no moov atom and no way to play it.
            _forget(states)

    def _many(self, items, over: Optional[str], settings: dict, on_error, workers: int,
              model_workers: int, stats: Optional[dict], driving: Optional[str] = None,
              run: Optional[Context] = None) -> Iterator[tuple]:
        """One pass over *items*, however they were come by.

        The two ways in meet here. :meth:`run_many` brings the items and binds each to a parameter
        named by *over*; :meth:`process` asks the source for them and names it in *driving*. From
        this point down there is no difference worth keeping: an item is a value that has to reach
        the graph, and the only question is which port it arrives on.
        """
        folded, folded_results = self._fold(settings, over)
        if workers > 1:
            # A staged pipeline: one queue and one worker set per node. Serial below is unchanged,
            # so asking for no workers runs exactly the code that ran before this existed. Imported
            # here rather than at module scope because ``pipeline`` imports from this module --
            # deferring the one arrow back is what keeps the pair acyclic.
            from .pipeline import run_pipelined
            yield from run_pipelined(self, items, over, settings, workers, on_error,
                                     model_workers, stats, folded, folded_results, driving, run)
            return
        if stats is not None:
            raise ValueError(
                "stats needs workers > 1: there are no stages to report on a serial run, and "
                "filling it with nothing would read as a run that did no work.")

        # The one leaf that changes per item. Everything else in ``settings`` is already final,
        # and ``_steps`` only reads it, so one dict is safe to carry across the whole run. This is
        # also why it cannot be shared by two items at once, and why the pipeline above binds the
        # value to each item's parcel instead.
        bound = next((values for values in settings.values() if over in values), None) \
            if over is not None else None
        port = self.steps[driving].spec.outputs[0].name if driving else None

        for index, item in enumerate(items):
            if bound is not None:
                bound[over] = item
            # The driving source is folded too, only per item rather than once: what it produced
            # is settled for this item exactly as a constant is settled for the run, so both reach
            # the graph by the same road and ``_steps`` does not have to know which is which.
            settled = {**folded, driving: {port: item}} if driving else folded
            reported = {**folded_results, driving: item} if driving else folded_results
            results, failure = {}, None
            for event in self._steps(settings, settled, reported,
                                     run.at(index) if run is not None else None):
                if event.status == "completed":
                    results[event.node] = event.output
                elif event.status == "failed":
                    failure = event
            yield from deliver(item, results, failure, on_error)

    def process(self, *, workers: int = 2, on_error=None, model_workers: int = 1,
                stats: Optional[dict] = None, **overrides) -> Iterator[tuple]:
        """Run once per item the source yields, yielding ``(item, results)`` as each finishes.

        A run is one pass over whatever the workflow's source produces: one image or every
        frame of a video from :func:`~mozo.workflow.nodes.io.read_media`, an unbounded stream from
        a camera. How many
        items there are stops being something the caller has to know and becomes something the
        workflow says -- which is the difference between this and :meth:`run_many`, where the
        caller brings the items and the workflow is only told what to do with each.

        The source is asked for its facts before its first item, and they are settled from then on:
        every node that asked for a :class:`~mozo.workflow.node.Context` reads the same rate, size
        and count, and reads this item's index off the one the engine already assigned.

        Args:
            workers: As :meth:`run_many` means it.
            on_error: As :meth:`run_many` means it.

        Yields:
            ``(item, results)`` in the order the source produced them.

        Raises:
            ValueError: If the workflow has no source, naming what it would need. A workflow whose
                items come from the caller is run with :meth:`run_many` instead.
        """
        source_id = self.source
        if source_id is None:
            raise ValueError(
                "this workflow has no source, so there is nothing for a run to be a pass over. "
                "Give it one -- read_media -- or bring the items yourself with "
                "run_many(items, over=...).")

        settings = self._resolve(overrides)
        states = self._remember(settings)
        try:
            step = self.steps[source_id]
            arguments = step.arguments(settings)
            # Empty: the source names the run, on the next line but one. Seeding this from the
            # first parameter guessed at which one was the name from dictionary order, and the
            # declare overwrote it anyway.
            run = Context()
            if step.spec.context:
                arguments[step.spec.context] = run
            # ``spec.run`` rather than the spec itself: a source is not called per item, so the
            # batching that ``NodeSpec.__call__`` exists for has nothing to fan out over.
            produced = step.spec.run(**arguments)
            first = next(produced, _NOTHING)
            # Settled here, after the source has had its chance to declare and before any node can
            # read: everything before this point is the source describing the run, everything
            # after is the run.
            run.seal()
            items = () if first is _NOTHING else chain((first,), produced)
            yield from self._many(items, None, settings, on_error, workers, model_workers, stats,
                                  driving=source_id, run=run)
        finally:
            _forget(states)

    def stream(self, **overrides) -> Iterator[Event]:
        """Run the workflow, reporting each node as it starts and as it finishes.

        Returns:
            An iterator of :class:`Event` -- one per node before it runs and another after, so a
            caller can show progress. A failure yields one ``"failed"`` event and stops:
            everything after it depended on the value that was not produced.

        Raises:
            KeyError: If an override names no parameter, or an ambiguous one. Raised here rather
                than on the first step, so that a caller who got a workflow back knows it will run
                -- the same reason construction is what validates the document.
        """
        settings = self._resolve(overrides)
        states = self._remember(settings)
        # Not a generator function: the overrides are refused here, before the caller has an
        # iterator, and a generator would defer that to the first ``next``. The wrapper is what
        # releases what the nodes opened -- including when the caller stops reading part way,
        # which reaches the generator as GeneratorExit and runs its ``finally`` all the same.
        return _closing(self._steps(settings), states)

    def _remember(self, settings: dict) -> list:
        """Give every node that keeps something a place to keep it, for this run and no other.

        Written into *settings* rather than handed to the executor, because that is already the
        one channel by which a node's arguments are settled once and read per item -- both engines
        go through :meth:`Step.arguments`, so neither has to learn what a state is. The pipeline
        reads it once at build time and the serial loop reads it per item, and they get the same
        object either way, which is the whole requirement.

        Returns:
            What was made, for the caller to :func:`_forget` when the run ends.
        """
        states = []
        for step_id, step in self.steps.items():
            if step.spec.state:
                state = State()
                settings.setdefault(step_id, {})[step.spec.state] = state
                states.append(state)
        return states

    def _fold(self, settings: dict, over: Optional[str] = None) -> tuple:
        """Run the nodes whose value cannot change, once, before the run begins.

        A node with no inputs and settled parameters answers the same thing every time it is
        asked. Asked per item it is asked once per item: a workflow matching every frame of a
        two-hour video against one reference image decoded that reference two hundred thousand
        times, which is not a slow run, it is the same run with a file read into it repeatedly.
        Measured on a fifty-item run before this existed: fifty calls, forty-nine of them for a
        value already held.

        Two no-input nodes are **not** constant, and both exclusions are the whole subtlety here:

        * the node the items bind to, named by *over*. Its parameter is what changes per item, so
          folding it would run the whole workflow on whichever item happened to be first.
        * a source, which is not one value but many, and drives the run rather than feeding it.

        A node that raises is left in the graph rather than folded, so that it fails where the
        engines already know how to report it -- naming the node, stopping the item, reaching
        ``on_error``. Folding its failure would turn a workflow's own error into this method's.

        Returns:
            ``({node id: {port: value}}, {node id: result})`` -- the wires for the engines to read
            instead of running the node, and the results to report as though it had run, because
            from the outside it did.
        """
        wires_by_node, results_by_node = {}, {}
        for step_id in self.order:
            step = self.steps[step_id]
            if self.incoming[step_id] or step.spec.produces_many:
                continue
            values = step.arguments(settings)
            if over is not None and over in values:
                continue
            wires, failure = step.call(values)
            if failure is not None:
                continue
            wires_by_node[step_id] = wires
            results_by_node[step_id] = step.spec.result(wires)
        return wires_by_node, results_by_node

    def _drain(self, settings: dict) -> tuple:
        """One run, as ``(what completed, the failure that stopped it or None)``.

        The one place a run is turned into an answer. :meth:`run` wants the first half and
        :meth:`run_many` wants both, which is one loop with two callers rather than the two loops
        this module's docstring describes having already had to merge once.
        """
        results, failure = {}, None
        for event in self._steps(settings):
            if event.status == "completed":
                results[event.node] = event.output
            elif event.status == "failed":
                failure = event
        return results, failure

    def _steps(self, settings: dict, folded: Optional[dict] = None,
               folded_results: Optional[dict] = None,
               context: Optional[Context] = None) -> Iterator[Event]:
        """Run each node in order, reporting as it goes.

        *folded* is what :meth:`_fold` already ran, as ``{node id: {port: value}}``. Those nodes
        are reported as completed and then skipped: a caller reads a run's results and has no
        business knowing which values were computed for it and which were computed once for every
        item, only that every node it drew produced what it produced.

        Left unset by :meth:`run` and :meth:`stream`, and set only by :meth:`run_many`. A constant
        is constant *across items*, so a run of one item has nothing to save and nothing to fold:
        folding there would buy nothing and would report the node as finishing without having been
        seen to start, which is the one thing a progress stream promises not to do.
        """
        folded = folded or {}
        folded_results = folded_results or {}
        #: ``(node id, output port) -> value``. Keyed by port because a node may produce several
        #: things, and a connection already says which one it wants.
        produced: dict = {}
        #: What is still to read each wire. A value is dropped when this reaches zero: it has been
        #: handed to every node that wanted it and yielded to the caller, so holding it only keeps
        #: an image alive for the rest of the run. On a five-node chain over one 4K photograph that
        #: is the difference between 182 MB and 70 MB of peak resident memory.
        waiting = dict(self.readers)

        for step_id, result in folded_results.items():
            yield Event(step_id, "completed", output=result)

        for step_id in self.order:
            if step_id in folded:
                continue
            step = self.steps[step_id]
            yield Event(step_id, "running")

            arguments = step.arguments(settings)

            if step.spec.produces_many:
                # A source in a one-item run. It gets an unsealed context to declare into and is
                # then asked for a single item, so the nodes after it read the real rate and size
                # of the video rather than nothing -- a preview of a workflow is still that
                # workflow, and a sink in it must open itself the way it would on the whole run.
                context = Context()
                if step.spec.context:
                    arguments[step.spec.context] = context
                wires, failure = step.first(arguments)
                context.seal()
                if failure is not None:
                    yield failure
                    return
                produced.update({(step_id, port): value for port, value in wires.items()})
                yield Event(step_id, "completed", output=step.spec.result(wires))
                continue

            if step.spec.context:
                arguments[step.spec.context] = context if context is not None else Context().seal()
            for port, wire in self.incoming[step_id].items():
                if wire[0] in folded:
                    # Held for the whole run rather than dropped when its last reader has taken
                    # it: the readers counted in ``self.readers`` are this item's, and a constant
                    # has every later item still to feed.
                    arguments[port] = folded[wire[0]][wire[1]]
                    continue
                arguments[port] = produced[wire]
                waiting[wire] -= 1
                if not waiting[wire]:
                    del produced[wire]

            wires, failure = step.call(arguments)
            if failure is not None:
                yield failure
                return

            produced.update({(step_id, port): value for port, value in wires.items()})
            yield Event(step_id, "completed", output=step.spec.result(wires))

    def _resolve(self, overrides: dict) -> dict:
        """Turn ``{parameter: value}`` into ``{node id: {parameter: value}}``, or explain why not."""
        available = self.parameters
        settings: dict = {}
        for name, value in overrides.items():
            owners = available.get(name)
            if not owners:
                raise KeyError(f"no parameter {name!r} in this workflow. It has: {sorted(available)}")
            if len(owners) > 1:
                raise KeyError(
                    f"{len(owners)} nodes have a parameter {name!r}: {owners}. Set it in the "
                    f"workflow rather than here -- an override cannot say which one you meant.")
            settings.setdefault(owners[0], {})[name] = value
        return settings

    # --- Validation ---

    def _check_wiring(self) -> dict:
        """Every connection names real ports of agreeing types, and every input is fed once.

        Returns what it had to work out to check that: which node feeds each input. Running reads
        it instead of scanning the edges again for every node.
        """
        fed: dict = {step_id: {} for step_id in self.steps}
        for connection in self.connections:
            for end in (connection.source, connection.target):
                if end not in self.steps:
                    raise ValueError(f"edge names a node that is not here: {end!r}")

            source, target = self.steps[connection.source], self.steps[connection.target]
            produced = {port.name: port for port in source.spec.outputs}
            if not produced:
                raise ValueError(f"{source.id} ({source.spec.name}) produces nothing to connect")
            if connection.source_output not in produced:
                raise ValueError(
                    f"{source.id} ({source.spec.name}) has no output "
                    f"{connection.source_output!r}. It produces {sorted(produced)}.")

            out = produced[connection.source_output]
            port = target.spec.port(connection.target_input)
            if port.type is not out.type:
                raise TypeError(
                    f"{source.id}.{out.name} is {out.type.value}, "
                    f"but {target.id}.{port.name} takes {port.type.value}")

            if port.name in fed[target.id]:
                raise ValueError(f"{target.id}.{port.name} is fed by two connections")
            fed[target.id][port.name] = (connection.source, connection.source_output)

        for step in self.steps.values():
            for port in step.spec.inputs:
                if port.name not in fed[step.id]:
                    raise ValueError(
                        f"{step.id} ({step.spec.name}) has nothing connected to its "
                        f"{port.name!r} input")
        return fed

    def _sorted(self) -> tuple:
        """Kahn's algorithm. Finding a cycle is the same work as finding the order."""
        waiting = {step_id: 0 for step_id in self.steps}
        downstream: dict = {step_id: [] for step_id in self.steps}
        for connection in self.connections:
            downstream[connection.source].append(connection.target)
            waiting[connection.target] += 1

        ready = [step_id for step_id, count in waiting.items() if count == 0]
        order = []
        while ready:
            step_id = ready.pop(0)
            order.append(step_id)
            for follower in downstream[step_id]:
                waiting[follower] -= 1
                if waiting[follower] == 0:
                    ready.append(follower)

        if len(order) != len(self.steps):
            raise ValueError(
                "the workflow has a cycle, through: "
                + ", ".join(sorted(set(self.steps) - set(order))))
        return tuple(order)
