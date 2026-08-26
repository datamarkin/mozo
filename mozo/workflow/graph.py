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
from pathlib import Path
from typing import Any, Iterator, Optional

from . import registry
from .node import Connection, NodeSpec

__all__ = ["Event", "Workflow"]


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
        return self._drain(self._resolve(overrides))[0]

    def run_many(self, items, *, over: str = "image", on_error=None,
                 **overrides) -> Iterator[tuple]:
        """Run the workflow once per item, yielding ``(item, results)`` as each finishes.

        A generator rather than a list, which is the whole reason this works at any size: nothing
        holds more than one item's results at a time, so a directory of a million costs the same
        memory as one photograph. ``items`` is consumed lazily, so it may be a generator itself
        and may never end.

        The overrides are settled **once**, before the first item, because their shape does not
        change -- only the one value does. Resolving per item would rebuild :attr:`parameters` a
        million times for a million items, which measured at a tenth of the whole run.

        Runs serially, one item at a time.

        Args:
            items: What to run on, one at a time. Anything :meth:`run` accepts for *over* -- paths,
                bytes, arrays -- and any iterable of them.
            over: The parameter each item is bound to. ``"image"`` because that is what
                :func:`~mozo.workflow.nodes.io.load_image` calls its own, so a workflow reads the
                way it should.
            on_error: Called as ``on_error(item, event)`` with the failing :class:`Event`, which
                skips that item and continues. Left unset, a failure raises and the run stops.
                A corrupt file in a million should not end a six-hour run, but silently dropping
                it is worse than stopping, so the caller says where the failure goes. Raising from
                inside the callback stops the run, which is how a budget is spelled while this
                runs serially.

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
        if over in overrides:
            raise KeyError(
                f"{over!r} is both the parameter each item binds to and an override, so every "
                f"item would run on {overrides[over]!r} rather than on itself. Pass one or "
                f"the other.")
        settings = self._resolve({over: None, **overrides})
        # The one leaf that changes per item. Everything else in ``settings`` is already final,
        # and ``_steps`` only reads it, so one dict is safe to carry across the whole run.
        bound = next(values for values in settings.values() if over in values)

        for item in items:
            bound[over] = item
            results, failure = self._drain(settings)
            if failure is None:
                yield item, results
            elif on_error is None:
                raise RuntimeError(f"{item!r}: {failure.error}")
            else:
                on_error(item, failure)

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
        return self._steps(self._resolve(overrides))

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

    def _steps(self, settings: dict) -> Iterator[Event]:
        """Run each node in order, reporting as it goes."""
        #: ``(node id, output port) -> value``. Keyed by port because a node may produce several
        #: things, and a connection already says which one it wants.
        produced: dict = {}
        #: What is still to read each wire. A value is dropped when this reaches zero: it has been
        #: handed to every node that wanted it and yielded to the caller, so holding it only keeps
        #: an image alive for the rest of the run. On a five-node chain over one 4K photograph that
        #: is the difference between 182 MB and 70 MB of peak resident memory.
        waiting = dict(self.readers)

        for step_id in self.order:
            step = self.steps[step_id]
            yield Event(step_id, "running")

            arguments = step.arguments(settings)
            for port, wire in self.incoming[step_id].items():
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
