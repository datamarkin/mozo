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
        results = {}
        for event in self.stream(**overrides):
            if event.status == "completed":
                results[event.node] = event.output
        return results

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

            arguments = dict(step.parameters)
            arguments.update(settings.get(step_id, {}))
            for port, wire in self.incoming[step_id].items():
                arguments[port] = produced[wire]
                waiting[wire] -= 1
                if not waiting[wire]:
                    del produced[wire]

            try:
                wires = step.spec(**arguments)
            except Exception as error:
                yield Event(step_id, "failed", error=f"{step.spec.name}: {error}")
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
