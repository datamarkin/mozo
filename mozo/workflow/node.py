"""What a node is: a function, read for its signature.

A node has a name, some inputs, some parameters and one output. All four are already stated by an
ordinary Python function -- its name, its annotations, its defaults and its docstring -- so this
module reads them off rather than asking for them again. A node author writes one function and
declares one thing that a signature cannot express, its category.

That is the whole design, and it exists because the alternative was measured. The implementation
this replaces carried a class per node, sixty lines of port declarations around a single call, and
a second entry in a nine-hundred-line metadata table for the same node's name, description and
parameter defaults. Two places, no link between them, and the catalogue the editor reads was the
second one -- so a parameter could be renamed in the code and stay right in the UI, which is the
failure ``tools/fetch/_ultralytics.py`` argues against in its own docstring.

Here the catalogue is derived. There is nowhere for it to disagree with what runs.

The rule that splits inputs from parameters is one line: **an annotation that names a port type is
an input; anything else is a parameter.** Inputs are wired from other nodes and carry pixels or
detections. Parameters are typed in by hand and carry numbers, strings and choices. The annotation
already knows which is which::

    @node(category="Annotate")
    def draw_boxes(image: Image, detections: Detections, thickness: int = 2) -> Image:
        '''Draw bounding boxes around detected objects.'''
        return pf.annotate.box(image, detections, thickness=thickness)

Two inputs, one parameter, one output, a name and a description -- none of it written twice.
"""

from __future__ import annotations

import inspect
import types
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, NewType, Sequence

import numpy as np
import pixelflow as pf

__all__ = [
    "Classifications", "Color", "Connection", "Depth", "Detections", "Embedding", "Image",
    "NodeSpec", "Parameter", "Port", "PortType", "Source",
]


class PortType(Enum):
    """What can travel along a connection.

    Five, because mozo's fourteen families return exactly five kinds of thing. Ten of them return
    ``pf.Detections`` -- including EasyOCR, whose read text is a detection with a string, and the
    promptable segmenters, whose masks are detections with no class. CLIP and SigLIP2 return
    ``pf.Classifications``. Depth Anything V2 returns an array, and the two embedding models return
    a different kind of array.

    ``CLASSIFICATIONS`` is separate from ``DETECTIONS`` although both are PixelFlow types, because
    ``pf.Classifications`` has no boxes at all -- it offers ``top1``, ``top_k`` and
    ``filter_by_confidence``, and nothing to draw or crop. Sharing one port type would let the
    editor offer a connection that fails the moment it runs, which is a type check that lies.

    ``DEPTH`` is separate from ``IMAGE`` because a depth map is a float array with a range, not
    pixels. Flatten it to eight bits and you have quantised a measurement to 256 levels; the
    endpoints travel with it instead, so it can be read back. Looking at a depth map is a
    conversion, and a node should have to say so.

    **What it does not carry is the unit.** ``/predict`` sends ``X-Depth-Unit`` because the adapter
    knows whether a variant predicts metres, but the value on this port is the bare array, so a
    workflow cannot yet tell a metric Depth Anything variant from a relative one. Carrying it needs
    a value type rather than an array, which is a design step nothing has asked for yet -- recorded
    here rather than left as a claim this port type does not honour.
    """

    IMAGE = "image"
    DETECTIONS = "detections"
    CLASSIFICATIONS = "classifications"
    DEPTH = "depth"
    EMBEDDING = "embedding"


#: An ``HxWx3`` RGB ``uint8`` array -- :mod:`mozo.image`'s contract, unchanged. Not PIL: every
#: adapter takes and returns this, so a workflow that carried PIL images would convert twice per
#: node for nothing.
Image = NewType("Image", np.ndarray)

#: An ``HxW`` float array of depth, with the unit and endpoints the producing node reports.
Depth = NewType("Depth", np.ndarray)

#: An ``NxD`` float array of embeddings.
Embedding = NewType("Embedding", np.ndarray)

Detections = pf.Detections
Classifications = pf.Classifications

#: A parameter holding ``"#RRGGBB"``. A distinct name from ``str`` only so the editor can offer a
#: colour picker instead of a text box; it is a string everywhere else.
Color = NewType("Color", str)

#: A parameter naming where something is read from. A string everywhere else, exactly as
#: :data:`Color` is -- but a path is the one kind of value a person at a browser cannot type,
#: because the file they mean is on their machine and the path they would have to write is on the
#: server's. So the editor offers a file picker beside the box, and what it uploads arrives as this
#: parameter's value. From Python or the command line nothing changes: it is a path.
Source = NewType("Source", str)

#: Annotation -> port type. The membership test that splits inputs from parameters.
_PORTS = {
    Image: PortType.IMAGE,
    Detections: PortType.DETECTIONS,
    Classifications: PortType.CLASSIFICATIONS,
    Depth: PortType.DEPTH,
    Embedding: PortType.EMBEDDING,
}

#: Annotation -> the widget the editor should offer. ``typing.Literal`` is handled separately: it
#: is a choice, and its options are the values it names.
_KINDS = {int: "int", float: "float", str: "str", bool: "bool", Color: "color",
          Source: "source"}


@dataclass(frozen=True)
class Port:
    """One end of a connection: a name and what may travel through it."""

    name: str
    type: PortType


@dataclass(frozen=True)
class Parameter:
    """A value typed in rather than wired: its widget, its default, and its choices if any.

    *optional* marks a parameter that may be left unset, annotated ``int | None`` with a default of
    ``None``. It is a flag on a widget rather than a seventh kind of widget: a thickness is still a
    number, it just also has a state where the node picks one. Before this existed every such
    parameter smuggled that state through a value -- ``thickness=0`` for automatic -- which the
    catalogue could not express, so the editor drew a spinner in which 0 and 1 looked alike and the
    rule lived in prose. ``draw_keypoints`` had ``radius=0`` and ``thickness=0`` meaning automatic
    beside ``min_confidence=0.0`` meaning zero, in one signature.
    """

    name: str
    kind: str
    default: Any
    options: tuple = ()
    optional: bool = False


@dataclass(frozen=True)
class Connection:
    """One edge: an output port on one node feeding an input port on another."""

    source: str
    source_output: str
    target: str
    target_input: str

    def to_dict(self) -> dict:
        """Serialise to the editor's edge format."""
        return {
            "source": self.source,
            "sourceHandle": self.source_output,
            "target": self.target,
            "targetHandle": self.target_input,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Connection:
        """Read one edge. Both handles are required -- a guessed port is a silent miswiring."""
        for key in ("source", "sourceHandle", "target", "targetHandle"):
            if not data.get(key):
                raise ValueError(f"edge is missing {key!r}: {data}")
        return cls(data["source"], data["sourceHandle"], data["target"], data["targetHandle"])


@dataclass(frozen=True)
class NodeSpec:
    """Everything about one kind of node, read from the function that implements it."""

    name: str
    category: str
    description: str
    inputs: tuple
    outputs: tuple
    parameters: tuple
    run: Callable
    #: May only one item be inside this node at a time?
    #:
    #: True for a node holding a model -- a second concurrent inference doubles activation memory,
    #: and running out of memory ends a run where being slow only delays it. Also true for a node
    #: holding any single resource: one device, one connection, one open file.
    #:
    #: Declared here rather than worked out by the executor, because only the node knows. The
    #: alternative -- an executor recognising models by which module they were declared in -- is a
    #: second statement of the same fact, and it is wrong for every node declared anywhere else:
    #: a node in a user's own file would quietly widen and run several inferences at once.
    exclusive: bool = False
    #: Must this node see items one at a time, in the order they arrived?
    #:
    #: False for almost everything: a node that turns one image into another has no memory
    #: between calls, so which frame it gets first cannot matter. It is True for a node that
    #: appends to something -- a video writer holds one open stream, and the order it is called
    #: in **is** the content of the file it produces. Concurrency reorders freely (measured: 36 of
    #: 40 frames arrive out of order at a node behind a jittery stage), so such a node is wrong
    #: without saying so.
    #:
    #: Saying so costs it its parallelism -- an ordered node runs one item at a time, whatever
    #: the caller asked for -- which is the honest price and not a limitation of the engine: a
    #: single open file handle cannot be written by four threads either way.
    #: Ordering implies exclusivity: four threads taking turns through a sequence is the same
    #: one-at-a-time with more machinery. The two are separate flags because the converse does
    #: not hold -- a model is exclusive and does not care in which order frames arrive.
    ordered: bool = False

    @classmethod
    def from_function(cls, function: Callable, category: str,
                      outputs: Sequence[str] | None = None,
                      ordered: bool = False, exclusive: bool = False) -> NodeSpec:
        """Describe *function* as a node.

        Args:
            function: The implementation. Every argument must be annotated; the return may be
                annotated ``None`` for a node that only consumes, such as one that writes a file,
                or a ``tuple`` for one that produces several things at once.
            category: How the editor groups this node. The one thing a signature cannot say.
            outputs: Names for the output ports, in order. Each defaults to its port type's own
                name, which reads correctly for the common cases (``image``, ``detections``) and
                needs overriding only where that would be ambiguous or vague.
            ordered: See :attr:`ordered`. Set it on a node whose calls are a sequence rather than
                a set -- a video writer, a running total, a tracker.
            exclusive: See :attr:`exclusive`. Set it on a node that holds a model or any other
                single resource. Implied by *ordered*.

        Returns:
            The spec. Building it is the only validation a node gets, and it happens at import.
        """
        hints = typing.get_type_hints(function)
        signature = inspect.signature(function)

        inputs, parameters = [], []
        for name, argument in signature.parameters.items():
            if name not in hints:
                raise TypeError(f"{function.__name__}: argument {name!r} has no annotation")
            annotation = _stated(hints[name], argument.default)
            if annotation in _PORTS:
                if argument.default is not inspect.Parameter.empty:
                    raise TypeError(
                        f"{function.__name__}: input {name!r} has a default. An input arrives over "
                        f"a connection, so a default would be a value nothing can produce.")
                inputs.append(Port(name, _PORTS[annotation]))
            else:
                parameters.append(_parameter(function.__name__, name, annotation, argument.default))

        # ``get_type_hints`` resolves ``-> None`` to ``NoneType``, and leaves the key out entirely
        # when there is no annotation at all. Those are different claims: one says "produces
        # nothing", the other says nothing. Only the first is a node.
        if "return" not in hints:
            raise TypeError(
                f"{function.__name__}: has no return annotation. Say what it produces, or None.")

        produced = _produced(function.__name__, hints["return"], outputs)

        return cls(
            name=function.__name__,
            category=category,
            description=(inspect.getdoc(function) or "").partition("\n")[0],
            inputs=tuple(inputs),
            outputs=produced,
            parameters=tuple(parameters),
            run=function,
            ordered=ordered,
            exclusive=exclusive or ordered,
        )

    def __call__(self, **arguments) -> dict:
        """Run the node, once per item if any of its inputs arrived as a list.

        Batching lives here rather than in each node so that every node gets it, correctly, from
        one implementation -- a node is written for one image and runs over fifty without knowing.
        It belongs to the spec rather than to the graph because deciding what fans out needs only
        the ports, which is what a spec is; no graph is involved, and none is needed to test it.

        Only inputs fan out. A parameter is one value the whole batch shares, so a list-valued
        parameter stays one argument -- which is what lets a text prompt be several phrases.

        Returns:
            ``{output port name: value}``. Every port, so that a node producing several things is
            not a special case anywhere downstream. Under batching each port carries the list of
            what that port produced, rather than the batch carrying a list of bundles -- a port is
            a wire, and a wire either carries one thing or many.

        Raises:
            ValueError: If two inputs arrived as lists of different lengths. The implementation
                this replaces repeated the last item of the shorter one, which produces an answer
                for every input and is wrong for some of them without saying so.
        """
        ports = {port.name for port in self.inputs}
        batched = {name: value for name, value in arguments.items()
                   if name in ports and isinstance(value, list)}
        if not batched:
            return self._wires(self.run(**arguments))

        sizes = {name: len(value) for name, value in batched.items()}
        if len(set(sizes.values())) > 1:
            raise ValueError(f"batched inputs have different lengths: {sizes}")

        each = [self._wires(self.run(**{**arguments,
                                        **{name: value[index] for name, value in batched.items()}}))
                for index in range(next(iter(sizes.values())))]
        return {port.name: [one[port.name] for one in each] for port in self.outputs}

    def _wires(self, returned: Any) -> dict:
        """One call's return value, spread over the ports it declared."""
        if not self.outputs:
            return {}
        if len(self.outputs) == 1:
            return {self.outputs[0].name: returned}
        if not isinstance(returned, tuple) or len(returned) != len(self.outputs):
            raise TypeError(
                f"{self.name}: declares {len(self.outputs)} outputs but returned "
                f"{type(returned).__name__}")
        return {port.name: value for port, value in zip(self.outputs, returned)}

    def paired(self, returned: Any) -> tuple:
        """A returned value spread back over the ports it came from, as ``(port, value)`` pairs.

        The inverse of :meth:`result`, and here beside it so the convention -- one output is the
        value, several are a tuple in declared order -- has one home. It had two: the HTTP layer
        was re-deriving the same rule to decide which port type each half of a tuple travelled on,
        and a third output shape would have had to change both, silently mispairing types onto
        values if it changed only one.
        """
        if not self.outputs:
            return ()
        if len(self.outputs) == 1:
            return ((self.outputs[0], returned),)
        return tuple(zip(self.outputs, returned))

    def result(self, wires: dict) -> Any:
        """What a caller sees: the value the node produced, from the engine's port map.

        One output is that value; several are a tuple in declared order, so ``image, detections =
        results["rotate-1"]`` reads the way the function that produced them was written. Nothing
        wrapped, because a caller asked for a node's output, not for a description of its ports.
        """
        if not self.outputs:
            return None
        if len(self.outputs) == 1:
            return wires[self.outputs[0].name]
        return tuple(wires[port.name] for port in self.outputs)

    def port(self, name: str) -> Port:
        """The input port called *name*, or a message naming the ones that exist."""
        for candidate in self.inputs:
            if candidate.name == name:
                return candidate
        offered = [port.name for port in self.inputs] or ["nothing -- this node takes no input"]
        raise KeyError(f"{self.name} has no input {name!r}. It takes: {offered}")

    def to_dict(self) -> dict:
        """The catalogue entry the editor reads. Derived, so it cannot drift from what runs."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "inputs": [_port_dict(port) for port in self.inputs],
            "outputs": [_port_dict(port) for port in self.outputs],
            "parameters": [_parameter_dict(parameter) for parameter in self.parameters],
        }


def _produced(node: str, returns: Any, names: Sequence[str] | None) -> tuple:
    """The output ports a return annotation declares.

    ``None`` is a node that only consumes. A port type is the ordinary one-output node. A ``tuple``
    of port types is a node that produces several things at once -- which exists because PixelFlow's
    detection-aware transforms do: rotating an image and its boxes together yields both, and making
    a caller choose one would be the model being wrong rather than the design being simple.
    """
    if returns is type(None):
        declared = ()
    elif returns in _PORTS:
        declared = (returns,)
    elif typing.get_origin(returns) is tuple:
        declared = typing.get_args(returns)
    else:
        raise TypeError(
            f"{node}: returns {returns!r}, which is not a port type. A node returns something that "
            f"can travel along a connection, a tuple of those, or None.")

    unknown = [item for item in declared if item not in _PORTS]
    if unknown:
        raise TypeError(f"{node}: returns {unknown}, which cannot travel along a connection")

    chosen = list(names) if names else [_PORTS[item].value for item in declared]
    if len(chosen) != len(declared):
        raise TypeError(f"{node}: names {len(chosen)} outputs but returns {len(declared)}")
    if len(set(chosen)) != len(chosen):
        raise TypeError(f"{node}: two outputs named the same: {chosen}")

    return tuple(Port(name, _PORTS[item]) for name, item in zip(chosen, declared))


def _port_dict(port: Port) -> dict:
    """One port, as the editor reads it."""
    return {"name": port.name, "type": port.type.value}


def _parameter_dict(parameter: Parameter) -> dict:
    """One parameter, as the editor reads it. The last two keys appear only when they apply."""
    described = {"name": parameter.name, "kind": parameter.kind, "default": parameter.default}
    if parameter.options:
        described["options"] = list(parameter.options)
    if parameter.optional:
        described["optional"] = True
    return described


def _stated(annotation: Any, default: Any) -> Any:
    """Recover what an argument was annotated, undoing one rewrite :func:`get_type_hints` applies.

    Before Python 3.11, ``get_type_hints`` reads ``x: T = None`` as ``Optional[T]`` -- a rule the
    language itself dropped. Left in place it decides an argument's fate by something its author
    did not write: ``image: Image = None`` would arrive here as ``Optional[Image]``, fail the
    port-type test, and be reported as a parameter with no widget instead of as an input that may
    not have a default.

    Undone only where it rewrote a *port*. ``thickness: int | None = None`` is a parameter whose
    author did say it may be left unset, and taking that back would close the door the sentinels
    used to climb around.
    """
    if default is not None:
        return annotation
    inner = _optional(annotation)
    return inner if inner in _PORTS else annotation


def _optional(annotation: Any) -> Any:
    """The ``T`` in ``T | None``, or ``None`` where the annotation is not exactly that.

    Both spellings. ``Optional[int]`` resolves to ``typing.Union``; ``int | None`` resolves to
    ``types.UnionType``, which is a different thing with the same meaning. Checking only the first
    refused the spelling this package's own docstrings tell node authors to use.
    """
    if typing.get_origin(annotation) not in (typing.Union, types.UnionType):
        return None
    named = [item for item in typing.get_args(annotation) if item is not type(None)]
    return named[0] if len(named) == 1 else None


def _parameter(node: str, name: str, annotation: Any, default: Any) -> Parameter:
    """Read one parameter's widget, default and choices from its annotation."""
    if default is inspect.Parameter.empty:
        raise TypeError(
            f"{node}: parameter {name!r} has no default. A parameter is typed in rather than "
            f"wired, so the editor needs a value to start from.")

    inner = _optional(annotation)
    if inner is not None:
        if default is not None:
            raise TypeError(
                f"{node}: parameter {name!r} may be left unset, so its default is None, not "
                f"{default!r}. A value that means 'unset' is the thing this replaces.")
        return Parameter(name, _kind(node, name, inner), None, optional=True)

    if typing.get_origin(annotation) is typing.Literal:
        options = typing.get_args(annotation)
        if default not in options:
            raise TypeError(f"{node}: {name}={default!r} is not one of {list(options)}")
        return Parameter(name, "select", default, options)

    return Parameter(name, _kind(node, name, annotation), default)


def _kind(node: str, name: str, annotation: Any) -> str:
    """The widget for a plain annotation."""
    try:
        return _KINDS[annotation]
    except (KeyError, TypeError):
        raise TypeError(
            f"{node}: parameter {name!r} is annotated {annotation!r}, which the editor has no "
            f"widget for. Use int, float, str, bool, Color, Source, or Literal for a "
            f"choice.") from None
