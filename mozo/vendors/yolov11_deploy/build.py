# SPDX-License-Identifier: Apache-2.0
"""Turn the recorded module tree into a live ``torch.nn`` tree, and run it.

Every leaf is constructed from the hyperparameters the checkpoint wrote down for it: channel
counts, kernel sizes, padding, epsilon. Nothing is recomputed here, and no architecture
description is parsed. Composite blocks become :class:`Block`, which keeps the recorded class
name and looks its forward up in :mod:`~mozo.vendors.yolov11_deploy.flow`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .flow import DATAFLOW
from .reader import load_model_record


def recorded(record, name):
    """Return the attribute ``name`` the checkpoint recorded for ``record``."""
    try:
        return record.__dict__[name]
    except KeyError:
        raise KeyError(f"{type(record).__name__} record has no recorded attribute {name!r}") from None


def _tensors(record, group):
    """Return the non-empty entries of a recorded ``_parameters`` or ``_buffers`` mapping."""
    return {name: array for name, array in record.__dict__.get(group, {}).items() if array is not None}


def _convolution(record, kind, **extra):
    """Build a convolution of *kind*; padding, dilation and groups are read, never derived.

    Shared by the two convolution leaves so that the padding-mode guard and the bias detection
    have one home. Anything a particular kind reads on top of these comes in through *extra*.
    """
    mode = recorded(record, "padding_mode")
    if mode != "zeros":
        raise ValueError(f"{kind.__name__} with padding_mode {mode!r} is not supported")
    return kind(
        recorded(record, "in_channels"),
        recorded(record, "out_channels"),
        recorded(record, "kernel_size"),
        stride=recorded(record, "stride"),
        padding=recorded(record, "padding"),
        dilation=recorded(record, "dilation"),
        groups=recorded(record, "groups"),
        bias="bias" in _tensors(record, "_parameters"),
        **extra,
    )


def _conv2d(record):
    return _convolution(record, nn.Conv2d)


def _convtranspose2d(record):
    """Build the mask prototypes' upsampling, which is a *learned* transposed convolution.

    Upstream's own source carries the comment ``# nn.Upsample(scale_factor=2, mode='nearest')``
    beside it, naming the substitution that looks equivalent and is not: an interpolation has no
    parameters, so swapping it in drops a 64x64x2x2 weight and a bias, which :func:`load_weights`
    would then report as unused rather than silently ignore.

    ``output_padding`` is the one hyperparameter a transposed convolution records that a forward
    one does not.
    """
    return _convolution(record, nn.ConvTranspose2d,
                        output_padding=recorded(record, "output_padding"))


def _batchnorm2d(record):
    """Build batch normalisation with the recorded epsilon, which is not the torch default."""
    return nn.BatchNorm2d(recorded(record, "num_features"), eps=recorded(record, "eps"))


def _maxpool2d(record):
    """Build max pooling from its recorded window."""
    return nn.MaxPool2d(
        recorded(record, "kernel_size"), stride=recorded(record, "stride"), padding=recorded(record, "padding")
    )


def _upsample(record):
    """Build upsampling from the recorded scale factor and interpolation mode."""
    factor = recorded(record, "scale_factor")
    if factor is None:
        raise ValueError("Upsample records no scale_factor; only scale-factor upsampling is supported")
    return nn.Upsample(scale_factor=factor, mode=recorded(record, "mode"))


LEAVES = {
    "Conv2d": _conv2d,
    "ConvTranspose2d": _convtranspose2d,
    "BatchNorm2d": _batchnorm2d,
    "MaxPool2d": _maxpool2d,
    "Upsample": _upsample,
    "SiLU": lambda record: nn.SiLU(),
    "Identity": lambda record: nn.Identity(),
}

# Attribute types worth carrying into a block's spec; tensors and submodules are handled elsewhere.
_SPEC_TYPES = (int, float, bool, str, tuple, list)

#: The heads this package serves: recorded class name -> the task the checkpoint must record for
#: it, and the attribute that head uses for its mask-coefficient count (``None`` for a head with no
#: mask branch). One row per head, so adding another is a deliberate, reviewable act rather than a
#: check getting looser. A head also needs a dataflow in :data:`~.flow.DATAFLOW`, and that table
#: refuses first -- a class nothing knows how to route tensors through cannot be built at all, so
#: this one only ever sees classes that already have one.
#:
#: The coefficient count is *looked up* rather than defaulted from a missing attribute.
#: ``spec.get("nm", 0)`` reads the absence of a name as the number zero, so a head added here that
#: recorded its coefficients under any other name would split cleanly, return the detection shape,
#: and serve boxes with no masks and no error anywhere.
HEADS = {
    "Detect": ("detect", None),
    "Segment": ("segment", "nm"),
}


class Block(nn.Module):
    """A composite module: the recorded children plus the dataflow selected by its class name."""

    def __init__(self, kind, children, spec):
        """Register the built children under their recorded names and bind the dataflow."""
        super().__init__()
        self.kind = kind
        self.spec = spec
        self.dataflow = DATAFLOW[kind]
        for name, child in children.items():
            self.add_module(name, child)

    def value(self, name):
        """Return a recorded scalar this block's dataflow needs."""
        try:
            return self.spec[name]
        except KeyError:
            raise KeyError(f"{self.kind} record has no recorded attribute {name!r}") from None

    def forward(self, x):
        """Run the dataflow registered for this block's class name."""
        return self.dataflow(self, x)

    def extra_repr(self):
        """Identify the block by the class name the checkpoint recorded."""
        return self.kind


def build_module(record):
    """Build the live module for one recorded module, recursing into its children."""
    kind = type(record).__name__
    children = record.__dict__.get("_modules", {})
    if kind in LEAVES:
        if children:
            raise ValueError(f"{kind} is built as a leaf but the checkpoint records {len(children)} children")
        return LEAVES[kind](record)
    built = {name: build_module(child) for name, child in children.items() if child is not None}
    if kind == "Sequential":
        return nn.Sequential(*built.values())
    if kind == "ModuleList":
        return nn.ModuleList(built.values())
    if kind not in DATAFLOW:
        raise NotImplementedError(f"module class {kind!r} has no dataflow and is not a known leaf")
    spec = {
        name: value
        for name, value in record.__dict__.items()
        if not name.startswith("_") and isinstance(value, _SPEC_TYPES)
    }
    return Block(kind, built, spec)


def recorded_state(record, prefix=""):
    """Flatten the recorded parameters and buffers into ``path -> array``."""
    state = {}
    for group in ("_parameters", "_buffers"):
        for name, array in _tensors(record, group).items():
            state[f"{prefix}{name}"] = array
    for name, child in record.__dict__.get("_modules", {}).items():
        if child is not None:
            state.update(recorded_state(child, f"{prefix}{name}."))
    return state


def load_weights(module, record):
    """Copy every recorded tensor into ``module``, refusing anything that does not line up.

    Shapes and element counts must match exactly. Storage precision need not: released checkpoints
    hold their weights in half precision, so each tensor is cast to the dtype the module expects.
    """
    saved = recorded_state(record)
    target = module.state_dict()
    missing = sorted(set(target) - set(saved))
    unused = sorted(set(saved) - set(target))
    if missing or unused:
        raise ValueError(f"checkpoint does not match the built model: missing {missing}, unused {unused}")
    tensors = {}
    for path, array in saved.items():
        wanted = tuple(target[path].shape)
        if tuple(array.shape) != wanted:
            # A batch counter is a scalar either way; some checkpoints store it with a leading axis.
            if not path.endswith("num_batches_tracked") or array.size != target[path].numel():
                raise ValueError(f"{path}: checkpoint holds shape {tuple(array.shape)}, model needs {wanted}")
            array = array.reshape(wanted)
        tensors[path] = torch.from_numpy(array).to(target[path].dtype)
    module.load_state_dict(tensors, strict=True)


@torch.no_grad()
def fuse_batchnorm(module):
    """Fold every ``conv``+``bn`` pair in the tree into the convolution and drop the norm.

    Convolutions that carry a bias and no norm — the head's final 1x1 layers — are left alone.
    """
    for child in module.children():
        fuse_batchnorm(child)
    conv, norm = getattr(module, "conv", None), getattr(module, "bn", None)
    if not (isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d)):
        return
    scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
    shift = norm.bias - norm.running_mean * scale
    if conv.bias is None:
        conv.bias = nn.Parameter(shift)
    else:
        conv.bias = nn.Parameter(conv.bias * scale + shift)
    conv.weight = nn.Parameter(conv.weight * scale.reshape(-1, 1, 1, 1))
    delattr(module, "bn")


class DetectionNetwork(nn.Module):
    """Runs the checkpoint's top-level layers over the graph the checkpoint recorded.

    Attributes:
        model (nn.Sequential): The built layers, named as the checkpoint named them.
        sources (tuple): Per layer, the index of the layer feeding it (``-1`` meaning the
            previous one), or a list of such indices for layers that take several inputs.
        keep (set): Indices some later layer reads back, and the only outputs held.
        strides (tuple): Pixel stride of each detection level.
        names (dict): Class index to class name.
    """

    def __init__(self, layers, sources, strides, names):
        """Assemble the network from built layers and the facts read off the checkpoint."""
        super().__init__()
        self.model = nn.Sequential(*layers)
        self.sources = sources
        # Which activations the wiring actually reads back later. Of 24 layers only 7 are ever
        # revisited, and holding the rest to the end of the forward pass costs 39 MB on nano and
        # 207 MB on xlarge for tensors nothing will look at again.
        self.keep = {index for source in sources
                     for index in (source if isinstance(source, list) else [source])
                     if index != -1}
        self.strides = strides
        self.names = names

    @torch.no_grad()
    def forward(self, x):
        """Return the head output for a preprocessed image batch."""
        held = {}
        for index, (layer, source) in enumerate(zip(self.model, self.sources)):
            if isinstance(source, list):
                x = layer([x if which == -1 else held[which] for which in source])
            else:
                x = layer(x if source == -1 else held[source])
            if index in self.keep:
                held[index] = x
        return x


def _strides_of(record, levels):
    """Read the per-level strides off the checkpoint root and check they make sense."""
    strides = recorded(record, "stride")
    if not isinstance(strides, np.ndarray):
        raise TypeError(f"recorded stride is a {type(strides).__name__}, expected an array of per-level strides")
    values = np.asarray(strides, dtype=np.float64).reshape(-1)
    if values.size != levels:
        raise ValueError(f"checkpoint records {values.size} strides for a head with {levels} detection levels")
    if not np.all(np.isfinite(values)) or not np.all(values > 0):
        raise ValueError(f"checkpoint records a non-positive or non-finite stride: {values.tolist()}")
    return tuple(float(v) for v in values)


def build_network(record):
    """Build a :class:`DetectionNetwork` from a recorded model object."""
    children = recorded(record, "_modules")["model"].__dict__["_modules"]
    layers, sources = [], []
    for position, (name, child) in enumerate(children.items()):
        if int(name) != position or recorded(child, "i") != position:
            raise ValueError(f"layer {name!r} is recorded at index {recorded(child, 'i')}, expected {position}")
        layers.append(build_module(child))
        sources.append(recorded(child, "f"))

    head_record = list(children.values())[-1]
    kind = type(head_record).__name__
    if kind not in HEADS:
        raise NotImplementedError(f"final layer is {kind!r}; this package reads {', '.join(sorted(HEADS))}")
    wanted_task, coefficient_count = HEADS[kind]
    task = recorded(record, "task")
    # The task lives on the root and the head is the last layer, so the two are recorded
    # independently and a checkpoint whose head and task disagree is one this package has
    # misread rather than one it can serve.
    if task != wanted_task:
        raise ValueError(f"checkpoint records task {task!r} for a {kind} head, which is {wanted_task!r}")
    if getattr(head_record, "end2end", False):
        raise NotImplementedError("checkpoint records an end-to-end head, which needs no NMS and is not supported")
    levels = recorded(head_record, "nl")
    classes, bins, channels = (recorded(head_record, key) for key in ("nc", "reg_max", "no"))
    if channels != 4 * bins + classes:
        raise ValueError(f"head emits {channels} channels, which is not 4*{bins} + {classes}")
    head = layers[-1]
    if len(head.cv2) != levels or len(head.cv3) != levels:
        raise ValueError(f"head records {levels} levels but has {len(head.cv2)} box and {len(head.cv3)} class branches")
    # Mask coefficients per anchor, or 0 for a head with no mask branch. Read through ``value``,
    # which raises on an attribute the checkpoint does not carry, so a head listed in ``HEADS`` as
    # having coefficients must actually record them.
    coefficients = head.value(coefficient_count) if coefficient_count else 0
    if coefficients and len(head.cv4) != levels:
        raise ValueError(f"head records {levels} levels but has {len(head.cv4)} mask-coefficient branches")
    # The strides live on the root. The head decodes with them and the detector checks imgsz
    # against them, so both the head's spec and the network carry the validated tuple.
    strides = _strides_of(record, levels)
    head.spec["strides"] = strides

    return DetectionNetwork(layers, tuple(sources), strides, recorded(record, "names"))


def load_network(weights, fuse=True):
    """Load a checkpoint from ``weights`` and return the network it describes, ready for inference."""
    record = load_model_record(weights)
    network = build_network(record)
    load_weights(network, record)
    if fuse:
        fuse_batchnorm(network)
    return network.eval()
