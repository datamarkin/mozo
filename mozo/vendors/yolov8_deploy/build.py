# SPDX-License-Identifier: Apache-2.0
"""Turn the object graph read from a checkpoint into a runnable ``torch.nn`` model.

Every construction argument is taken from what the checkpoint recorded: channel counts, kernel sizes,
paddings, batch-norm epsilons, split widths, repeat counts and the layer-to-layer wiring. Nothing is
scaled, derived or guessed here. The only thing this module supplies that the file does not contain is
the dataflow of composite blocks, which lives in :mod:`.flow`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .flow import DATAFLOW, Block
from .reader import class_name


def recorded(module: Any, name: str) -> Any:
    """Return the attribute ``name`` recorded for ``module``, or raise naming what is missing."""
    try:
        return module.__dict__[name]
    except KeyError:
        raise KeyError(f"{class_name(module)} in the checkpoint has no recorded attribute {name!r}") from None


def _children(module: Any) -> list[tuple[str, Any]]:
    """Return the recorded child modules of ``module`` in their recorded order."""
    return [(name, child) for name, child in module.__dict__.get("_modules", {}).items() if child is not None]


def _conv2d(rec: Any) -> nn.Conv2d:
    return nn.Conv2d(
        recorded(rec, "in_channels"),
        recorded(rec, "out_channels"),
        recorded(rec, "kernel_size"),
        recorded(rec, "stride"),
        recorded(rec, "padding"),
        recorded(rec, "dilation"),
        recorded(rec, "groups"),
        bias=rec.__dict__["_parameters"].get("bias") is not None,
    )


def _batchnorm2d(rec: Any) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(
        recorded(rec, "num_features"),
        eps=recorded(rec, "eps"),
        affine=recorded(rec, "affine"),
        track_running_stats=recorded(rec, "track_running_stats"),
    )


def _maxpool2d(rec: Any) -> nn.MaxPool2d:
    return nn.MaxPool2d(recorded(rec, "kernel_size"), recorded(rec, "stride"), recorded(rec, "padding"))


def _upsample(rec: Any) -> nn.Upsample:
    return nn.Upsample(scale_factor=recorded(rec, "scale_factor"), mode=recorded(rec, "mode"))


#: Recorded class name -> constructor for the equivalent ``torch.nn`` leaf module.
LEAVES: dict[str, Callable[[Any], nn.Module]] = {
    "Conv2d": _conv2d,
    "BatchNorm2d": _batchnorm2d,
    "MaxPool2d": _maxpool2d,
    "Upsample": _upsample,
    "SiLU": lambda _: nn.SiLU(),
}


class Ledger:
    """Every tensor the checkpoint records, handed out once each and audited at the end."""

    def __init__(self, root: Any):
        self.pending: dict[str, np.ndarray] = {}
        self._collect(root, "")

    def _collect(self, rec: Any, path: str) -> None:
        for group in ("_parameters", "_buffers"):
            for name, array in rec.__dict__.get(group, {}).items():
                if array is not None:
                    self.pending[f"{path}.{name}" if path else name] = array
        for name, child in _children(rec):
            self._collect(child, f"{path}.{name}" if path else name)

    def take(self, path: str, shape: torch.Size) -> torch.Tensor:
        """Consume the recorded tensor at ``path``, checking it against the shape the model needs."""
        array = self.pending.pop(path, None)
        if array is None:
            raise KeyError(f"checkpoint is missing the tensor {path!r} that the model requires")
        wanted = tuple(shape)
        if array.shape != wanted:
            # A 0-d counter and a 1-element counter mean the same thing; any other mismatch is fatal.
            if not (path.endswith("num_batches_tracked") and array.size == 1 and int(np.prod(wanted)) == 1):
                raise ValueError(f"tensor {path!r} has shape {array.shape}, the model needs {wanted}")
            array = array.reshape(wanted)
        return torch.from_numpy(array)

    def audit(self) -> None:
        """Raise if the checkpoint holds tensors no part of the model consumed."""
        if self.pending:
            raise ValueError(f"checkpoint holds {len(self.pending)} unused tensors: {sorted(self.pending)}")


def _build(rec: Any, path: str, ledger: Ledger) -> nn.Module:
    """Build the torch module mirroring the recorded module at ``path``."""
    kind = class_name(rec)
    leaf = LEAVES.get(kind)
    if leaf is not None:
        module = leaf(rec)
        module.load_state_dict(
            {name: ledger.take(f"{path}.{name}", t.shape) for name, t in module.state_dict().items()}
        )
        return module
    if kind not in ("Sequential", "ModuleList") and kind not in DATAFLOW:
        raise NotImplementedError(f"module class {kind!r} at {path!r} has no leaf builder and no dataflow")
    children = {name: _build(child, f"{path}.{name}", ledger) for name, child in _children(rec)}
    if kind == "Sequential":
        return nn.Sequential(*children.values())
    if kind == "ModuleList":
        return nn.ModuleList(children.values())
    attributes = {name: value for name, value in rec.__dict__.items() if not name.startswith("_")}
    return Block(kind, DATAFLOW[kind], children, attributes)


class Network(nn.Module):
    """The recorded layer stack, executed along the recorded wiring."""

    def __init__(self, layers: nn.ModuleList, sources: list, keep: set[int], stride: torch.Tensor, names: dict):
        super().__init__()
        self.layers = layers
        self.sources = sources
        self.keep = keep
        self.register_buffer("stride", stride)
        self.names = names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        held: dict[int, torch.Tensor] = {}
        for index, (layer, source) in enumerate(zip(self.layers, self.sources)):
            if isinstance(source, int):
                x = layer(x if source == -1 else held[source])
            else:
                x = layer([x if s == -1 else held[s] for s in source])
            if index in self.keep:
                held[index] = x
        return x


def build_network(root: Any) -> Network:
    """Build, strictly load and validate the detection network recorded under ``root``."""
    task = recorded(root, "args")["task"]
    if task != "detect":
        raise ValueError(f"checkpoint records task {task!r}; this package runs detection checkpoints only")

    ledger = Ledger(root)
    stack = dict(_children(root)).get("model")
    if stack is None or class_name(stack) != "Sequential":
        raise ValueError("checkpoint root has no 'model' Sequential of layers")
    records = _children(stack)
    layers = nn.ModuleList(_build(rec, f"model.{name}", ledger) for name, rec in records)
    ledger.audit()
    sources = [recorded(rec, "f") for _, rec in records]
    wiring = [source if isinstance(source, list) else [source] for source in sources]

    head = records[-1][1]
    levels = recorded(head, "nl")
    stride = torch.as_tensor(np.asarray(recorded(root, "stride"), dtype=np.float32).reshape(-1))
    if stride.numel() != levels:
        raise ValueError(f"checkpoint records {stride.numel()} strides for a head with {levels} levels")
    if not bool(torch.isfinite(stride).all()) or not bool((stride > 0).all()):
        raise ValueError(f"checkpoint records a non-positive or non-finite stride: {stride.tolist()}")
    if len(wiring[-1]) != levels:
        raise ValueError(f"head reads {len(wiring[-1])} inputs but declares {levels} levels")
    outputs, classes, bins = recorded(head, "no"), recorded(head, "nc"), recorded(head, "reg_max")
    if outputs != classes + 4 * bins:
        raise ValueError(f"head records no={outputs}, which is not nc={classes} plus 4 sides of {bins} bins")
    branches = (len(layers[-1].cv2), len(layers[-1].cv3))
    if branches != (levels, levels):
        raise ValueError(f"head holds {branches[0]} box and {branches[1]} class branches for {levels} levels")

    # The head decodes with the strides validated here, so it never has to re-check them.
    layers[-1].attributes["stride"] = tuple(stride.tolist())
    keep = {index for inputs in wiring for index in inputs if index != -1}
    return Network(layers, sources, keep, stride, recorded(root, "names"))


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a single convolution equivalent to ``conv`` followed by ``bn`` in inference mode."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        bias=True,
    )
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        shift = torch.zeros_like(bn.running_mean) if conv.bias is None else conv.bias
        fused.bias.copy_(bn.bias + (shift - bn.running_mean) * scale)
    return fused


def fuse(module: nn.Module) -> None:
    """Fold every batch norm that directly follows a convolution into it, throughout ``module``."""
    for child in module.children():
        fuse(child)
    if not isinstance(module, Block):
        return
    chain: list[tuple[str, nn.Module]] = []
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d) and chain and isinstance(chain[-1][1], nn.Conv2d):
            chain[-1] = (chain[-1][0], _fold(chain[-1][1], child))
        else:
            chain.append((name, child))
    module._modules.clear()
    for name, child in chain:
        module.add_module(name, child)
