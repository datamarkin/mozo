# SPDX-License-Identifier: Apache-2.0
"""Build a runnable torch model out of what the checkpoint records.

Leaf modules are constructed from their recorded hyper-parameters, composite modules become
a generic container whose forward is looked up by recorded class name in :mod:`~mozo.vendors.yolov12_deploy.flow`, and
the top-level graph is executed from each layer's recorded ``f`` (source) index. Nothing about
the architecture is computed here — every width, padding, epsilon and wiring decision is read.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from .flow import DATAFLOW, SCALARS, TENSORS

LEAF_KINDS = ("Conv2d", "BatchNorm2d", "SiLU", "Identity", "Upsample")


def _attr(node: Any, path: str, name: str) -> Any:
    """Read a recorded attribute, naming the module and the attribute when it is absent."""
    try:
        return node.__dict__[name]
    except KeyError:
        raise ValueError(f"{path}: the checkpoint records no {name!r} attribute for this module") from None


def _build_leaf(kind: str, node: Any, path: str) -> nn.Module:
    """Construct a torch leaf module from the hyper-parameters recorded for it."""

    def read(name: str) -> Any:
        return _attr(node, path, name)

    if kind == "Conv2d":
        return nn.Conv2d(
            read("in_channels"),
            read("out_channels"),
            read("kernel_size"),
            read("stride"),
            read("padding"),
            read("dilation"),
            read("groups"),
            bias=node.__dict__.get("_parameters", {}).get("bias") is not None,
        )
    if kind == "BatchNorm2d":
        return nn.BatchNorm2d(read("num_features"), eps=read("eps"))
    if kind == "Upsample":
        return nn.Upsample(scale_factor=read("scale_factor"), mode=read("mode"))
    if kind == "SiLU":
        return nn.SiLU()
    return nn.Identity()


class Block(nn.Module):
    """A composite module: its recorded children plus the recorded scalars its dataflow reads."""

    def __init__(self, kind: str, children: dict, scalars: dict, tensors: dict) -> None:
        """Attach the built children, the scalars this class's dataflow reads, and its own weights.

        *tensors* are parameters the block itself owns rather than delegating to a child. Only
        ``A2C2f`` has any, and only in the larger variants: a per-channel ``gamma`` that scales the
        block's residual branch. It has to be registered before the loader runs, because the loader
        refuses a checkpoint that records a tensor the model has nowhere to put -- which is exactly
        how this was found rather than silently dropped.
        """
        super().__init__()
        self.flow = DATAFLOW[kind]
        for name, child in children.items():
            self.add_module(name, child)
        for name, value in scalars.items():
            setattr(self, name, value)
        for name, shape in tensors.items():
            self.register_parameter(name, nn.Parameter(torch.empty(shape)))

    def forward(self, x: Any) -> Any:
        """Route ``x`` through the children the way this class does."""
        return self.flow(self, x)


def _load_tensors(module: nn.Module, node: Any, path: str) -> None:
    """Copy this module's own recorded tensors into it, refusing any mismatch."""
    recorded = {name: t for name, t in node.__dict__.get("_parameters", {}).items() if t is not None}
    recorded.update(node.__dict__.get("_buffers", {}))
    targets = dict(module.named_parameters(recurse=False))
    targets.update(module.named_buffers(recurse=False))

    missing = sorted(set(targets) - set(recorded))
    unused = sorted(set(recorded) - set(targets))
    if missing or unused:
        raise ValueError(f"{path}: checkpoint is missing {missing} and records unusable {unused}")

    with torch.no_grad():
        for name, array in recorded.items():
            # copy_ casts as it copies, which is what carries a half-precision checkpoint into a
            # float32 model; only shape and element count have to agree, never dtype.
            target = targets[name]
            # torch stores the batch-norm counter as a 0-d tensor; some checkpoints record it
            # with a single-element shape instead, so only its element count has to agree.
            exact = name != "num_batches_tracked"
            if (tuple(array.shape) != tuple(target.shape)) if exact else (array.size != target.numel()):
                raise ValueError(f"{path}.{name}: checkpoint shape {tuple(array.shape)} != {tuple(target.shape)}")
            target.copy_(torch.from_numpy(array).reshape(target.shape))


def _build(node: Any, path: str) -> nn.Module:
    """Mirror one recorded module (and everything under it) as a torch module."""
    kind = type(node).__name__
    children = {name: _build(child, f"{path}.{name}") for name, child in node.__dict__.get("_modules", {}).items()}
    if kind in LEAF_KINDS:
        module = _build_leaf(kind, node, path)
    elif kind == "Sequential":
        module = nn.Sequential(*children.values())
    elif kind == "ModuleList":
        module = nn.ModuleList(children.values())
    elif kind in DATAFLOW:
        # Tensors this block class may own itself. Which names are possible is declared in
        # ``flow.TENSORS``; whether one is present, and its shape, come from the file. A variant
        # that records none simply registers none, which is how ``gamma`` distinguishes the
        # larger YOLO12 variants from the smaller ones.
        recorded_tensors = node.__dict__.get("_parameters") or {}
        own = {name: tuple(recorded_tensors[name].shape)
               for name in TENSORS.get(kind, ())
               if recorded_tensors.get(name) is not None}
        module = Block(kind, children,
                       {name: _attr(node, path, name) for name in SCALARS.get(kind, ())}, own)
    else:
        raise ValueError(f"{path}: no dataflow is implemented for module class {kind!r}")
    _load_tensors(module, node, path)
    return module


class Network(nn.Module):
    """The recorded layer graph, executed from each layer's recorded source index."""

    def __init__(self, layers: list, sources: list, keep: list, names: dict) -> None:
        """Hold the built layers together with the graph recorded for them."""
        super().__init__()
        self.model = nn.ModuleList(layers)
        self.sources = sources
        self.keep = keep
        self.names = names

    @property
    def strides(self) -> tuple:
        """The detection levels' strides, coarsest last."""
        return self.model[-1].stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one image tensor through every layer, feeding each from its recorded source."""
        cached: list = [None] * len(self.model)
        for index, (layer, source) in enumerate(zip(self.model, self.sources)):
            if isinstance(source, list):
                x = [x if origin == -1 else cached[origin] for origin in source]
            elif source != -1:
                x = cached[source]
            x = layer(x)
            if self.keep[index]:
                cached[index] = x
        return x


def build_network(root: Any) -> Network:
    """Build the model recorded by ``root`` (the object :func:`~.reader.load_checkpoint` returns)."""
    end2end = _attr(root, "checkpoint root", "end2end")
    if end2end:
        raise ValueError("this checkpoint records an end-to-end head; this package implements the classic head")
    task = _attr(root, "checkpoint root", "args").get("task")
    if task is None:
        raise ValueError("checkpoint root: the recorded training arguments name no task")
    names = _attr(root, "checkpoint root", "names")

    recorded_layers = _attr(root, "checkpoint root", "_modules").get("model")
    if recorded_layers is None:
        raise ValueError("checkpoint root: the recorded module tree has no 'model' sequence of layers")
    layers, sources = [], []
    for name, node in recorded_layers.__dict__["_modules"].items():
        path = f"model.{name}"
        layers.append(_build(node, path))
        sources.append(_attr(node, path, "f"))
    # Only the outputs the checkpoint lists as reused have to survive their layer.
    reused = set(_attr(root, "checkpoint root", "save"))
    keep = [index in reused for index in range(len(layers))]

    head = layers[-1]
    head.stride = _validate_strides(head)
    if len(names) != head.nc:
        raise ValueError(f"checkpoint root: {len(names)} class names recorded for a {head.nc}-class head")

    network = Network(layers, sources, keep, names)
    network.eval()
    network.requires_grad_(False)
    return network


def _validate_strides(head: nn.Module) -> tuple:
    """Check the recorded strides against the head's detection levels and return them as floats."""
    strides = head.stride.tolist() if isinstance(head.stride, np.ndarray) else head.stride
    levels = len(head.cv2)
    if len(strides) != levels:
        raise ValueError(f"model head: {len(strides)} strides recorded for {levels} detection levels")
    for stride in strides:
        if not math.isfinite(stride) or stride <= 0:
            raise ValueError(f"model head: recorded stride {stride} is not a positive, finite number")
    return tuple(float(stride) for stride in strides)


def fuse_conv_bn(network: Network) -> None:
    """Fold every batch norm into the convolution it follows, then drop it from its block.

    Any bias the convolution already carries is folded in as well, never overwritten.
    """
    for block in list(network.modules()):
        conv, norm = block._modules.get("conv"), block._modules.get("bn")
        if not (isinstance(conv, nn.Conv2d) and isinstance(norm, nn.BatchNorm2d)):
            continue
        scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
        bias = torch.zeros_like(norm.running_mean) if conv.bias is None else conv.bias
        conv.bias = nn.Parameter(norm.bias + (bias - norm.running_mean) * scale, requires_grad=False)
        conv.weight.mul_(scale.view(-1, 1, 1, 1))
        del block._modules["bn"]
