# SPDX-License-Identifier: Apache-2.0
"""Turn a recorded module tree into a live ``torch.nn`` tree.

:data:`LEAVES` says, for every leaf class the checkpoint names, how to construct the real
``torch.nn`` module from the hyperparameters the checkpoint already stores. Nothing is computed
here; padding, strides, groups and BatchNorm's ``eps`` are all read.

Composite classes become :class:`Block`, which looks its forward up in
:data:`~mozo.vendors.yolov26_deploy.flow.DATAFLOW`. Anything named by the checkpoint and in neither
table is an error, never a guess.
"""

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from . import reader
from .flow import DATAFLOW

# Bookkeeping attributes every ``nn.Module`` carries; never part of a block's configuration.
_MODULE_STATE = frozenset({"training", "np", "type", "name"})


class Block(nn.Module):
    """A composite module: its built children, the scalars it recorded, and a dataflow."""

    def __init__(self, kind: str, recorded: dict[str, Any], children: dict[str, nn.Module]) -> None:
        super().__init__()
        self.kind = kind
        self.recorded = recorded
        self.flow = DATAFLOW[kind]
        for name, child in children.items():
            self.add_module(name, child)

    def rec(self, name: str) -> Any:
        """Read a recorded scalar this block's dataflow depends on."""
        if name not in self.recorded:
            raise KeyError(f"{self.kind} block does not record required attribute {name!r}")
        return self.recorded[name]

    def forward(self, x):
        return self.flow(self, x)


def build(node: reader.Placeholder) -> nn.Module:
    """Build the ``torch.nn`` module recorded by ``node``, recursively."""
    kind = type(node).__name__
    if kind in LEAVES:
        return LEAVES[kind](node)
    kids = {name: build(child) for name, child in reader.children(node).items()}
    if kind == "Sequential":
        return nn.Sequential(*kids.values())
    if kind == "ModuleList":
        return nn.ModuleList(kids.values())
    if kind not in DATAFLOW:
        if kids:
            raise NotImplementedError(f"no dataflow implemented for composite module class {kind!r}")
        raise NotImplementedError(f"unknown leaf module class {kind!r} in checkpoint")
    recorded = {k: v for k, v in node.__dict__.items() if isinstance(v, (int, float, str)) and k not in _MODULE_STATE}
    return Block(kind, recorded, kids)


def _conv2d(node: reader.Placeholder) -> nn.Conv2d:
    return nn.Conv2d(
        reader.attr(node, "in_channels"),
        reader.attr(node, "out_channels"),
        reader.attr(node, "kernel_size"),
        stride=reader.attr(node, "stride"),
        padding=reader.attr(node, "padding"),
        dilation=reader.attr(node, "dilation"),
        groups=reader.attr(node, "groups"),
        bias="bias" in reader.tensors(node),
    )


def _batchnorm2d(node: reader.Placeholder) -> nn.BatchNorm2d:
    # eps is read, not assumed: this family trains with 1e-3, an order above torch's default.
    return nn.BatchNorm2d(reader.attr(node, "num_features"), eps=reader.attr(node, "eps"))


def _maxpool2d(node: reader.Placeholder) -> nn.MaxPool2d:
    return nn.MaxPool2d(
        reader.attr(node, "kernel_size"), stride=reader.attr(node, "stride"), padding=reader.attr(node, "padding")
    )


def _convtranspose2d(node: reader.Placeholder) -> nn.ConvTranspose2d:
    """The proto branch's upsample, which is a *learned* transposed convolution.

    Upstream's own source carries the comment ``# nn.Upsample(scale_factor=2, mode='nearest')``
    beside it, naming the substitution that looks equivalent and is not: an interpolation has no
    parameters, so swapping it in drops a 64x64x2x2 weight and a bias the strict load would then
    report as surplus. It is read like every other leaf.
    """
    return nn.ConvTranspose2d(
        reader.attr(node, "in_channels"),
        reader.attr(node, "out_channels"),
        reader.attr(node, "kernel_size"),
        stride=reader.attr(node, "stride"),
        padding=reader.attr(node, "padding"),
        output_padding=reader.attr(node, "output_padding"),
        groups=reader.attr(node, "groups"),
        dilation=reader.attr(node, "dilation"),
        bias="bias" in reader.tensors(node),
    )


def _upsample(node: reader.Placeholder) -> nn.Upsample:
    return nn.Upsample(scale_factor=reader.attr(node, "scale_factor"), mode=reader.attr(node, "mode"))


LEAVES: dict[str, Callable[[reader.Placeholder], nn.Module]] = {
    "Conv2d": _conv2d,
    "ConvTranspose2d": _convtranspose2d,
    "BatchNorm2d": _batchnorm2d,
    "MaxPool2d": _maxpool2d,
    "Upsample": _upsample,
    "SiLU": lambda node: nn.SiLU(),
    "Identity": lambda node: nn.Identity(),
}
