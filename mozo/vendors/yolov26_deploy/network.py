# SPDX-License-Identifier: Apache-2.0
"""Assemble the detector graph from a checkpoint and run it.

The layer-to-layer wiring, the per-level strides, the class names, the image size and the
detection budget are all read from the file. What is written here is the execution of that graph
and the geometry the head leaves implicit: where a grid cell sits, and how the two-stage top-k
turns a wall of per-anchor scores into a fixed-length detection list.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import reader
from .build import Block, build


class Yolo(nn.Module):
    """The recorded graph, executed, with box decoding and end-to-end selection on top."""

    def __init__(
        self,
        layers: list[nn.Module],
        sources: list[int | list[int]],
        strides: list[float],
        names: dict[int, str],
        imgsz: int,
        max_det: int,
        end2end: bool,
    ) -> None:
        super().__init__()
        self.model = nn.ModuleList(layers)
        self.sources = sources
        # Which activations the wiring reads back later, and where each one is read for the last
        # time. Of 23 layers only 7 are ever revisited; holding even those to the end of the pass
        # leaves 30.7 MB of xlarge alive past its final use, most of it one 19.7 MB early feature
        # map that nothing touches after layer 15.
        last_read: dict[int, int] = {}
        for consumer, source in enumerate(sources):
            for index in (source if isinstance(source, list) else [source]):
                if index != -1:
                    last_read[index] = consumer
        self.keep = set(last_read)
        self.release: dict[int, list[int]] = {}
        for index, consumer in last_read.items():
            self.release.setdefault(consumer, []).append(index)
        self.strides = strides
        self.names = names
        self.imgsz = imgsz
        self.max_det = max_det
        # Read by tools/export: it decides the ``end2end`` metadata the published graph carries,
        # and this is the only place that fact is established from the checkpoint.
        self.end2end = end2end
        self.nc = layers[-1].rec("nc")
        self._anchor_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the graph on a preprocessed batch and return ``(batch, max_det, 6)``.

        Each row is ``[x1, y1, x2, y2, score, class]`` in the coordinates of the network input.
        """
        held: dict[int, Any] = {}
        current = x
        for index, (layer, source) in enumerate(zip(self.model[:-1], self.sources[:-1])):
            current = layer(self._gather(held, current, source))
            for spent in self.release.get(index, ()):
                del held[spent]
            if index in self.keep:
                held[index] = current
        feats = self._gather(held, current, self.sources[-1])
        shapes = [(int(f.shape[2]), int(f.shape[3])) for f in feats]
        boxes, scores = self._decode(self.model[-1](feats), shapes)
        return self._select(boxes, scores, min(self.max_det, sum(h * w for h, w in shapes)))

    @staticmethod
    def _gather(held: dict[int, Any], current: Any, source: int | list[int]) -> Any:
        """Resolve a layer's recorded inputs; ``-1`` means whatever the previous layer produced."""
        if isinstance(source, int):
            return current if source == -1 else held[source]
        return [current if i == -1 else held[i] for i in source]

    def _anchor_grid(self, shapes: list[tuple[int, int]], dtype: torch.dtype, device: torch.device):
        """Cell centres and their strides for every level, cached per grid geometry.

        The head records ``anchors``, ``strides`` and ``shape`` attributes, and they are the one
        thing in the file that must not be read: they are a cache from the last batch the model
        saw in training and are wrong for any other input size. Anchors depend on the input, so
        they are built here from the feature maps in hand and the strides recorded on the model.
        """
        key = (tuple(shapes), dtype, device)
        if key not in self._anchor_cache:
            centres, scales = [], []
            for (height, width), stride in zip(shapes, self.strides):
                rows, cols = torch.meshgrid(
                    torch.arange(height, dtype=dtype, device=device) + 0.5,
                    torch.arange(width, dtype=dtype, device=device) + 0.5,
                    indexing="ij",
                )
                centres.append(torch.stack((cols.flatten(), rows.flatten())))
                scales.append(torch.full((1, height * width), stride, dtype=dtype, device=device))
            self._anchor_cache[key] = (torch.cat(centres, 1), torch.cat(scales, 1))
        return self._anchor_cache[key]

    def _decode(self, raw: torch.Tensor, shapes: list[tuple[int, int]]):
        """Turn per-anchor distances and logits into pixel boxes and probabilities.

        The box branch carries one value per side, so the four numbers are the distances from the
        cell centre to each edge; there is no distribution to take an expectation over.
        """
        anchors, scales = self._anchor_grid(shapes, raw.dtype, raw.device)
        distances, logits = raw.split((4, self.nc), 1)
        to_top_left, to_bottom_right = distances.split((2, 2), 1)
        # ``cat`` returns a fresh tensor, so the scaling can be in place rather than allocating a
        # second copy of every box.
        boxes = torch.cat((anchors - to_top_left, anchors + to_bottom_right), 1).mul_(scales)
        # Logits, not probabilities. Sigmoid is monotonic, so ranking is unaffected and
        # :meth:`_select` applies it to the few hundred scores that survive instead of to all
        # 672,000 -- which is the difference between one transcendental per anchor per class and
        # one per detection returned.
        return boxes.transpose(1, 2), logits.transpose(1, 2)

    def _select(self, boxes: torch.Tensor, logits: torch.Tensor, budget: int) -> torch.Tensor:
        """Two-stage top-k: best anchors first, then the best class hits among them.

        The head is trained to fire once per object, so this replaces non-maximum suppression
        entirely — no box ever suppresses another.

        *logits* are pre-sigmoid, and the sigmoid is applied last, to the ``budget`` scores that
        are actually returned. Both top-k passes rank on a monotonic function of the same values,
        so the selection is identical either way.
        """
        anchor_rank = logits.amax(2).topk(budget, dim=1).indices.unsqueeze(-1)
        boxes = boxes.gather(1, anchor_rank.expand(-1, -1, 4))
        logits = logits.gather(1, anchor_rank.expand(-1, -1, self.nc))
        best, flat = logits.flatten(1).topk(budget, dim=1)
        chosen = boxes.gather(1, torch.div(flat, self.nc, rounding_mode="floor").unsqueeze(-1).expand(-1, -1, 4))
        labels = (flat % self.nc).to(boxes.dtype)
        return torch.cat((chosen, best.sigmoid().unsqueeze(-1), labels.unsqueeze(-1)), 2)


def recorded_tensors(node: reader.Placeholder, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten every parameter and buffer in the recorded tree into ``state_dict`` form."""
    flat = {prefix + name: array for name, array in reader.tensors(node).items()}
    for name, child in reader.children(node).items():
        flat.update(recorded_tensors(child, f"{prefix}{name}."))
    return flat


def load_recorded_weights(module: nn.Module, recorded: dict[str, np.ndarray]) -> None:
    """Move every recorded tensor into ``module``, insisting the two agree exactly."""
    target = module.state_dict()
    missing = sorted(set(target) - set(recorded))
    surplus = sorted(set(recorded) - set(target))
    if missing or surplus:
        raise ValueError(f"weight mismatch: model is missing {missing}, checkpoint has extra {surplus}")
    loaded = {}
    for key, array in recorded.items():
        wanted = target[key]
        if tuple(array.shape) != tuple(wanted.shape):
            # A counter saved as a 1-element vector is still a counter; anything else is a bug.
            if not (key.endswith("num_batches_tracked") and array.size == 1 and wanted.numel() == 1):
                raise ValueError(f"{key}: checkpoint holds shape {array.shape}, model needs {tuple(wanted.shape)}")
        loaded[key] = torch.from_numpy(array).reshape(wanted.shape).to(wanted.dtype)
    module.load_state_dict(loaded, strict=True)


def fuse_batchnorm(module: nn.Module) -> None:
    """Fold every BatchNorm into the convolution feeding it, in place.

    ``scale = gamma / sqrt(var + eps)`` rescales the kernel; the shift lands on the bias, added to
    whatever bias the convolution already had. Convolutions with no BatchNorm are left alone.
    """
    pairs = [
        (m, m.conv, m.bn)
        for m in module.modules()
        if isinstance(getattr(m, "conv", None), nn.Conv2d) and isinstance(getattr(m, "bn", None), nn.BatchNorm2d)
    ]
    for owner, conv, bn in pairs:
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bias = torch.zeros(conv.out_channels) if conv.bias is None else conv.bias.detach()
        conv.weight = nn.Parameter(conv.weight.detach() * scale.reshape(-1, 1, 1, 1))
        conv.bias = nn.Parameter(bn.bias.detach() + (bias - bn.running_mean) * scale)
        owner.bn = nn.Identity()


def _read_strides(root: reader.Placeholder, levels: int) -> list[float]:
    """Read the per-level strides recorded on the model and prove they are usable."""
    if "stride" not in root.__dict__:
        raise ValueError("checkpoint records no 'stride' on the model; cannot place anchors")
    strides = np.asarray(root.__dict__["stride"], dtype=np.float64).reshape(-1)
    if strides.size != levels:
        raise ValueError(f"checkpoint records {strides.size} strides but the head has {levels} detection levels")
    if not np.all(np.isfinite(strides)) or not np.all(strides > 0):
        raise ValueError(f"checkpoint records unusable strides {strides.tolist()}")
    return [float(s) for s in strides]


def check_imgsz(imgsz: int, strides: list[float]) -> int:
    """A network input has to land exactly on the coarsest grid, or the levels do not line up."""
    coarsest = int(max(strides))
    if imgsz <= 0 or imgsz % coarsest:
        raise ValueError(f"imgsz {imgsz} is not a positive multiple of the coarsest stride {coarsest}")
    return imgsz


def build_detector(path: str | Path, fuse: bool = True) -> Yolo:
    """Read ``path`` and return the ready-to-run graph it describes."""
    root = reader.read_model(path)
    layers = list(build(reader.children(root)["model"]).children())
    sources = [reader.attr(node, "f") for node in reader.children(reader.children(root)["model"]).values()]

    head = layers[-1]
    if not isinstance(head, Block) or head.kind != "Detect":
        raise ValueError(f"the last layer is {type(head).__name__}, not a detection head")
    end2end = bool(reader.attr(root, "yaml").get("end2end"))
    if not end2end:
        raise ValueError("this checkpoint's head is not end-to-end; it would need NMS, which this package omits")
    bins = (head.rec("no") - head.rec("nc")) // 4
    if bins != 1:
        raise ValueError(f"head regresses {bins} box bins; this package decodes single-bin distances only")

    strides = _read_strides(root, head.rec("nl"))
    args = reader.attr(root, "args")
    model = Yolo(
        layers,
        sources,
        strides,
        dict(reader.attr(root, "names")),
        check_imgsz(int(args["imgsz"]), strides),
        int(args["max_det"]),
        end2end,
    )
    load_recorded_weights(model, recorded_tensors(root))
    model.eval()
    if fuse:
        fuse_batchnorm(model)
    return model
