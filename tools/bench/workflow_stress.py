#!/usr/bin/env python3
"""Stress the workflow scheduler: many graphs, many worker counts, one answer that must not move.

Bootstrap tooling; never ships. ``run_many`` claims three things at once, and only the first is a
benchmark: that widening it goes faster, that it changes *nothing else* -- same items, same
results, same order, same failures -- and that what a run holds stays bounded however long the
source is. A number without the other two is not evidence, so this measures all three on the same
runs.

    python tools/bench/workflow_stress.py --images /path/to/photos
    python tools/bench/workflow_stress.py --images photos --only geometry_chain --workers 1,2,4

Each graph runs once per worker count over the same photographs. ``workers=1`` is the serial
engine, so it is the baseline in both senses: the time everything else is a speedup over, and the
answer everything else has to reproduce exactly. A digest per node per item is what makes that
comparison possible without holding five hundred images in memory -- a mismatch names the item and
the node, which is the difference between "the pipeline is wrong" and a bug report.

The graphs are deliberately awkward. Racing branches of unequal cost that rejoin, nodes with two
inputs, nodes with two outputs, a node that fans one image out into a batch of crops, chains long
enough that the admission semaphore is what bounds them. A graph whose branches are steady agrees
with an arrival-order bug, so steady graphs prove nothing here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mozo.workflow import Workflow  # noqa: E402

def document(nodes: dict, edges: list) -> dict:
    """The editor's ``{nodes, edges}`` document, from ``{id: (type, parameters)}`` and edges."""
    return {
        "nodes": [{"id": node_id, "type": kind, "position": {"x": 0, "y": 0},
                   "data": {"parameters": parameters}}
                  for node_id, (kind, parameters) in nodes.items()],
        "edges": [{"source": source.split(":")[0], "sourceHandle": source.split(":")[1],
                   "target": target.split(":")[0], "targetHandle": target.split(":")[1]}
                  for source, target in edges],
    }


@dataclass(frozen=True)
class Case:
    """One graph to stress, and how much of the image budget it can afford."""

    name: str
    note: str
    nodes: dict
    edges: list
    #: Fraction of ``--count``. A graph whose model takes a second an image cannot have five
    #: hundred of them, and pretending otherwise means the heavy families never run at all.
    share: float = 1.0

    def build(self) -> Workflow:
        return Workflow.from_dict(document(self.nodes, self.edges))


# --- The graphs ---------------------------------------------------------------------------------

CASES = [
    Case("detect_annotate", "linear: one detector, two annotators",
         {"in": ("read_media", {}),
          "det": ("yolov11", {"variant": "nano", "threshold": 0.4}),
          "boxes": ("draw_boxes", {}),
          "labels": ("draw_labels", {})},
         [("in:image", "det:image"), ("in:image", "boxes:image"),
          ("det:detections", "boxes:detections"),
          ("boxes:image", "labels:image"), ("det:detections", "labels:detections")]),

    Case("two_detectors", "two model stages racing into one chain of joins",
         {"in": ("read_media", {}),
          "a": ("yolov8", {"variant": "nano", "threshold": 0.4}),
          "b": ("yolov26", {"variant": "nano", "threshold": 0.4}),
          "boxes": ("draw_boxes", {}),
          "labels": ("draw_labels", {})},
         [("in:image", "a:image"), ("in:image", "b:image"), ("in:image", "boxes:image"),
          ("a:detections", "boxes:detections"),
          ("boxes:image", "labels:image"), ("b:detections", "labels:detections")]),

    Case("three_way_join", "three detectors of unequal cost, joined one after another",
         {"in": ("read_media", {}),
          "a": ("yolov8", {"variant": "nano", "threshold": 0.4}),
          "b": ("yolov12", {"variant": "nano", "threshold": 0.4}),
          "c": ("rfdetr", {"variant": "nano", "threshold": 0.4}),
          "boxes": ("draw_boxes", {}),
          "labels": ("draw_labels", {}),
          "blur": ("blur_regions", {})},
         [("in:image", "a:image"), ("in:image", "b:image"), ("in:image", "c:image"),
          ("in:image", "boxes:image"), ("a:detections", "boxes:detections"),
          ("boxes:image", "labels:image"), ("b:detections", "labels:detections"),
          ("labels:image", "blur:image"), ("c:detections", "blur:detections")]),

    Case("preprocess_branches", "two preprocessing branches, a detector on each, rejoined",
         {"in": ("read_media", {}),
          "grey": ("to_grayscale", {}),
          "clahe": ("enhance_clahe", {"clip_limit": 3.0}),
          "a": ("yolov12", {"variant": "nano", "threshold": 0.4}),
          "b": ("rfdetr", {"variant": "nano", "threshold": 0.4}),
          "boxes": ("draw_boxes", {}),
          "pixelate": ("pixelate_regions", {})},
         [("in:image", "grey:image"), ("in:image", "clahe:image"),
          ("grey:image", "a:image"), ("clahe:image", "b:image"),
          ("clahe:image", "boxes:image"), ("a:detections", "boxes:detections"),
          ("boxes:image", "pixelate:image"), ("b:detections", "pixelate:detections")]),

    Case("depth_and_detect", "a dense-regression branch beside a detection branch, never rejoined",
         {"in": ("read_media", {}),
          "depth": ("depth_anything_v2", {"variant": "small"}),
          "det": ("yolov11", {"variant": "nano", "threshold": 0.4}),
          "boxes": ("draw_boxes", {})},
         [("in:image", "depth:image"), ("in:image", "det:image"),
          ("in:image", "boxes:image"), ("det:detections", "boxes:detections")]),

    Case("geometry_chain", "four two-output nodes in a row -- every one can swap its ports",
         {"in": ("read_media", {}),
          "det": ("yolov8", {"variant": "nano", "threshold": 0.4}),
          "rot": ("rotate_with_detections", {"angle": 17.0}),
          "flip": ("flip_horizontal_with_detections", {}),
          "vflip": ("flip_vertical_with_detections", {}),
          "cropped": ("crop_with_detections", {"left": 20, "top": 20}),
          "boxes": ("draw_boxes", {}),
          "gamma": ("gamma_correction", {"gamma": 1.2})},
         [("in:image", "det:image"), ("in:image", "rot:image"), ("det:detections", "rot:detections"),
          ("rot:image", "flip:image"), ("rot:detections", "flip:detections"),
          ("flip:image", "vflip:image"), ("flip:detections", "vflip:detections"),
          ("vflip:image", "cropped:image"), ("vflip:detections", "cropped:detections"),
          ("cropped:image", "boxes:image"), ("cropped:detections", "boxes:detections"),
          ("boxes:image", "gamma:image")]),

    Case("classify_pair", "two classifiers, no join -- both branches are terminal",
         {"in": ("read_media", {}),
          "clip": ("clip", {"variant": "base",
                            "text": "a photograph of a person, a street, a landscape, food"}),
          "siglip": ("siglip2", {"variant": "base-224",
                                 "text": "a photograph of a person, a street, a landscape, food"})},
         [("in:image", "clip:image"), ("in:image", "siglip:image")]),

    Case("ocr_read", "text detection and reading, whose cost swings hugely per image",
         {"in": ("read_media", {}),
          "ocr": ("easyocr", {"variant": "english"}),
          "boxes": ("draw_boxes", {}),
          "labels": ("draw_labels", {})},
         [("in:image", "ocr:image"), ("in:image", "boxes:image"),
          ("ocr:detections", "boxes:detections"),
          ("boxes:image", "labels:image"), ("ocr:detections", "labels:detections")],
         share=0.4),

    Case("pose_chain", "detector into pose, then two annotators that read the same keypoints",
         {"in": ("read_media", {}),
          "det": ("yolov11", {"variant": "nano", "threshold": 0.5}),
          "pad": ("pad_detections", {"padding": 0.1}),
          "pose": ("vitpose", {"variant": "small"}),
          "skeleton": ("draw_skeleton", {}),
          "points": ("draw_keypoints", {})},
         [("in:image", "det:image"), ("det:detections", "pad:detections"),
          ("in:image", "pose:image"), ("pad:detections", "pose:detections"),
          ("in:image", "skeleton:image"), ("pose:detections", "skeleton:detections"),
          ("skeleton:image", "points:image"), ("pose:detections", "points:detections")],
         share=0.4),

    Case("fanout_crops", "one image becomes a batch of crops, and a model runs once per crop",
         {"in": ("read_media", {}),
          "det": ("yolov8", {"variant": "nano", "threshold": 0.5}),
          "crops": ("crop_around_detections", {"padding": 0.05}),
          "clip": ("clip", {"variant": "base", "text": "a person, a vehicle, an animal"})},
         [("in:image", "det:image"), ("in:image", "crops:image"),
          ("det:detections", "crops:detections"), ("crops:image", "clip:image")],
         share=0.4),

    Case("segment_masks", "instance masks, where the compositing is the parallel part",
         {"in": ("read_media", {}),
          "seg": ("yolov26", {"variant": "seg-nano", "threshold": 0.4}),
          "masks": ("draw_masks", {"opacity": 0.4}),
          "polys": ("draw_polygons", {})},
         [("in:image", "seg:image"), ("in:image", "masks:image"),
          ("seg:detections", "masks:detections"),
          ("masks:image", "polys:image"), ("seg:detections", "polys:detections")]),

    Case("long_chain", "ten stages, most of them cheap -- what bounds this is the admission limit",
         {"in": ("read_media", {}),
          "contrast": ("auto_contrast", {"cutoff": 1.0}),
          "gamma": ("gamma_correction", {"gamma": 1.1}),
          "clahe": ("enhance_clahe", {}),
          "rot": ("rotate", {"angle": 5.0}),
          "flip": ("flip_horizontal", {}),
          "vflip": ("flip_vertical", {}),
          "cropped": ("crop", {"left": 8, "top": 8}),
          "det": ("yolov11", {"variant": "nano", "threshold": 0.4}),
          "boxes": ("draw_boxes", {})},
         [("in:image", "contrast:image"), ("contrast:image", "gamma:image"),
          ("gamma:image", "clahe:image"), ("clahe:image", "rot:image"),
          ("rot:image", "flip:image"), ("flip:image", "vflip:image"),
          ("vflip:image", "cropped:image"), ("cropped:image", "det:image"),
          ("cropped:image", "boxes:image"), ("det:detections", "boxes:detections")]),

    Case("save_sink", "a terminal node with a side effect on disk",
         # The path is rewritten into the run's scratch directory before this runs -- see
         # :func:`writes_into`. Left as a bare name it lands in the working directory, which for
         # a tool run from the repository root is the repository.
         {"in": ("read_media", {}),
          "det": ("yolov11", {"variant": "nano", "threshold": 0.4}),
          "boxes": ("draw_boxes", {}),
          "out": ("save_image", {"path": "out.jpg"})},
         [("in:image", "det:image"), ("in:image", "boxes:image"),
          ("det:detections", "boxes:detections"), ("boxes:image", "out:image")],
         share=0.2),

    Case("grounded", "open-vocabulary detection from a phrase -- seconds an image",
         {"in": ("read_media", {}),
          "det": ("grounding_dino", {"variant": "tiny", "text": "person, car, dog, tree",
                                     "threshold": 0.3}),
          "boxes": ("draw_boxes", {}),
          "labels": ("draw_labels", {})},
         [("in:image", "det:image"), ("in:image", "boxes:image"),
          ("det:detections", "boxes:detections"),
          ("boxes:image", "labels:image"), ("det:detections", "labels:detections")],
         share=0.2),

    Case("owl_prompted", "a second open-vocabulary family, drawn on the greyscale of the original",
         {"in": ("read_media", {}),
          "grey": ("to_grayscale", {}),
          "det": ("owlv2", {"variant": "base-ensemble", "text": "a person, a car, a dog",
                            "threshold": 0.15}),
          "labels": ("draw_labels", {})},
         [("in:image", "det:image"), ("in:image", "grey:image"),
          ("grey:image", "labels:image"), ("det:detections", "labels:detections")],
         share=0.1),

    Case("sam3_masks", "concept segmentation into mask compositing",
         {"in": ("read_media", {}),
          "seg": ("sam3", {"variant": "sam3", "text": "person", "threshold": 0.5}),
          "masks": ("draw_masks", {"opacity": 0.5}),
          "polys": ("draw_polygons", {})},
         [("in:image", "seg:image"), ("in:image", "masks:image"),
          ("seg:detections", "masks:detections"),
          ("masks:image", "polys:image"), ("seg:detections", "polys:detections")],
         share=0.1),

    Case("inpaint", "segment then repaint the gap -- the heaviest graph mozo can express",
         # Masks from a closed-vocabulary segmenter rather than from SAM 3, because the point of
         # this graph is to put moebius under load: a prompt that finds nothing makes the repaint
         # a no-op, and a run of no-ops measures the scheduler against no work at all.
         {"in": ("read_media", {}),
          "seg": ("yolov26", {"variant": "seg-nano", "threshold": 0.4}),
          "paint": ("moebius", {"variant": "general", "seed": 7, "dilate": 4})},
         [("in:image", "seg:image"), ("in:image", "paint:image"),
          ("seg:detections", "paint:detections")],
         share=0.05),
]

BY_NAME = {case.name: case for case in CASES}


# --- Graphs nobody designed -----------------------------------------------------------------------

#: What a generated graph may use, and what to set on it. Every model here is a cheap variant: the
#: point of a random graph is the *shape* -- how many branches, how deep, what joins what -- and a
#: shape is no more interesting for having taken a second an image to run.
PALETTE = {
    "yolov8": {"variant": "nano", "threshold": 0.4},
    "yolov11": {"variant": "nano", "threshold": 0.4},
    "yolov12": {"variant": "nano", "threshold": 0.4},
    "yolov26": {"variant": "seg-nano", "threshold": 0.4},
    "rfdetr": {"variant": "nano", "threshold": 0.4},
    "clip": {"variant": "base", "text": "a person, a vehicle, an animal"},
    "depth_anything_v2": {"variant": "small"},
    "to_grayscale": {}, "auto_contrast": {}, "gamma_correction": {"gamma": 1.1},
    "enhance_clahe": {}, "rotate": {"angle": 11.0}, "flip_horizontal": {}, "flip_vertical": {},
    "crop": {"left": 12, "top": 12},
    "rotate_with_detections": {"angle": 9.0}, "flip_horizontal_with_detections": {},
    "flip_vertical_with_detections": {}, "crop_with_detections": {"left": 10, "top": 10},
    "pad_detections": {"padding": 0.05}, "crop_around_detections": {"padding": 0.02},
    "draw_boxes": {}, "draw_labels": {}, "draw_masks": {}, "draw_polygons": {},
    "blur_regions": {}, "pixelate_regions": {},
}


def random_case(seed: int, size: int) -> Case:
    """A type-valid graph nobody designed, grown one node at a time from what it has produced.

    Hand-written graphs test the shapes their author thought of, and the join bug this scheduler
    was built around is one that only appears in shapes where two branches of unequal cost meet.
    So these are grown instead: pick any node whose inputs are all available, wire each input to a
    random value of the right type, and repeat. Every wire is type-checked by construction, which
    is what makes the result a graph rather than a rejection.

    A value may feed several nodes, and nothing forces a node to be used, so branches, diamonds
    and dead ends all arise on their own rather than being arranged.
    """
    from mozo.workflow import catalogue

    generator = random.Random(seed)
    specs = {entry["name"]: entry for entry in catalogue() if entry["name"] in PALETTE}
    nodes: dict = {"in": ("read_media", {})}
    edges: list = []
    #: port type -> the ``"node:port"`` values of that type produced so far.
    pool: dict = {"image": ["in:image"]}

    for step in range(size):
        usable = [entry for entry in specs.values()
                  if all(port["type"] in pool for port in entry["inputs"])]
        if not usable:
            break
        entry = generator.choice(usable)
        node_id = f"n{step}"
        nodes[node_id] = (entry["name"], dict(PALETTE[entry["name"]]))
        for port in entry["inputs"]:
            edges.append((generator.choice(pool[port["type"]]), f"{node_id}:{port['name']}"))
        for port in entry["outputs"]:
            pool.setdefault(port["type"], []).append(f"{node_id}:{port['name']}")

    kinds = [kind for kind, _ in nodes.values()]
    return Case(f"random-{seed}", f"{len(nodes)} nodes: {', '.join(kinds[1:])}", nodes, edges)


def force_runtime(runtime: str) -> dict:
    """Make every model node ask for *runtime*, where the family publishes it.

    A workflow cannot say which artifact to run: the nodes call :func:`mozo.get_model` with a
    family and a variant and nothing else, and ``auto`` prefers ``torch-fp32`` on CUDA. So this
    stands in front of that call rather than changing it -- mozo is what is being measured and is
    left exactly as it ships, and a family that publishes only torch keeps running torch instead
    of failing.

    Returns the map of what each model actually loaded, so a result can say which half of the
    graph the comparison applies to.
    """
    import mozo
    from mozo.manager import ModelManager
    from mozo.weights import artifacts

    models = ModelManager()
    chosen: dict = {}

    def get_model(identifier: str, variant: Optional[str] = None, device: Optional[str] = None):
        if variant is None:
            identifier, variant = identifier.split("/", 1)
        published = artifacts(identifier, variant)
        wanted = runtime if runtime in published else None
        chosen[f"{identifier}/{variant}"] = wanted or "torch-fp32 (nothing else published)"
        extra = {"runtime": wanted} if wanted else {}
        # Not ``mozo.get_model``: the public function forwards only ``device``, so the ``runtime``
        # keyword that ModelManager documents cannot be reached through it.
        return models.get_model(identifier, variant, device=device, **extra)

    mozo.get_model = get_model
    return chosen


# --- Digesting an answer -------------------------------------------------------------------------

#: Elements of a large array that are actually hashed. Digesting whole images turned out to cost
#: more than the graph did -- a depth map and two annotated frames is 16 MiB an item, hashed on the
#: same thread that drains the run, which both inflated the time and held items in flight waiting
#: for it (11 GiB resident, every stage under a quarter busy). A stride over the array answers the
#: question this digest exists to ask: an image assembled from the wrong item's arguments differs
#: everywhere, not in one pixel.
SAMPLE = 1 << 18


def digest(value: Any) -> str:
    """A short stable fingerprint of one node's output.

    Comparing runs means comparing every node's answer for every item, and holding those answers
    would cost more memory than the run does. Floats are rounded before they are hashed, at a
    precision far coarser than anything a scheduler could change and far finer than a moved box:
    the question is whether the pipeline handed the node the same arguments, not whether the GPU
    is bit-reproducible.
    """
    accumulator = hashlib.sha1()
    _feed(accumulator, value)
    return accumulator.hexdigest()[:16]


def _feed(accumulator, value: Any) -> None:
    if value is None:
        accumulator.update(b"none")
    elif isinstance(value, np.ndarray):
        accumulator.update(f"{value.shape}{value.dtype}".encode())
        flat = value.reshape(-1)
        if flat.size > SAMPLE:
            flat = flat[::max(1, flat.size // SAMPLE)]
        if value.dtype.kind == "f":
            flat = np.round(flat, 3)
        accumulator.update(np.ascontiguousarray(flat).tobytes())
    elif isinstance(value, (list, tuple)):
        accumulator.update(f"seq{len(value)}".encode())
        for item in value:
            _feed(accumulator, item)
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            accumulator.update(str(key).encode())
            _feed(accumulator, value[key])
    elif isinstance(value, float):
        accumulator.update(f"{round(value, 4):.4f}".encode())
    elif isinstance(value, (int, str, bool, bytes)):
        accumulator.update(str(value).encode())
    elif hasattr(value, "to_dict"):
        _feed(accumulator, value.to_dict())
    else:
        accumulator.update(repr(value).encode())
    accumulator.update(b"|")


# --- One run -------------------------------------------------------------------------------------

@dataclass
class Run:
    """What one (graph, worker count) run did."""

    workers: int
    items: int
    elapsed: float
    order_ok: bool
    answers: list                       # [(item, {node id: digest})]
    failures: list                      # [(item, node, error)]
    read_ahead: int                     # most items pulled from the source but not yet handed back
    peak_rss_mb: float
    peak_gpu_mb: float
    #: Mean utilisation of the device itself. Not stage saturation -- see :class:`Watch`.
    gpu_busy: float = 0.0
    #: Percentiles behind that mean, so a steady load is distinguishable from a bursty one.
    gpu_spread: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    #: Mismatches traced to a node that is not deterministic on its own. Not scheduling faults.
    unstable: list = field(default_factory=list)
    error: Optional[str] = None

    @property
    def throughput(self) -> float:
        return self.items / self.elapsed if self.elapsed else 0.0


class Watch:
    """What the machine was doing while a run went on: memory, and how busy the device was.

    Device utilisation is here because stage saturation cannot answer for it. A model stage
    reporting 100% means its worker was inside the node for the whole run -- and the node is
    letterboxing, copying to the device, running kernels, then decoding boxes or compositing
    masks, most of which is the CPU's work. So the scheduler can saturate every stage it has while
    the GPU sits at a third, and only one of those two numbers says whether there is anything left
    to win.
    """

    def __init__(self, interval: float = 0.05) -> None:
        self.interval = interval
        self.peak_rss = 0.0
        self.samples: list = []
        #: Driver timestamp of the newest sample already taken, so none is counted twice.
        self.since = 0
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.handle = self._device()

    @staticmethod
    def _device():
        """The NVML handle for the GPU this process is actually using, or None.

        By UUID rather than by index: ``CUDA_VISIBLE_DEVICES`` renumbers what the process sees, so
        device 0 here and device 0 in NVML are not the same card on a two-GPU machine, and the
        wrong one reads as an idle device however busy the run is.
        """
        try:
            import pynvml
            import torch
            pynvml.nvmlInit()
            if not torch.cuda.is_available():
                return None
            uuid = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
            if uuid:
                return pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode())
            return pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:                                 # noqa: BLE001 -- no NVML, no utilisation
            return None

    def _sample(self) -> None:
        page = 4096 / (1 << 20)
        while not self.stop.wait(self.interval):
            try:
                resident = int(Path("/proc/self/statm").read_text().split()[1]) * page
            except (OSError, IndexError, ValueError):
                return
            self.peak_rss = max(self.peak_rss, resident)
            self._drain_nvml()
        self._drain_nvml()

    def _drain_nvml(self) -> None:
        """Take the driver's own samples since the last one seen.

        Not ``nvmlDeviceGetUtilizationRates`` in a loop. That returns one figure for a window the
        driver decides, so polling it faster than the window re-reads the same number: measured
        here, thirty polls at 20 ms on a busy card returned 23 thirty times. Averaging that is
        averaging one window many times over, weighted by nothing meaningful.
        ``nvmlDeviceGetSamples`` hands back the driver's timestamped series instead -- 200 ms
        apart on this card -- so every sample is counted once and none is counted twice.
        """
        if self.handle is None:
            return
        try:
            import pynvml
            _, samples = pynvml.nvmlDeviceGetSamples(
                self.handle, pynvml.NVML_GPU_UTILIZATION_SAMPLES, self.since)
        except Exception:                                 # noqa: BLE001 -- nothing new, or no NVML
            return
        for sample in samples:
            if sample.timeStamp > self.since:
                self.since = sample.timeStamp
                self.samples.append(sample.sampleValue.uiVal)

    @property
    def gpu_busy(self) -> float:
        """Mean device utilisation across the run, as a percentage."""
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def gpu_spread(self) -> dict:
        """The shape of the utilisation, because a mean cannot tell 90-100 from 20-100-20.

        Both average around 60 and they are not the same machine: one is a device kept busy, the
        other is a device that idles between bursts and has headroom the mean hides. So the tenth
        and ninetieth percentiles travel with the mean, along with how many samples they came from
        -- a two-second run yields about ten, and ten samples do not describe a distribution.
        """
        if not self.samples:
            return {"n": 0}
        ordered = sorted(self.samples)
        def at(fraction: float) -> int:
            return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
        return {"n": len(ordered), "mean": self.gpu_busy, "p10": at(0.10), "p50": at(0.50),
                "p90": at(0.90), "max": ordered[-1],
                "above_90": sum(1 for v in ordered if v >= 90) / len(ordered)}

    def __enter__(self) -> Watch:
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.thread = threading.Thread(target=self._sample, daemon=True, name="stress-watch")
        self.thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    @property
    def peak_gpu(self) -> float:
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1 << 20)


def tracked(paths: list, counter: list) -> Iterator:
    """*paths*, counting how many have been pulled -- which is what read-ahead is measured from."""
    for path in paths:
        counter[0] += 1
        yield str(path)


def warm(case: Case, paths: list) -> None:
    """Run the graph on a couple of images and throw the answers away.

    Loading a checkpoint is a once-per-process cost that lands on whichever run happens to be
    first, and the first run is the serial baseline everything else is a speedup over. Left in, it
    reports the model loading as though widening the pipeline had caused it: a graph whose model
    takes four seconds to load and ten milliseconds to run measured 7.2x at two workers, all of it
    the load.
    """
    workflow = case.build()
    for path in paths[:2]:
        try:
            workflow.run(source=str(path))
        except Exception:                                 # noqa: BLE001 -- a warm-up proves nothing
            pass


def measure(case: Case, paths: list, workers: int, model_workers: int,
            fingerprint: bool = False) -> Run:
    """Run one graph over *paths* at one worker count -- to time it, or to check what it said.

    Never both, which cost a rewrite to learn. Fingerprinting every node of every item runs on the
    thread that drains the run, so it overlaps the pipeline rather than following it: subtracting
    what it took subtracts time the graph was using too, and the faster the graph the worse the
    error. Measured at the extreme, ``geometry_chain`` on ONNX at four workers: 4.16 s of digesting
    inside a 4.31 s run, an ``elapsed`` of 0.15 s, and a reported 2714 images a second for a graph
    that really does about 93. It is not a correction factor -- past a certain speed the harness
    becomes the bottleneck and throttles what it is measuring.

    So a timing pass discards each answer as it arrives and is the only pass whose clock is
    reported, and a correctness pass fingerprints everything and its clock is ignored.
    """
    workflow = case.build()
    counter, answers, failures, read_ahead = [0], [], [], 0
    stats: dict = {}

    def failed(item, event) -> None:
        failures.append((str(item), event.node, event.error))

    keywords = {"workers": workers, "on_error": failed, "model_workers": model_workers}
    if workers > 1:
        keywords["stats"] = stats

    with Watch() as watch:
        began = time.perf_counter()
        try:
            for item, results in workflow.run_many(tracked(paths, counter), **keywords):
                if fingerprint:
                    answers.append((str(item), {node: digest(value)
                                                for node, value in results.items()}))
                else:
                    # Counted, not kept. The whole point of a timing pass is that the consumer
                    # costs nothing, so that what is timed is the graph.
                    answers.append((str(item), {}))
                read_ahead = max(read_ahead, counter[0] - len(answers) - len(failures))
            error = None
        except Exception as failure:                      # noqa: BLE001 -- the run itself broke
            error = f"{type(failure).__name__}: {failure}"
        elapsed = time.perf_counter() - began

    # Arrival order, minus the items that never arrived. ``run_many`` promises the order its
    # source had; a failed item is not yielded at all, so it is the only thing allowed to be
    # missing from the sequence.
    broke = {item for item, _, _ in failures}
    due = [str(path) for path in paths if str(path) not in broke]
    return Run(workers=workers, items=len(answers), elapsed=elapsed,
               order_ok=[item for item, _ in answers] == due,
               answers=answers, failures=failures, read_ahead=read_ahead,
               peak_rss_mb=watch.peak_rss, peak_gpu_mb=watch.peak_gpu, gpu_busy=watch.gpu_busy,
               gpu_spread=watch.gpu_spread, stats=dict(stats), error=error)


# --- Checking that widening changed nothing --------------------------------------------------------

def unstable(case: Case, item: str, node: str, times: int = 5) -> bool:
    """Does *node* answer differently on the same image, run serially, with no pipeline involved?

    A digest that moved between worker counts is either a scheduling bug or a model that does not
    give the same answer twice, and the two are not distinguishable from one run. So a mismatch is
    put back through the serial engine on that one image before it is reported: if the answer moves
    there too, the pipeline is not what moved it.

    Measured on ViTPose, which is where this came from: one image in 205 differed at eight workers,
    and the same image differed once in five serial runs -- by a hundredth of a pixel, across the
    rounding boundary that a digest turns into a different string. Reporting that as a scheduling
    failure would bury a real one.
    """
    workflow = case.build()
    seen = set()
    for _ in range(times):
        try:
            seen.add(digest(workflow.run(source=item)[node]))
        except Exception:                                 # noqa: BLE001 -- it failed, not moved
            return False
    return len(seen) > 1


def compare(case: Case, baseline: Run, run: Run) -> list:
    """Every way a widened run may differ from the serial one. Empty is the only acceptable answer."""
    issues = []
    if run.error:
        return [f"workers={run.workers}: the run itself raised -- {run.error}"]
    if run.items != baseline.items:
        issues.append(f"workers={run.workers}: {run.items} items came back, "
                      f"{baseline.items} at workers=1")
    if not run.order_ok:
        issues.append(f"workers={run.workers}: items came back out of the order they went in")

    seen = {item for item, _ in run.answers}
    if len(seen) != len(run.answers):
        issues.append(f"workers={run.workers}: an item was handed back more than once")

    expected = dict(baseline.answers)
    for item, nodes in run.answers:
        if item not in expected:
            issues.append(f"workers={run.workers}: {Path(item).name} was not in the serial run")
            continue
        for node, value in sorted(nodes.items()):
            if expected[item].get(node) != value:
                if unstable(case, item, node):
                    run.unstable.append(f"{Path(item).name}: {node} does not answer the same "
                                        f"twice on its own, so its mismatch says nothing")
                else:
                    issues.append(f"workers={run.workers}: {Path(item).name} node {node!r} "
                                  f"differs from serial ({expected[item].get(node)} vs {value})")
                break                                    # one node per item is enough to report

    if {f[0] for f in run.failures} != {f[0] for f in baseline.failures}:
        issues.append(f"workers={run.workers}: a different set of items failed than at workers=1")

    # The documented ceiling on how far ahead of its results a run may read.
    nodes = len(case.build().order)
    bound = nodes * (2 * run.workers + run.workers)
    if run.read_ahead > bound:
        issues.append(f"workers={run.workers}: read {run.read_ahead} items ahead of its results, "
                      f"above the documented {bound}")

    for node_id, stage in run.stats.get("stages", {}).items():
        if stage["saturation"] > 1.0001:
            issues.append(f"workers={run.workers}: stage {node_id!r} reports "
                          f"saturation {stage['saturation']:.2f}")
        if stage["queue_peak"] > stage["queue_size"]:
            issues.append(f"workers={run.workers}: stage {node_id!r} queued "
                          f"{stage['queue_peak']} above its bound of {stage['queue_size']}")
    return issues


def bottleneck(run: Run) -> str:
    """The stage that was busiest, which is what a further worker would have to make faster."""
    stages = run.stats.get("stages", {})
    if not stages:
        return "-"
    node_id, stage = max(stages.items(), key=lambda pair: pair[1]["saturation"])
    return f"{node_id} {stage['saturation'] * 100:.0f}%"


# --- Driving -------------------------------------------------------------------------------------

def photographs(directory: Path, count: int) -> list:
    """The images to run on, in a fixed order so every run sees the same sequence."""
    found = sorted(p for p in directory.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not found:
        raise SystemExit(f"no images in {directory}")
    return found[:count]


def writes_into(case: Case, scratch: Path) -> Case:
    """*case* with every file a node writes redirected under *scratch*.

    A benchmark that leaves files in the repository it is benchmarking is a benchmark you have to
    clean up after by hand, and one whose ``git status`` lies about what you changed. Rewriting the
    path here rather than in the case keeps the graphs readable -- ``out.jpg`` is what the node
    means -- and keeps the redirection in one place for however many graphs end up writing.
    """
    redirected = {
        node_id: (kind, {**parameters, "path": str(scratch / Path(parameters["path"]).name)}
                  if kind == "save_image" and "path" in parameters else parameters)
        for node_id, (kind, parameters) in case.nodes.items()
    }
    return Case(case.name, case.note, redirected, case.edges, case.share)


def corrupt(directory: Path, how_many: int) -> list:
    """Files that are not images, to prove a broken item reaches ``on_error`` and nothing else does."""
    directory.mkdir(parents=True, exist_ok=True)
    made = []
    for index in range(how_many):
        path = directory / f"corrupt-{index}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"not a photograph" * 8)
        made.append(path)
    return made


def interleave(good: list, bad: list) -> list:
    """Spread the broken files through the run rather than bunching them at one end."""
    if not bad:
        return good
    every = max(1, len(good) // len(bad))
    mixed, spare = [], list(bad)
    for index, path in enumerate(good):
        mixed.append(path)
        if index % every == every - 1 and spare:
            mixed.append(spare.pop())
    return mixed + spare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="Directory of photographs.")
    parser.add_argument("--count", type=int, default=500,
                        help="Images for a graph with share 1.0; heavier graphs take a fraction.")
    parser.add_argument("--workers", default="1,2,4,8", help="Worker counts to compare.")
    parser.add_argument("--model-workers", type=int, default=1,
                        help="Items allowed inside an exclusive node at once.")
    parser.add_argument("--only", default="", help="Comma-separated graph names. Default: all.")
    parser.add_argument("--corrupt", type=int, default=0,
                        help="Unreadable files to mix in, to exercise the failure path.")
    parser.add_argument("--random", type=int, default=0,
                        help="Generate this many graphs nobody designed, and check them too.")
    parser.add_argument("--random-size", type=int, default=7, help="Nodes per generated graph.")
    parser.add_argument("--random-images", type=int, default=60,
                        help="Images each generated graph runs on.")
    parser.add_argument("--seed", type=int, default=1, help="Where the generated graphs start.")
    parser.add_argument("--check", type=int, default=120,
                        help="Images the correctness pass fingerprints, per worker count.")
    parser.add_argument("--runtime", default="",
                        help="Force this artifact where published, e.g. onnx-fp32.")
    parser.add_argument("--out", type=Path, help="Write the full result as JSON here.")
    parser.add_argument("--random-only", action="store_true",
                        help="Run only the generated graphs, not the written ones.")
    parser.add_argument("--list", action="store_true", help="Name the graphs and stop.")
    arguments = parser.parse_args()

    if arguments.list:
        for case in CASES:
            print(f"{case.name:22} share={case.share:<5} {case.note}")
        return

    chosen = [BY_NAME[name] for name in arguments.only.split(",")] if arguments.only else CASES
    if arguments.random:
        generated = [random_case(arguments.seed + index, arguments.random_size)
                     for index in range(arguments.random)]
        chosen = ([] if arguments.random_only else chosen) + generated
    workers = [int(w) for w in arguments.workers.split(",")]
    if workers[0] != 1:
        raise SystemExit("the first worker count must be 1: it is the answer the rest are checked "
                         "against, not merely the slowest")

    loaded = force_runtime(arguments.runtime) if arguments.runtime else {}

    # Outside the repository, and taken away again at the end. What a run writes -- unreadable
    # fixtures, whatever ``save_image`` produces -- is working material, not output anyone keeps.
    scratch = Path(tempfile.mkdtemp(prefix="workflow-stress-"))
    broken = corrupt(scratch, arguments.corrupt)
    report: dict = {"images": str(arguments.images), "workers": workers,
                    "model_workers": arguments.model_workers,
                    "runtime": arguments.runtime or "auto", "models": loaded, "graphs": {}}
    issues: list = []

    for case in (writes_into(case, scratch) for case in chosen):
        budget = arguments.random_images if case.name.startswith("random-") else arguments.count
        count = max(1, int(budget * case.share))
        paths = interleave(photographs(arguments.images, count), broken)
        print(f"\n=== {case.name} -- {case.note}")
        print(f"    {len(paths)} images, {len(case.build().order)} nodes")
        baseline: Optional[Run] = None
        entry: dict = {"note": case.note, "images": len(paths), "runs": {}}
        warm(case, paths)
        #: The correctness pass runs on fewer images than the timing pass. Checking is per item
        #: and finds what it finds in the first hundred as readily as in the five hundredth, while
        #: timing wants a run long enough that starting and finishing it are noise.
        checked = paths[:arguments.check]

        for width in workers:
            run = measure(case, paths, width, arguments.model_workers)
            proof = measure(case, checked, width, arguments.model_workers, fingerprint=True)
            if baseline is None:
                baseline, proof_baseline = run, proof
            found = compare(case, proof_baseline, proof) if width != workers[0] else (
                [f"workers=1: the run itself raised -- {run.error}"] if run.error else [])
            run.unstable = proof.unstable
            issues += [f"{case.name}: {issue}" for issue in found]
            speedup = baseline.elapsed / run.elapsed if run.elapsed and baseline.elapsed else 0.0
            print(f"    workers={width:<2} m={arguments.model_workers} {run.elapsed:7.1f}s  "
                  f"{run.throughput:6.2f} img/s  {speedup:4.2f}x  "
                  f"gpu {run.gpu_busy:3.0f}% [{run.gpu_spread.get('p10', 0):>2}-"
                  f"{run.gpu_spread.get('p90', 0):>3}] n={run.gpu_spread.get('n', 0):<3} "
                  f"rss {run.peak_rss_mb:6.0f}MiB  vram {run.peak_gpu_mb:5.0f}MiB  "
                  f"ahead {run.read_ahead:3}  fail {len(run.failures):3}  "
                  f"slowest {bottleneck(run)}"
                  + ("  ISSUES" if found else ""))
            for issue in found:
                print(f"        ! {issue}")
            for note in run.unstable:
                print(f"        ~ {note}")
            entry["runs"][width] = {
                "elapsed_s": run.elapsed, "throughput": run.throughput, "speedup": speedup,
                "items": run.items, "failures": len(run.failures), "read_ahead": run.read_ahead,
                "peak_rss_mb": run.peak_rss_mb, "peak_gpu_mb": run.peak_gpu_mb,
                "gpu_busy_percent": run.gpu_busy, "gpu_spread": run.gpu_spread,
                "unstable": run.unstable,
                "stats": run.stats, "issues": found, "error": run.error,
            }
        report["graphs"][case.name] = entry

    print("\n=== summary")
    for name, entry in report["graphs"].items():
        times = {width: run["elapsed_s"] for width, run in entry["runs"].items()}
        best = min(times, key=times.get)
        print(f"    {name:22} best workers={best} at {times[best]:.1f}s "
              f"({entry['runs'][best]['speedup']:.2f}x over serial)")

    if loaded:
        print("\n=== what each model ran on")
        for model, runtime in sorted(loaded.items()):
            print(f"    {model:24} {runtime}")

    print(f"\n=== {len(issues)} issue(s)")
    for issue in issues:
        print(f"    ! {issue}")
    report["issues"] = issues

    shutil.rmtree(scratch, ignore_errors=True)

    if arguments.out:
        arguments.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwritten to {arguments.out}")


if __name__ == "__main__":
    main()
