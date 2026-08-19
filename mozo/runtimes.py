"""Run a published artifact, whatever it happens to be.

A model is an architecture plus weights, but at inference time it is just a function from an
input array to a tuple of output arrays. That function can come from several places -- a torch
module, an ONNX graph, a CoreML package -- and which one a machine should use depends on what
mozo publishes for that model and what the machine can actually execute.

This module owns the artifact-shaped half of that: everything here works from a file and knows
nothing about any particular model. Torch runners are built by the adapter that owns the
architecture, because only it knows how to turn a checkpoint into a module.

    >>> from mozo.runtimes import OnnxRunner
    >>> run = OnnxRunner(path, device="cuda")            # doctest: +SKIP
    >>> boxes, logits = run(batch)                       # doctest: +SKIP

Pre- and post-processing stay with the model, not here: they are model maths and must be
identical whichever runtime produced the numbers in between.
"""

from __future__ import annotations

__all__ = ["CoreMLRunner", "OnnxRunner", "RuntimeError_", "executable", "get_default_device",
           "make_runner", "providers_for", "runnable", "select_runtime"]

import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

#: Cached: the answer cannot change within a process, and ``torch.cuda.is_available()`` costs
#: real time the first time it is asked.
_default_device: str | None = None


def get_default_device() -> str:
    """Return the best device this machine has: ``cuda``, else ``mps``, else ``cpu``.

    Which artifact to run and what to run it on are one question -- :func:`select_runtime` takes
    the answer as its first argument -- so they are answered in one place.

    Examples:
        >>> get_default_device() in {"cuda", "mps", "cpu"}
        True
    """
    global _default_device
    if _default_device is None:
        # torch is an unconditional dependency, and every caller of this is an adapter that has
        # already imported it. The module that must answer without torch is mozo.registry, and
        # it earns that by importing nothing at all.
        import torch

        if torch.cuda.is_available():
            _default_device = "cuda"
        elif torch.backends.mps.is_available():
            _default_device = "mps"
        else:
            _default_device = "cpu"
    return _default_device


#: Preferred artifact per device, best first.
#:
#: Torch leads everywhere, because it is the only runtime measured on every device and the one
#: every model publishes. It is tempting to assume ONNX Runtime wins on CPU -- it is the usual
#: claim -- but on Apple silicon, RF-DETR small measures p50 136 ms on torch against 184 ms
#: through ONNX Runtime on the same CPU, and 106 ms on torch MPS against 183 ms for ONNX, which
#: has no MPS path at all and quietly runs on the CPU. Defaulting to a guess would have shipped
#: that regression silently to everyone on a Mac.
#:
#: fp16 does not appear because mozo does not publish it -- it measured worse on accuracy and
#: no better on speed everywhere it was tried. Keys are still matched by prefix, so a future
#: fp16 artifact would be selectable by name without changing anything here.
#:
#: CoreML leads on Apple silicon because it was measured to: RF-DETR nano runs 10.8 ms through
#: CoreML against 53.3 ms on torch MPS, five times faster, at a worst output delta of 0.001.
#: That is the one entry here that overrides torch, and it is the one with the widest margin.
_PREFERENCE: dict[str, tuple[str, ...]] = {
    "cuda": ("torch-fp32", "onnx-fp32"),
    "cpu": ("torch-fp32", "onnx-fp32"),
    "mps": ("coreml-fp32", "torch-fp32", "onnx-fp32"),
}


#: Key prefixes that name an execution path. A revision also publishes data artifacts -- the
#: licence, the label vocabulary -- and those are not things a model can be run as.
_RUNNABLE = ("torch", "onnx", "coreml", "tensorrt")


#: What each runtime needs importable before it can execute anything. torch is a core
#: dependency and always present; the rest are the caller's to install.
_REQUIRES = {"onnx": "onnxruntime", "coreml": "coremltools", "tensorrt": "tensorrt"}


def runnable(published: list[str]) -> list[str]:
    """Return only the artifact keys that name a runtime.

    Examples:
        >>> runnable(["labels", "onnx-fp16", "torch-fp32"])
        ['onnx-fp16', 'torch-fp32']
    """
    return [key for key in published if key.split("-")[0] in _RUNNABLE]


def executable(published: list[str]) -> list[str]:
    """Return the runtimes this machine can actually run, not merely the ones published.

    Publishing a CoreML artifact does not put ``coremltools`` on the machine reading the
    manifest, and ``auto`` choosing a runtime whose library is missing would make the default
    path fail on a working install. Asking for one by name still raises -- with instructions --
    because that is a request that cannot be honoured rather than a preference to work around.
    """
    from importlib.util import find_spec

    return [key for key in runnable(published)
            if (module := _REQUIRES.get(key.split("-")[0])) is None or find_spec(module)]


#: ONNX Runtime reports input types as strings; these are the ones an artifact can declare.
_ORT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16}


class RuntimeError_(RuntimeError):
    """Raised when an artifact cannot be executed on the requested device."""


def providers_for(device: str) -> list[str]:
    """Return the ONNX Runtime execution providers to try for *device*, best first.

    The CPU provider is always last, because a session that cannot place a node anywhere else
    must still run rather than fail.
    """
    if device.startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # Everything else, Apple silicon included, runs on the CPU provider. Not
    # CoreMLExecutionProvider: it claims partial support and then fails -- on RF-DETR it placed
    # 608 of 1415 nodes, split the graph into 189 partitions, and errored at run time. Reaching
    # Apple's accelerators needs a real CoreML artifact, not this provider.
    return ["CPUExecutionProvider"]


def select_runtime(device: str, published: list[str], requested: str = "auto") -> str:
    """Choose which published artifact to run on *device*.

    Args:
        device: Where the model will run -- ``cpu``, ``cuda``, ``mps``.
        published: Artifact keys the model publishes, from :func:`mozo.weights.artifacts`.
        requested: An explicit key such as ``"onnx-fp16"``, or ``"auto"`` to take the best
            published option for the device.

    Returns:
        One key from *published*.

    Raises:
        RuntimeError_: If *requested* is not published, or nothing suitable exists.

    Examples:
        >>> select_runtime("cpu", ["onnx-fp32", "torch-fp32"])
        'torch-fp32'
        >>> select_runtime("cpu", ["onnx-fp32"])
        'onnx-fp32'
        >>> select_runtime("cpu", ["onnx-fp32", "torch-fp32"], requested="onnx-fp32")
        'onnx-fp32'
    """
    if requested != "auto":
        if requested not in runnable(published):
            raise RuntimeError_(
                f"{requested!r} is not published for this model. "
                f"Available: {', '.join(runnable(published))}"
            )
        return requested

    published = executable(published)
    for key in _PREFERENCE.get(device.split(":")[0], _PREFERENCE["cpu"]):
        if key in published:
            return key

    # Nothing preferred is available, but something executable is -- take it rather than refuse.
    if published:
        return published[0]
    raise RuntimeError_(
        "nothing this model publishes can run here. Published: "
        f"{', '.join(runnable(published)) or 'nothing runnable'}"
    )


def make_runner(path: str | Path, key: str, device: str = "cpu"):
    """Return a runner for one artifact, chosen by its key.

    Args:
        path: The artifact on disk.
        key: Its artifact key, e.g. ``"onnx-fp32"`` or ``"coreml-fp32"``.
        device: Device the caller wants, where the runtime can be pointed at one.

    Raises:
        RuntimeError_: If nothing here can execute that kind of artifact. Torch artifacts are
            excluded deliberately: building a module from a checkpoint needs the architecture,
            which only the family's adapter has.
    """
    kind = key.split("-")[0]
    if kind == "onnx":
        return OnnxRunner(path, device=device)
    if kind == "coreml":
        return CoreMLRunner(path)
    raise RuntimeError_(f"no runner for {key!r}; torch artifacts are built by their adapter")


def _unpacked(archive: Path) -> Path:
    """Return the ``.mlpackage`` directory for *archive*, extracting it beside itself once.

    A CoreML package is a directory, and an artifact is a single verified file, so it travels
    zipped. Extraction lands next to the archive in the cache and is skipped when it is already
    there -- the cache entry is immutable, so one unpack per revision is all it can need.
    """
    if archive.is_dir():
        return archive
    unpacked = archive.with_suffix(".mlpackage")
    if not unpacked.is_dir():
        staging = archive.with_suffix(".unpacking")
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
        staging.replace(unpacked)
    return unpacked


class CoreMLRunner:
    """A CoreML package loaded for inference, callable on a batch.

    CoreML picks its own hardware and offers no execution-provider knob: it schedules across the
    ANE, the GPU and the CPU as it sees fit. On RF-DETR the ANE contributes nothing measurable
    (compute units ALL and CPU_AND_GPU both run at 10.8 ms, CPU-only at 42 ms), so no attempt is
    made to steer it -- the default is already what the measurements were taken with.

    Attributes:
        inputs: Input tensor names, in the order the package declares them.
        outputs: Output tensor names, in the order the package declares them.

    Examples:
        >>> run = CoreMLRunner("model.mlpackage")             # doctest: +SKIP
        >>> boxes, logits = run(batch)                        # doctest: +SKIP
    """

    def __init__(self, path: str | Path) -> None:
        try:
            import coremltools as ct
        except ImportError as error:  # pragma: no cover - depends on the install
            raise RuntimeError_(
                "coremltools is not installed, so CoreML artifacts cannot be run."
            ) from error

        self._model = ct.models.MLModel(str(_unpacked(Path(path))))
        description = self._model.get_spec().description
        self.inputs: list[str] = [i.name for i in description.input]
        self.outputs: list[str] = [o.name for o in description.output]

    def __call__(self, batch: np.ndarray, **extra: Any) -> tuple[np.ndarray, ...]:
        """Run the package and return its outputs in declaration order."""
        feed: dict[str, Any] = {self.inputs[0]: batch}
        feed.update(extra)
        got = self._model.predict(feed)
        return tuple(np.asarray(got[name]) for name in self.outputs)


class OnnxRunner:
    """An ONNX graph loaded into an inference session, callable on a batch.

    ONNX Runtime falls back to the CPU provider silently when the one you asked for is missing
    -- a GPU box with the CPU-only ``onnxruntime`` wheel installed will happily serve at a
    fraction of the speed and say nothing. :attr:`provider` records what actually got used, and
    a fallback prints a line rather than passing unnoticed.

    Args:
        path: Path to the ``.onnx`` file.
        device: Device the caller wants, used to order execution providers.

    Attributes:
        provider: The execution provider the session actually placed nodes on.
        inputs: Input tensor names, in the order the graph declares them.
        outputs: Output tensor names, in the order the graph declares them.
        input_dtype: What the graph wants fed to it. An fp16 artifact declares fp16 inputs, and
            callers should not have to know that -- :meth:`__call__` casts.
        input_shape: The first input's declared shape. A graph exported at a fixed side fixes the
            size its caller must letterbox to, and asking it beats assuming.

    Examples:
        >>> run = OnnxRunner("model.onnx", device="cpu")  # doctest: +SKIP
        >>> outputs = run(np.zeros((1, 3, 512, 512), dtype="float32"))  # doctest: +SKIP
    """

    def __init__(self, path: str | Path, device: str = "cpu") -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - depends on the install
            raise RuntimeError_(
                "onnxruntime is not installed, so ONNX artifacts cannot be run. "
                "Install it, or select a torch runtime."
            ) from error

        wanted = providers_for(device)
        available = ort.get_available_providers()
        usable = [p for p in wanted if p in available]

        self._session = ort.InferenceSession(str(path), providers=usable)
        self.provider: str = self._session.get_providers()[0]
        self.inputs: list[str] = [i.name for i in self._session.get_inputs()]
        self.outputs: list[str] = [o.name for o in self._session.get_outputs()]
        self.input_dtype = _ORT_DTYPES.get(self._session.get_inputs()[0].type, np.float32)
        self.input_shape: tuple = tuple(self._session.get_inputs()[0].shape)

        if self.provider != wanted[0]:
            print(
                f"[mozo] {Path(path).name}: asked for {wanted[0]}, running on {self.provider}. "
                f"Latency will not reflect {device}.",
                flush=True,
            )

    def __call__(self, batch: np.ndarray, **extra: Any) -> tuple[np.ndarray, ...]:
        """Run the graph and return its outputs in declaration order.

        Args:
            batch: The first input, typically an ``NCHW`` float32 array.
            **extra: Any further inputs the graph declares, by name.

        Returns:
            One array per graph output, in the order :attr:`outputs` lists them.
        """
        if batch.dtype != self.input_dtype:
            batch = batch.astype(self.input_dtype)
        feed: dict[str, Any] = {self.inputs[0]: batch}
        feed.update(extra)
        return tuple(self._session.run(None, feed))
