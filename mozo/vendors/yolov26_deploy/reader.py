# SPDX-License-Identifier: Apache-2.0
"""Read a PyTorch ``.pt`` checkpoint with nothing but ``zipfile``, ``pickle`` and ``numpy``.

A ``.pt`` file is a ZIP archive holding one pickle (``<prefix>/data.pkl``) plus the raw bytes of
every tensor storage under ``<prefix>/data/<key>``. Unpickling it normally would import and
execute the classes it names. This module instead unpickles it under a restricted
:class:`Unpickler` that resolves tensors to numpy arrays and every foreign class to an inert
placeholder that keeps its own class name. The name is the payload: it is what later selects how
a module is built and how it computes.
"""

from __future__ import annotations

import builtins
import pickle
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

# Storage class name (minus the "Storage" suffix) -> element dtype.
STORAGE_DTYPES = {
    "Float": np.float32,
    "Half": np.float16,
    "Double": np.float64,
    "Long": np.int64,
    "Int": np.int32,
    "Short": np.int16,
    "Char": np.int8,
    "Byte": np.uint8,
    "Bool": np.bool_,
}


class Placeholder:
    """Stand-in for a class the checkpoint names that we deliberately refuse to import.

    Subclasses are minted on demand, one per pickled class name, so ``type(obj).__name__``
    survives the round trip. Pickle restores object state through ``__setstate__`` and nothing
    else: a stand-in that also accepted constructor arguments would quietly hand back an inert
    fake of a class that genuinely needed constructing — a builtin resolved under the wrong module
    name, say — and the mistake would surface far from its cause, if at all.
    """

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class _StorageDtype:
    """Marker returned for ``torch.<X>Storage``; only its dtype survives."""

    def __init__(self, dtype: np.dtype) -> None:
        self.dtype = dtype


def _rebuild_tensor(storage, offset: int, size, stride, *_) -> np.ndarray:
    """Materialise a tensor: a strided window into a storage, copied into memory it owns.

    ``offset`` and ``stride`` are counted in elements, not bytes. The copy is not an
    optimisation to skip: ``np.frombuffer`` hands back a read-only view of the archive bytes, and
    every consumer downstream expects a writable array it can hand to ``torch.from_numpy``.
    """
    size, stride = tuple(size), tuple(stride)
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=size, strides=tuple(s * storage.itemsize for s in stride)
    )
    return np.array(view, copy=True)


def _rebuild_parameter(data: np.ndarray, *_) -> np.ndarray:
    return data


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that never imports anything the checkpoint asks for."""

    def __init__(self, file: BinaryIO, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix
        self._minted: dict[str, type] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor
            if name == "_rebuild_parameter":
                return _rebuild_parameter
        if module == "torch":
            if name == "Size":
                return tuple
            if name.endswith("Storage"):
                tag = name[: -len("Storage")]
                if tag not in STORAGE_DTYPES:
                    raise ValueError(f"unsupported storage type 'torch.{name}' in checkpoint")
                return _StorageDtype(STORAGE_DTYPES[tag])
        if module in ("builtins", "__builtin__"):  # torch writes the Python 2 spelling of the module
            return getattr(builtins, name)
        if module == "collections":
            return super().find_class(module, name)
        if name not in self._minted:
            self._minted[name] = type(name, (Placeholder,), {})
        return self._minted[name]

    def persistent_load(self, pid) -> np.ndarray:
        kind, storage_dtype, key, _location, _numel = pid
        if kind != "storage":
            raise ValueError(f"unsupported persistent id kind {kind!r} in checkpoint")
        raw = self._archive.read(f"{self._prefix}/data/{key}")
        return np.frombuffer(raw, dtype=storage_dtype.dtype)


def read_checkpoint(path: str | Path) -> Any:
    """Unpickle ``path`` and return its top-level object (usually a dict)."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected exactly one data.pkl in the archive, found {len(names)}")
        prefix = names[0][: -len("/data.pkl")]
        with archive.open(names[0]) as pickled:
            return _RestrictedUnpickler(pickled, archive, prefix).load()


def read_model(path: str | Path) -> Placeholder:
    """Return the model tree recorded in ``path``, preferring the EMA weights when present."""
    ckpt = read_checkpoint(path)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: checkpoint is a {type(ckpt).__name__}, expected a dict")
    model = ckpt.get("ema")
    if model is None:
        model = ckpt.get("model")
    if not isinstance(model, Placeholder):
        raise ValueError(f"{path}: checkpoint holds no 'ema' or 'model' module tree")
    return model


def children(node: Placeholder) -> dict[str, Placeholder]:
    """Child modules of ``node``, in recorded order."""
    return dict(node.__dict__.get("_modules") or {})


def tensors(node: Placeholder) -> dict[str, np.ndarray]:
    """Parameters and buffers held directly by ``node``, in recorded order."""
    out: dict[str, np.ndarray] = {}
    for group in ("_parameters", "_buffers"):
        for name, value in (node.__dict__.get(group) or {}).items():
            if value is not None:
                out[name] = value
    return out


def attr(node: Placeholder, name: str) -> Any:
    """Read a recorded scalar attribute, raising if the checkpoint does not carry it."""
    if name not in node.__dict__:
        raise KeyError(f"{type(node).__name__} does not record required attribute {name!r}")
    return node.__dict__[name]
