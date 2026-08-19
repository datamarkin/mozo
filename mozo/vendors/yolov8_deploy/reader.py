# SPDX-License-Identifier: Apache-2.0
"""Read a PyTorch ``.pt`` archive into plain Python objects and numpy arrays, without importing torch.

A ``.pt`` file is a ZIP archive holding one pickle (``data.pkl``) plus the raw bytes of every tensor
storage. This module walks that pickle with a restricted unpickler: tensors become numpy arrays and
every class that is not explicitly whitelisted becomes an inert placeholder that keeps its class name
and its recorded attributes. Nothing from the framework that wrote the file is imported or executed.
"""

from __future__ import annotations

import pickle
import zipfile
from typing import Any

import numpy as np

#: Storage class name -> element type. A tag outside this table is a file we do not understand.
STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class Placeholder:
    """Stand-in for a class we refuse to import. Carries the recorded attributes and the class name."""

    def __setstate__(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError(f"{type(self).__name__} was pickled with a {type(state).__name__} state, expected a dict")
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {sorted(self.__dict__)}>"


def class_name(obj: Any) -> str:
    """Return the class name recorded in the checkpoint for ``obj``."""
    return type(obj).__name__


def _rebuild_tensor(storage: np.ndarray, offset: int, size: tuple, stride: tuple, *_ignored: Any) -> np.ndarray:
    """Materialise a tensor as a writable numpy array of shape ``size`` strided over ``storage``."""
    itemsize = storage.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
    )
    return np.array(view, copy=True)  # own writable memory; the archive buffer is read-only


class _Unpickler(pickle.Unpickler):
    """Unpickler that resolves tensors and containers and refuses to import anything else."""

    def __init__(self, file: Any, archive: zipfile.ZipFile, prefix: str):
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix
        self._placeholders: dict[str, type] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor
            if name == "_rebuild_parameter":
                return lambda data, *_ignored: data
        if module == "torch":
            if name == "Size":
                return tuple
            if name in STORAGE_DTYPES:
                return np.dtype(STORAGE_DTYPES[name])
            if name.endswith("Storage"):
                raise ValueError(f"unsupported storage type torch.{name}")
        if module == "collections" and name == "OrderedDict":
            return dict
        if module in ("builtins", "__builtin__"):  # older checkpoints carry the Python 2 module name
            return super().find_class("builtins", name)
        cls = self._placeholders.get(f"{module}.{name}")
        if cls is None:
            cls = type(name, (Placeholder,), {})
            self._placeholders[f"{module}.{name}"] = cls
        return cls

    def persistent_load(self, pid: tuple) -> np.ndarray:
        kind, dtype, key = pid[0], pid[1], pid[2]
        if kind != "storage":
            raise ValueError(f"unsupported persistent id kind {kind!r}")
        numel = pid[4]
        raw = self._archive.read(f"{self._prefix}data/{key}")
        return np.frombuffer(raw, dtype=dtype, count=numel)


def load_checkpoint(path: str) -> Any:
    """Load ``path`` and return the unpickled object graph (arrays are writable numpy arrays)."""
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a zip-format PyTorch checkpoint")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickles = [n for n in names if n.endswith("data.pkl")]
        if len(pickles) != 1:
            raise ValueError(f"{path} holds {len(pickles)} data.pkl entries, expected exactly 1")
        prefix = pickles[0][: -len("data.pkl")]
        with archive.open(pickles[0]) as handle:
            return _Unpickler(handle, archive, prefix).load()
