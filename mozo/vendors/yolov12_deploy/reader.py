# SPDX-License-Identifier: Apache-2.0
"""Read a PyTorch ``.pt`` checkpoint into numpy without importing torch or ultralytics.

A ``.pt`` file is a ZIP archive containing a pickled object graph plus the raw bytes of
every tensor storage. This module replays that pickle with a restricted unpickler:

* tensors become numpy arrays that own writable, C-contiguous memory;
* every class the pickle names that we do not understand becomes an inert placeholder
  which keeps its **original class name** and absorbs its recorded attributes.

The class name is what later selects a module's dataflow, so it has to survive. Nothing
from the checkpoint is executed and no third-party package is imported.
"""

from __future__ import annotations

import builtins
import pickle
import zipfile
from collections import OrderedDict
from typing import Any, BinaryIO

import numpy as np

# Storage class names appear in the pickle purely as dtype tags.
_DTYPE_OF_STORAGE = {
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
    """Inert stand-in for a class we refuse to import.

    ``type(self).__name__`` is the class name as written in the checkpoint, and every
    recorded attribute lands in ``__dict__``.

    It deliberately defines no constructor. A stand-in is only ever right for a class that
    pickle *reconstructs* and then fills in through ``__setstate__``; a class pickle tries to
    *call* with arguments is one this reader has not accounted for, and the resulting
    ``TypeError`` names it instead of handing back a plausible-looking fake.
    """

    def __setstate__(self, state: Any) -> None:
        """Absorb the attributes the checkpoint recorded for this object."""
        if not isinstance(state, dict):
            raise TypeError(f"unsupported pickled state of type {type(state).__name__} for {type(self).__name__}")
        self.__dict__.update(state)


class _StorageTag:
    """Marker returned for ``torch.<X>Storage``; carries only the element dtype."""

    def __init__(self, dtype: np.dtype) -> None:
        self.dtype = dtype


def _rebuild_tensor(storage: np.ndarray, offset: int, size: tuple, stride: tuple, *_: Any) -> np.ndarray:
    """Rebuild a tensor as a numpy array; ``stride`` is in elements, as torch records it."""
    itemsize = storage.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
    )
    # copy(): the archive bytes are a read-only buffer, and torch.from_numpy needs writable
    # memory it can trust. Making that an invariant here means no call site has to care.
    return view.copy()


def _rebuild_parameter(data: np.ndarray, *_: Any) -> np.ndarray:
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, file: BinaryIO, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix

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
                dtype = _DTYPE_OF_STORAGE.get(name)
                if dtype is None:
                    raise ValueError(f"unrecognised storage type in checkpoint: torch.{name}")
                return _StorageTag(dtype)
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module in ("builtins", "__builtin__"):  # torch writes the Python 2 module name
            return getattr(builtins, name)
        return type(name, (Placeholder,), {})

    def persistent_load(self, saved_id: tuple) -> np.ndarray:
        kind, tag, key, _location, numel = saved_id
        if kind != "storage":
            raise ValueError(f"unsupported persistent id kind in checkpoint: {kind!r}")
        raw = self._archive.read(f"{self._prefix}data/{key}")
        array = np.frombuffer(raw, dtype=tag.dtype)
        if array.size != numel:
            raise ValueError(f"storage {key!r} holds {array.size} elements, checkpoint declares {numel}")
        return array


def load_checkpoint(path: str) -> Any:
    """Load ``path`` and return the model object graph (EMA weights when the file has them)."""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith("data.pkl")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected exactly one data.pkl in the archive, found {len(names)}")
        prefix = names[0][: -len("data.pkl")]
        with archive.open(names[0]) as handle:
            checkpoint = _Unpickler(handle, archive, prefix).load()

    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path}: expected a checkpoint dict, got {type(checkpoint).__name__}")
    model = checkpoint.get("ema") if checkpoint.get("ema") is not None else checkpoint.get("model")
    if model is None:
        raise ValueError(f"{path}: checkpoint contains neither an 'ema' nor a 'model' entry")
    return model
