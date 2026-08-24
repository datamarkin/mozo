# SPDX-License-Identifier: Apache-2.0
"""Read a PyTorch ``.pt`` archive into plain Python objects and numpy arrays.

The archive is a ZIP holding one pickle (``data.pkl``) plus one raw byte blob per tensor
storage. Nothing here imports torch: tensors become numpy arrays and every class the pickle
refers to is either translated to a numpy/builtin equivalent or replaced by an inert
:class:`Placeholder` subclass that keeps the original class name and the recorded attributes.
"""

from __future__ import annotations

import builtins
import io
import pickle
import zipfile
from collections import OrderedDict

import numpy as np

# Storage class name -> element dtype. Anything else is an error, not a guess.
_STORAGE_DTYPES = {
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

# Builtins a checkpoint legitimately contains. Everything else in ``builtins`` is refused.
_ALLOWED_BUILTINS = frozenset({"set", "frozenset", "list", "dict", "tuple", "bytearray", "complex"})

# Pickles written by torch name the builtins module the Python 2 way; both spellings mean it.
_BUILTIN_MODULES = ("builtins", "__builtin__")


class Placeholder:
    """Inert stand-in for a class we will not import, e.g. every ``ultralytics`` module.

    Subclasses are created per pickled class and named after it; that name is what later
    selects the block's dataflow, so it must survive loading.
    """

    def __init__(self, *args, **kwargs):
        """Refuse to stand in for a class that is reconstructed by calling it with arguments.

        Those arguments carry meaning a placeholder would throw away, leaving an inert object
        where real data belongs, so this is a loud failure rather than a silent one.
        """
        if args or kwargs:
            raise TypeError(f"{type(self).__name__} is rebuilt from constructor arguments and cannot be a placeholder")

    def __setstate__(self, state):
        """Absorb the recorded attribute dictionary."""
        if not isinstance(state, dict):
            raise TypeError(f"{type(self).__name__} was pickled with a {type(state).__name__} state, expected dict")
        self.__dict__.update(state)

    def __repr__(self):
        """Show the original class name so debugging output stays meaningful."""
        return f"<{type(self).__name__} {sorted(self.__dict__)}>"


class MethodReference:
    """Inert stand-in for one method of a class we will not import.

    A module may record a *function* among its attributes rather than a number: the segmentation
    head written by ``ultralytics`` 8.2.100 keeps ``self.detect = Detect.forward``, which pickles
    as ``getattr(Detect, "forward")``. It is plumbing the head used to reach its base class's
    forward pass, not configuration, and nothing this package builds reads it.

    It is kept by name rather than dropped for the reason :class:`Placeholder` exists at all: a
    reader that silently discarded whatever it did not recognise could not tell a checkpoint that
    records nothing from one whose contents it failed to understand.
    """

    def __init__(self, owner: str, name: str):
        """Remember which class's attribute this was, without resolving either."""
        self.owner = owner
        self.name = name

    def __repr__(self):
        """Show what was referenced, which is all this object is."""
        return f"<method reference {self.owner}.{self.name}>"


def _reference_attribute(owner, name):
    """Resolve a recorded ``getattr(cls, name)`` to an inert :class:`MethodReference`.

    ``getattr`` is a pickle's general-purpose way to reach anything a module exposes, so it is
    resolved only against a class this reader has already refused to import -- where the answer
    can be nothing but inert. Applied to anything else it is an escape from the restriction the
    rest of :meth:`_CheckpointUnpickler.find_class` exists to impose, and is refused.
    """
    if not (isinstance(owner, type) and issubclass(owner, Placeholder)):
        raise ValueError(f"checkpoint takes attribute {name!r} of {owner!r}, which this reader will not resolve")
    return MethodReference(owner.__name__, name)


def _rebuild_tensor(storage, offset, size, stride, *_ignored):
    """Materialise a tensor as a writable, contiguous numpy array.

    ``storage`` is the whole flat array, ``stride`` counts elements (not bytes). Every tensor in
    the archive is created here, so copying here is what makes writability an invariant of the
    reader: views over the archive bytes are read-only and torch refuses to adopt them.
    """
    size, stride = tuple(size), tuple(stride)
    itemsize = storage.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape=size, strides=tuple(s * itemsize for s in stride))
    return np.array(view)


def _rebuild_parameter(data, *_ignored):
    """A parameter is its data; gradients and hooks are training state we do not keep."""
    return data


class _CheckpointUnpickler(pickle.Unpickler):
    """Unpickler that resolves torch constructs and refuses to import anything else."""

    def __init__(self, payload, archive, prefix):
        """Bind the pickle stream to the archive its storages live in."""
        super().__init__(io.BytesIO(payload))
        self._archive = archive
        self._prefix = prefix
        self._placeholders = {}

    def find_class(self, module, name):
        """Translate torch/builtin constructs; everything else becomes a named placeholder."""
        if module == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor
            if name == "_rebuild_parameter":
                return _rebuild_parameter
        if module == "torch":
            if name == "Size":
                return tuple
            if name.endswith("Storage"):
                dtype = _STORAGE_DTYPES.get(name)
                if dtype is None:
                    raise ValueError(f"checkpoint uses unsupported storage type torch.{name}")
                return np.dtype(dtype)
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module in _BUILTIN_MODULES:
            if name == "getattr":
                return _reference_attribute
            if name not in _ALLOWED_BUILTINS:
                raise ValueError(f"checkpoint references disallowed builtin {name!r}")
            return getattr(builtins, name)
        key = f"{module}.{name}"
        if key not in self._placeholders:
            self._placeholders[key] = type(name, (Placeholder,), {})
        return self._placeholders[key]

    def persistent_load(self, saved_id):
        """Return a storage as a flat numpy array over the archive bytes."""
        kind = saved_id[0]
        if kind != "storage":
            raise ValueError(f"checkpoint contains unsupported persistent id kind {kind!r}")
        _, dtype, key, _location, numel = saved_id
        raw = self._archive.read(f"{self._prefix}data/{key}")
        if len(raw) != numel * dtype.itemsize:
            raise ValueError(f"storage {key!r}: {len(raw)} bytes for {numel} {dtype} elements")
        return np.frombuffer(raw, dtype=dtype)


def load_checkpoint(path):
    """Load ``path`` and return the unpickled top-level object of the archive."""
    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith("data.pkl")]
        if len(pickles) != 1:
            raise ValueError(f"{path}: expected exactly one data.pkl in the archive, found {len(pickles)}")
        prefix = pickles[0][: -len("data.pkl")]
        return _CheckpointUnpickler(archive.read(pickles[0]), archive, prefix).load()


def load_model_record(path):
    """Load ``path`` and return the network object it stores (the EMA weights when present)."""
    checkpoint = load_checkpoint(path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path}: expected a checkpoint dict, found {type(checkpoint).__name__}")
    record = checkpoint.get("ema")
    if record is None:
        record = checkpoint.get("model")
    if record is None:
        raise ValueError(f"{path}: checkpoint has neither an 'ema' nor a 'model' entry")
    return record
