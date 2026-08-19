"""Load each model once, and keep it.

Loading costs seconds and hundreds of megabytes, so a server that loads per request is unusable.
This holds a model after its first use and hands the same object to every later caller. Nothing
is evicted.

An earlier version bounded the cache by model count. That was the wrong question twice over: a
count cannot tell 0.10 GB from 1.34 GB, and mozo publishes both, so "two models" meant anywhere
from 0.20 GB to 2.68 GB. The same number could not be right for a 6 GB laptop and an 80 GB
accelerator, and a hundred 100 MB models are no problem at all while two large ones may already
be too many. Measured, it also broke the obvious deployment -- detection, segmentation and depth
from one instance, 0.60 GB between them -- into an eviction on every request: 762 ms each,
silently, instead of free.

Memory is therefore the caller's, and the lever is which models you ask for. For a separate
lifetime build a separate :class:`ModelManager` and drop it; for no cache at all, import an
adapter directly.

    >>> from mozo.manager import ModelManager
    >>> models = ModelManager()
    >>> models.get_model("rfdetr", "nano")     # doctest: +SKIP
"""

from __future__ import annotations

__all__ = ["ModelManager"]

from importlib import import_module
from threading import Lock
from typing import Any

from .registry import get_model_info


def _build(family: str, variant: str, **kwargs: Any) -> Any:
    """Import the family's adapter class and instantiate it.

    The registry says which class to import and nothing more. In particular the *variant* is not
    checked here: RF-DETR accepts an unpublished variant name when you bring your own checkpoint,
    because there the variant names an architecture rather than a published model, and only the
    adapter knows that. The registry's own message names the families that do exist.
    """
    info = get_model_info(family)
    adapter = getattr(import_module(info["module"]), info["adapter_class"])
    return adapter(variant=variant, **kwargs)


class ModelManager:
    """A thread-safe cache of loaded models, one entry per distinct request.

    Examples:
        >>> models = ModelManager()
        >>> models.loaded()
        []
    """

    def __init__(self) -> None:
        #: model id -> model, in the order they were first asked for.
        self._models: dict[str, Any] = {}
        #: Held across a build, so two threads cannot load the same model twice. Reads are not
        #: locked: a single dict lookup needs no help to be atomic, and there is no longer any
        #: read-modify-write to protect -- the bookkeeping that needed one was the eviction
        #: order, and nothing is evicted.
        self._load_lock = Lock()

    def get_model(self, family: str, variant: str, device: str | None = None, **kwargs: Any) -> Any:
        """Return a loaded model, building it on first use.

        Args:
            family: Model family, e.g. ``"rfdetr"``.
            variant: Variant within that family, e.g. ``"nano"``.
            device: Where to run. ``None`` lets the adapter pick the best this machine has.
            **kwargs: Passed to the adapter -- ``checkpoint_path``, ``labels``, ``revision``,
                ``runtime``. Part of the cache key, so two checkpoints under one variant name
                are two entries rather than one.

        Raises:
            ValueError: If the family is not registered, or the adapter rejects the variant.
        """
        # Everything that changes what gets built belongs in the identity. A checkpoint of your
        # own, a pinned revision, a different runtime -- each is a different model wearing the
        # same name, and handing back the first build for the second request is the kind of wrong
        # answer that never raises. RF-DETR accepts unpublished variant names precisely so that
        # two people's "my-training" can be two different models.
        extra = "".join(f"|{key}={value!r}" for key, value in sorted(kwargs.items()))
        model_id = f"{family}/{variant}" + (f"@{device}" if device else "") + extra

        resident = self._models.get(model_id)
        if resident is not None:
            return resident

        # The lock is held across the build and nowhere else. A load takes seconds, and a request
        # for a model already in memory is answered above without waiting for it -- that is the
        # case that has to stay fast. Distinct models therefore load one at a time, a warm-up
        # cost paid once.
        with self._load_lock:
            resident = self._models.get(model_id)
            if resident is not None:
                return resident  # another thread loaded it while this one queued

            print(f"[mozo] loading {model_id}", flush=True)
            model = _build(family, variant, device=device, **kwargs)
            self._models[model_id] = model
            return model

    def loaded(self) -> list[str]:
        """Resident model ids, in the order they were first asked for."""
        return list(self._models)
