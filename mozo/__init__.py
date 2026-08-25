"""mozo -- computer vision models that run from a pip install.

Seventy-eight published variants across fifteen families, served over HTTP or called from Python. No
Docker, no cluster, no configuration: weights are fetched and verified on first use and cached
under ``~/.cache/mozo``.

Server:

    $ mozo start
    $ curl -X POST http://localhost:8000/predict/rfdetr/nano -F file=@image.jpg

Python:

    >>> from mozo import get_model
    >>> model = get_model("rfdetr/nano")             # doctest: +SKIP
    >>> detections = model.predict("image.jpg")      # doctest: +SKIP

:func:`get_model` shares one cache, so asking twice loads once and the model stays for the life
of the process. Build a :class:`~mozo.manager.ModelManager` of your own for a separate cache and
a separate lifetime, or import an adapter directly for no cache at all.

See https://github.com/datamarkin/mozo.
"""

from __future__ import annotations

# Single source of truth for the package version.
# pyproject.toml reads this via [tool.setuptools.dynamic] version.attr
__version__ = "1.0.2"

__all__ = ["MODEL_REGISTRY", "ModelManager", "__version__", "get_model", "get_model_info"]

from mozo.manager import ModelManager
from mozo.registry import MODEL_REGISTRY, get_model_info

#: The cache behind :func:`get_model`. A dict and a lock; it loads nothing until asked.
_shared = ModelManager()


def get_model(identifier: str, variant: str | None = None, device: str | None = None):
    """Load a model from a shared cache.

    Args:
        identifier: ``"family/variant"``, or just the family when *variant* is given.
        variant: The variant, if *identifier* did not carry it.
        device: ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` to take the best available.

    Returns:
        The family's predictor, with a ``predict()`` method.

    Examples:
        >>> get_model("rfdetr/nano")                    # doctest: +SKIP
        >>> get_model("rfdetr", "nano", device="cpu")   # doctest: +SKIP
    """
    if variant is None:
        if "/" not in identifier:
            raise ValueError(
                f"Expected 'family/variant' or a separate variant argument, got {identifier!r}")
        identifier, variant = identifier.split("/", 1)

    return _shared.get_model(identifier, variant, device=device)
