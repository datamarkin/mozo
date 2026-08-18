"""Decide what a model's class ids are called.

A model emits an integer. That integer is exact: it is the slot the weights were trained to
activate, and it means the same thing every time. What that slot is *called* is not in the
weights — it is metadata, and metadata has to come from somewhere.

So mozo takes names from exactly two places, in order: the caller, then the model's own
published labels. Where neither has them, detections carry the id and no name. That is a worse
answer than a name and a much better answer than a wrong one -- a checkpoint fine-tuned on
twelve classes of your own has ids that mean nothing in COCO's space, and reporting class 3 as
"car" because COCO's class 3 is a car would be wrong in the most convincing way available.

Published models ship a ``labels.json`` beside their weights, so the vocabulary travels with the
bytes it describes, the same way the licence does. Nothing here has a default.

    >>> from mozo.labels import resolve
    >>> resolve("rfdetr", "small")            # doctest: +SKIP
    ['__background__', 'person', 'bicycle', ...]
"""

from __future__ import annotations

__all__ = ["resolve"]

import json
from typing import Any

from . import weights  # imported as a module: this file defines its own resolve()

#: Artifact key of the label vocabulary, published alongside the weights it describes.
_LABELS_KEY = "labels"


def resolve(
    family: str,
    variant: str,
    *,
    caller: Any = None,
    checkpoint: Any = None,
    revision: str | None = None,
    published: bool = False,
) -> Any:
    """Return the class names to attach to this model's results, or ``None`` if unknown.

    Args:
        family: Model family, e.g. ``"rfdetr"``.
        variant: Variant within that family.
        caller: Names the caller passed. Wins over everything -- they know their own model.
        checkpoint: Names read out of the checkpoint, if it carried any.
        revision: Published revision to read labels from. Defaults to the latest.
        published: Whether these are the weights mozo published under *family*/*variant*. It
            defaults to ``False`` so that forgetting it withholds a name rather than inventing
            one -- for a checkpoint the caller supplied, the variant names an architecture, not
            a vocabulary, and the published labels describe different weights entirely.

    Returns:
        Whatever :func:`pixelflow.detections.from_arrays` accepts as ``labels`` -- a list of
        names, a list of ``{"id", "name"}`` dicts, or an id-to-name mapping -- or ``None`` when
        no source has them.

    Examples:
        >>> resolve("rfdetr", "small", caller=["hardhat", "vest"])
        ['hardhat', 'vest']
        >>> resolve("rfdetr", "small") is None  # nothing said these are ours
        True
    """
    if caller is not None:
        return caller
    if checkpoint:
        return checkpoint
    if not published:
        return None

    try:
        path = weights.resolve(family, variant, _LABELS_KEY, revision=revision)
    except weights.WeightsError:
        # Not published, or published without labels. Either way there is nothing to read.
        return None

    return json.loads(path.read_text())
