"""What model families exist, answerable without importing one.

Every adapter pulls torch in, and the server has to list what it serves before anything is
loaded. So the catalogue is plain data here, and each family's variant list is written twice --
once here, once as the adapter's ``VARIANTS``. A test per family holds the two in step
(``tests/families/test_*.py::test_registry_agrees_with_the_adapter``); importing one from the
other would put torch back on the import path and defeat the point.

To add a family: add an entry below, and give its adapter the same ``VARIANTS``.

    >>> from mozo.registry import get_model_info
    >>> get_model_info("rfdetr")["task_type"]
    'object_detection'
"""

from __future__ import annotations

__all__ = ["MODEL_REGISTRY", "get_model_info"]

from typing import Any

#: Family -> where its adapter lives, what it does, and which variants it publishes.
#: An empty ``variants`` list means the family accepts any variant name.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    'depth_anything_v2': {
        'adapter_class': 'DepthAnythingV2Predictor',
        'module': 'mozo.adapters.depth_anything_v2',
        'task_type': 'depth_estimation',
        'description': (
            'Depth Anything V2 by TikTok & HKU — monocular depth estimation. '
            '3 relative-depth variants (small/base/large) and the same three sizes '
            'fine-tuned for metric depth indoors (indoor-*) and outdoors (outdoor-*). '
            'Relative base/large are CC-BY-NC-4.0; the rest are Apache 2.0.'
        ),
        'variants': [
            'small', 'base', 'large',
            'indoor-small', 'indoor-base', 'indoor-large',
            'outdoor-small', 'outdoor-base', 'outdoor-large',
        ],
    },

    'rfdetr': {
        'adapter_class': 'RFDETRPredictor',
        'module': 'mozo.adapters.rfdetr',
        'task_type': 'object_detection',
        'description': (
            'RF-DETR by Roboflow — real-time transformer detection & instance segmentation. '
            '4 detection variants (nano/small/medium/large) and 4 segmentation variants '
            '(seg-nano/seg-small/seg-medium/seg-large). All variants Apache 2.0 licensed.'
        ),
        'variants': [
            'nano', 'small', 'medium', 'large',
            'seg-nano', 'seg-small', 'seg-medium', 'seg-large',
        ],
    },
}


def get_model_info(family: str, variant: str | None = None) -> dict[str, Any]:
    """Return one family's catalogue entry, optionally checking that a variant is in it.

    Args:
        family: Model family, e.g. ``"rfdetr"``.
        variant: When given, verify the family publishes it. Families with an empty variant
            list accept anything, so there is nothing to check for those.

    Returns:
        The family's entry from :data:`MODEL_REGISTRY`, not a copy of it.

    Raises:
        ValueError: If the family is unknown, or *variant* is not one it publishes. The message
            names what is available, so callers can pass it straight through rather than
            composing a second one.

    Examples:
        >>> get_model_info("rfdetr", "nano")["module"]
        'mozo.adapters.rfdetr'
        >>> get_model_info("rfdetr", "giant")
        Traceback (most recent call last):
        ValueError: Unknown variant 'giant' for family 'rfdetr'. Available: ...
    """
    entry = MODEL_REGISTRY.get(family)
    if entry is None:
        raise ValueError(
            f"Unknown model family: '{family}'. Available families: {list(MODEL_REGISTRY)}")

    variants = entry.get('variants', [])
    if variant is not None and variants and variant not in variants:
        raise ValueError(
            f"Unknown variant '{variant}' for family '{family}'. Available: {variants}")

    return entry
