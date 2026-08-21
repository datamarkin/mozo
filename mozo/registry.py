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

__all__ = ["MODEL_REGISTRY", "PROMPTED", "get_model_info"]

from typing import Any

#: Task types whose model is asked a question in words. They differ in what comes back -- SAM 3
#: answers with masks, OWLv2 with boxes -- but the request is the same shape, and so is the way a
#: missing prompt has to be refused. Named here rather than in the server because it is a
#: statement about the task vocabulary, and this is the module that owns those strings; the
#: endpoint and the tests both read it from here rather than keeping copies in step.
PROMPTED = frozenset({"concept_segmentation", "open_vocabulary_detection"})

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

    'easyocr': {
        'adapter_class': 'EasyOCRPredictor',
        'module': 'mozo.adapters.easyocr',
        'task_type': 'text_recognition',
        'description': (
            'EasyOCR by Jaided AI — text detection and recognition. Finds every line of text '
            'and reads it, returning the string, the four corners as read and a confidence. '
            'Two graphs: CRAFT locates, a CRNN reads. A variant is a script rather than a '
            'language — 5 of them, covering 88% of upstream\'s own downloads, with latin alone '
            'spanning 41 languages. Detections carry text, not a class name: OCR reads content, '
            'it does not pick a class out of a vocabulary. Code and weights are Apache 2.0.'
        ),
        'variants': ['english', 'latin', 'chinese-simplified', 'japanese', 'korean'],
    },

    'edgetam': {
        'adapter_class': 'EdgeTamPredictor',
        'module': 'mozo.adapters.edgetam',
        'task_type': 'promptable_segmentation',
        'description': (
            'EdgeTAM by Meta — promptable segmentation, SAM 2 distilled for phones. Click a '
            'point or draw a box and it returns the thing you pointed at, with a mask, a box '
            'and the model\'s own predicted IoU. A 9.1M-parameter image path against SAM 2 '
            'tiny\'s 31.4M. One published model, so one variant. Code and weights are both '
            'Apache 2.0.'
        ),
        'variants': ['edgetam'],
    },

    'owlv2': {
        'adapter_class': 'OwlV2Predictor',
        'module': 'mozo.adapters.owlv2',
        'task_type': 'open_vocabulary_detection',
        'description': (
            'OWLv2 by Google Research — open-vocabulary detection. Name anything in words '
            'and it returns boxes for it, with no class list and no training. 4 variants: '
            'base-ensemble/large-ensemble average the self-trained and fine-tuned checkpoints, '
            'base/large are self-training only. Boxes, not masks — pair it with SAM 2 or '
            'EdgeTAM for those. Unlike SAM 3, the code and all four checkpoints are Apache 2.0.'
        ),
        'variants': ['base-ensemble', 'base', 'large-ensemble', 'large'],
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

    'sam2': {
        'adapter_class': 'Sam2Predictor',
        'module': 'mozo.adapters.sam2',
        'task_type': 'promptable_segmentation',
        'description': (
            'SAM 2 by Meta — promptable segmentation. Click a point or draw a box and it '
            'returns the thing you pointed at, with a mask, a box and the model\'s own '
            'predicted IoU. 4 variants (tiny/small/base_plus/large); the image path only, not '
            'the video tracker. Unlike SAM 3, the weights are Apache 2.0 like the code.'
        ),
        'variants': ['tiny', 'small', 'base_plus', 'large'],
    },

    'sam3': {
        'adapter_class': 'Sam3Predictor',
        'module': 'mozo.adapters.sam3',
        'task_type': 'concept_segmentation',
        'description': (
            'SAM 3 by Meta — promptable concept segmentation. Name a thing in words and it '
            'returns every instance of it, with a mask, a box and a score. One published model, '
            'so one variant. Unlike every other family here the weights are neither Apache-2.0 '
            'nor AGPL: they carry Meta\'s SAM License, which restricts what they may be used '
            'for and binds whoever you serve predictions to. See the NOTICE published beside '
            'the checkpoint.'
        ),
        'variants': ['sam3'],
    },

    'yolov8': {
        'adapter_class': 'YOLOv8Predictor',
        'module': 'mozo.adapters.yolov8',
        'task_type': 'object_detection',
        'description': (
            'YOLOv8 by Ultralytics -- real-time object detection. '
            '5 variants (nano/small/medium/large/xlarge). '
            'The weights are AGPL-3.0, unlike the rest of mozo: serving predictions from them '
            'over a network places obligations on you. See the NOTICE published beside each '
            'checkpoint.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge'],
    },

    'yolov11': {
        'adapter_class': 'YOLOv11Predictor',
        'module': 'mozo.adapters.yolov11',
        'task_type': 'object_detection',
        'description': (
            'YOLO11 by Ultralytics -- real-time object detection, the generation after YOLOv8. '
            '5 variants (nano/small/medium/large/xlarge). '
            'The weights are AGPL-3.0, unlike the rest of mozo: serving predictions from them '
            'over a network places obligations on you. See the NOTICE published beside each '
            'checkpoint.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge'],
    },

    'yolov12': {
        'adapter_class': 'YOLOv12Predictor',
        'module': 'mozo.adapters.yolov12',
        'task_type': 'object_detection',
        'description': (
            'YOLO12 by Ultralytics -- real-time object detection with area attention. '
            '5 variants (nano/small/medium/large/xlarge). '
            'The weights are AGPL-3.0, unlike the rest of mozo: serving predictions from them '
            'over a network places obligations on you. See the NOTICE published beside each '
            'checkpoint.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge'],
    },

    'yolov26': {
        'adapter_class': 'YOLOv26Predictor',
        'module': 'mozo.adapters.yolov26',
        'task_type': 'object_detection',
        'description': (
            'YOLO26 by Ultralytics -- real-time object detection, NMS-free: the head fires once '
            'per object and the network returns a ranked detection list. '
            '5 variants (nano/small/medium/large/xlarge). '
            'The weights are AGPL-3.0, unlike the rest of mozo: serving predictions from them '
            'over a network places obligations on you. See the NOTICE published beside each '
            'checkpoint.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge'],
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
