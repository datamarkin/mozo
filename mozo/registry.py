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

__all__ = ["BOXED", "ENCODES", "MODEL_REGISTRY", "PROMPTED", "get_model_info"]

from typing import Any

#: Task types whose model is asked a question in words. They differ in what comes back -- SAM 3
#: answers with masks, OWLv2 with boxes -- but the request is the same shape, and so is the way a
#: missing prompt has to be refused. Named here rather than in the server because it is a
#: statement about the task vocabulary, and this is the module that owns those strings; the
#: endpoint and the tests both read it from here rather than keeping copies in step.
PROMPTED = frozenset({
    "concept_segmentation", "open_vocabulary_detection", "zero_shot_classification"})

#: Task types whose model is told *where* to look and cannot answer without being told. Two of
#: them, needing a box for opposite reasons: a top-down pose model has no detector and is handed
#: the subject's box, while an inpainter is handed the thing to delete. What they share is all
#: this set encodes -- a request with no box is a mistake rather than an empty frame, and the
#: editor has to offer a rectangle before the request can be made at all. The refusals read
#: differently, and ``mozo.server`` branches on the task to get each one right.
#:
#: Named here for the same reason PROMPTED is, and it is the same shape of fact: the endpoint has
#: to refuse before the image decode and the multi-gigabyte load, and the browser page has to know
#: which families need boxes drawn on the picture before it can offer the control. Neither can be
#: derived from the task name by a reader who does not already know.
#:
#: Distinct from promptable segmentation, which *accepts* a box and equally accepts a click. This
#: is the set that requires one.
BOXED = frozenset({"pose_estimation", "image_inpainting"})

#: What each family can encode, and from what. A family absent from here does not encode at all.
#:
#: Read before anything is loaded, for the same reason PROMPTED is: ``/encode``'s refusal has to
#: come before the image decode and the multi-gigabyte download, not after. ``/predict`` can afford
#: a late 501 because a test proves every registered task has a branch there, so it never fires;
#: on ``/encode`` the reverse holds, and two families in fourteen have one.
#:
#: A dict rather than a set because the kinds differ: CLIP takes both, a re-identification embedder
#: would take images only, and ``/models`` should be able to say which without loading anything.
#: The task type cannot carry this -- two families can classify while only one of them embeds.
ENCODES: dict[str, frozenset[str]] = {
    "clip": frozenset({"image", "text"}),
    "siglip2": frozenset({"image", "text"}),
}

#: Family -> where its adapter lives, what it does, and which variants it publishes.
#: An empty ``variants`` list means the family accepts any variant name.
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    'clip': {
        'adapter_class': 'ClipPredictor',
        'module': 'mozo.adapters.clip',
        'task_type': 'zero_shot_classification',
        'description': (
            'CLIP by OpenAI — zero-shot classification, and the embeddings behind it. Name any '
            'classes in words and it scores an image against them, with no training and no '
            'labelled data. 4 variants (base/base-16/large/large-336), all Vision Transformers. '
            'It also hands back the vectors: an image and a phrase embed into one shared space, '
            'so a corpus embedded once can be searched by words afterwards through a vector '
            'database of your own. Scores are cosine similarities, not probabilities. Code and '
            'weights are both MIT.'
        ),
        'variants': ['base', 'base-16', 'large', 'large-336'],
    },

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

    'grounding_dino': {
        'adapter_class': 'GroundingDinoPredictor',
        'module': 'mozo.adapters.grounding_dino',
        'task_type': 'open_vocabulary_detection',
        'description': (
            'Grounding DINO by IDEA Research — open-vocabulary detection. Describe anything in '
            'words and it returns boxes for it, with no class list and no training. 2 variants: '
            'tiny (Swin-T) and base (Swin-B, 8.3 box AP better zero-shot). Text is fused into '
            'the image features six times over and the decoder attends back to the words, so a '
            'phrase is read rather than treated as a bag of words — reach for it over OWLv2 '
            'when the prompt is a description. Boxes, not masks — pair it with SAM 2 or EdgeTAM '
            'for those. Code is Apache 2.0, and the authors publish the weights under it too.'
        ),
        'variants': ['tiny', 'base'],
    },

    'moebius': {
        'adapter_class': 'MoebiusPredictor',
        'module': 'mozo.adapters.moebius',
        'task_type': 'image_inpainting',
        'description': (
            'Moebius by HUST and vivo AI Lab — object removal. Hand it an image and a mask and '
            'it repaints the hole so the thing was never there. 226M parameters matching '
            'FLUX.1-Fill-dev at 11.9B across six benchmarks, which is what makes it small '
            'enough to be worth deploying. 2 variants: general and places2 (scenes and '
            'backgrounds). Pair the mask with SAM 3 or EdgeTAM. Unlike every other family here '
            'it answers with an image rather than a description of one, and it answers with a '
            'sample rather than an estimate — change the seed and you get a different, equally '
            'valid removal. Runs at 512x512 and nothing else. Apache 2.0 code; the authors '
            'state the weights as Apache 2.0 on GitHub and MIT on the model card, both '
            'permissive and both permitting commercial use.'
        ),
        'variants': ['general', 'places2'],
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
            'RF-DETR by Roboflow — real-time transformer detection, instance segmentation & '
            'keypoints. 4 detection variants (nano/small/medium/large), 4 segmentation variants '
            '(seg-nano/seg-small/seg-medium/seg-large), and keypoint-preview, which returns '
            "COCO's 17 person joints. All variants Apache 2.0 licensed."
        ),
        'variants': [
            'nano', 'small', 'medium', 'large',
            'seg-nano', 'seg-small', 'seg-medium', 'seg-large',
            'keypoint-preview',
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

    'siglip2': {
        'adapter_class': 'Siglip2Predictor',
        'module': 'mozo.adapters.siglip2',
        'task_type': 'zero_shot_classification',
        'description': (
            'SigLIP 2 by Google — zero-shot classification, and the embeddings behind it. Name '
            'any classes in words and it scores an image against them, with no training and no '
            'labelled data. 5 variants across three sizes (base/so400m/giant). Unlike CLIP it '
            'was trained pair by pair with a sigmoid loss, so each score is a probability for '
            'that one image-and-phrase on its own: adding a phrase moves no other score, the set '
            'does not sum to one, and every phrase can be near zero '
            'when none of them fits. Multilingual. It also hands back the vectors, for a vector '
            'database of your own. Code and weights are both Apache-2.0.'
        ),
        'variants': ['base-224', 'base-256', 'so400m-384', 'so400m16-256', 'giant-384'],
    },

    'vitpose': {
        'adapter_class': 'ViTPosePredictor',
        'module': 'mozo.adapters.vitpose',
        'task_type': 'pose_estimation',
        'description': (
            'ViTPose++ by the University of Sydney — human pose estimation. Give it a frame and '
            "the boxes of the people in it, and it returns those same detections with COCO's 17 "
            'joints attached to each. 4 variants (small/base/large/huge). It is top-down: it does '
            'not find people, so pair it with a detector — RF-DETR and the YOLO families all '
            'produce boxes it accepts. It also does not filter them, so pass it the people rather '
            'than everything, or it will place joints on a car as readily as on a person. Code '
            'and weights are both Apache-2.0.'
        ),
        'variants': ['small', 'base', 'large', 'huge'],
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
            'YOLO11 by Ultralytics -- real-time object detection and instance segmentation, the '
            'generation after YOLOv8. 10 variants -- 5 detection '
            '(nano/small/medium/large/xlarge) and 5 segmentation '
            '(seg-nano/seg-small/seg-medium/seg-large/seg-xlarge), which add a mask per detection '
            'and change nothing else. '
            'The weights are AGPL-3.0, unlike the rest of mozo: serving predictions from them '
            'over a network places obligations on you. See the NOTICE published beside each '
            'checkpoint.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge',
                     'seg-nano', 'seg-small', 'seg-medium', 'seg-large', 'seg-xlarge'],
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
            'YOLO26 by Ultralytics -- real-time object detection and instance segmentation, '
            'NMS-free: the head fires once per object and the network returns a ranked detection '
            'list. 10 variants -- 5 detection (nano/small/medium/large/xlarge) and 5 segmentation '
            '(seg-nano/seg-small/seg-medium/seg-large/seg-xlarge), which add a mask per detection '
            'and change nothing else. '
            'The weights are AGPL-3.0, unlike the rest of mozo: serving predictions from them '
            'over a network places obligations on you. See the NOTICE published beside each '
            'checkpoint.'
        ),
        'variants': ['nano', 'small', 'medium', 'large', 'xlarge',
                     'seg-nano', 'seg-small', 'seg-medium', 'seg-large', 'seg-xlarge'],
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
