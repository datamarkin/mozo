"""One node per model family.

A node per family rather than one node with a family parameter, because the port system requires
it. What a family returns is fixed -- YOLO returns detections, Depth Anything returns a depth map,
CLIP returns classifications -- but it differs *between* families. A single node would have an
output type that depended on the value of one of its parameters, which is not a type the editor can
check a connection against. Refusing a bad wire before anything runs is the point of having port
types at all.

Each node is its family's ``predict`` with the arguments the editor can offer. Nothing here knows
how a model works; :func:`mozo.get_model` does, and it caches, so a node that runs over fifty
images loads the checkpoint once.

Thirteen of the fifteen families are here. SAM 2 and EdgeTAM are not: they are prompted with points
and boxes -- pixel coordinates picked on the image -- and there is no widget for that yet. Adding
one before anything asks for it would be inventing a requirement, so they wait until the editor can
express a click. Everything else about them already works through :func:`mozo.get_model`.

The variants each node offers are read from the registry rather than typed out again -- see
:func:`variants`. There is one list of what mozo publishes and this is not a second one.

What *is* stated twice is each family's one-line description: the registry has one for the
catalogue, and the docstrings here are what the editor's palette shows. They are written for
different places and different lengths, and deriving one from the other would mean splitting a
paragraph whose sentence separator is not consistent across families. Kept deliberately, and said
here so the duplication is a decision rather than a slip.
"""

from __future__ import annotations

from typing import Any, Literal

import mozo
from mozo.registry import get_model_info
from mozo.text import comma_separated

from ..node import Classifications, Depth, Detections, Image
from ..registry import node


def variants(family: str) -> Any:
    """The variants *family* publishes, as a choice the editor renders as a dropdown.

    Read from the registry so that publishing a variant offers it, and retiring one stops offering
    it, without this module being touched. ``Literal`` accepts a tuple, which is what makes a
    runtime list expressible as a static-looking annotation.

    Through :func:`~mozo.registry.get_model_info` rather than indexing the registry, so that a
    misspelled family says which families there are -- at import, before anything can run.
    """
    return Literal[tuple(get_model_info(family)["variants"])]


# --- Detection ---------------------------------------------------------------------------------

@node(category="Detect")
def yolov8(image: Image, variant: variants("yolov8") = "nano",
           threshold: float = 0.5) -> Detections:
    """YOLOv8 by Ultralytics -- real-time object detection."""
    return mozo.get_model("yolov8", variant).predict(image, threshold=threshold)


@node(category="Detect")
def yolov11(image: Image, variant: variants("yolov11") = "nano",
            threshold: float = 0.5) -> Detections:
    """YOLO11 by Ultralytics -- real-time object detection."""
    return mozo.get_model("yolov11", variant).predict(image, threshold=threshold)


@node(category="Detect")
def yolov12(image: Image, variant: variants("yolov12") = "nano",
            threshold: float = 0.5) -> Detections:
    """YOLO12 by Ultralytics -- attention-centric real-time object detection."""
    return mozo.get_model("yolov12", variant).predict(image, threshold=threshold)


@node(category="Detect")
def yolov26(image: Image, variant: variants("yolov26") = "nano",
            threshold: float = 0.5) -> Detections:
    """YOLO26 by Ultralytics -- NMS-free detection, and instance masks on the seg variants."""
    return mozo.get_model("yolov26", variant).predict(image, threshold=threshold)


@node(category="Detect")
def rfdetr(image: Image, variant: variants("rfdetr") = "nano",
           threshold: float = 0.5) -> Detections:
    """RF-DETR by Roboflow -- NMS-free transformer detection, under a permissive licence."""
    return mozo.get_model("rfdetr", variant).predict(image, threshold=threshold)


# --- Detection from a description ----------------------------------------------------------------

@node(category="Detect")
def owlv2(image: Image, text: str = "a person, a car, a dog",
          variant: variants("owlv2") = "base-ensemble", threshold: float = 0.1) -> Detections:
    """OWLv2 by Google -- find anything you can name, with no training."""
    return mozo.get_model("owlv2", variant).predict(
        image, text=comma_separated(text), threshold=threshold)


@node(category="Detect")
def grounding_dino(image: Image, text: str = "person, car, dog",
                   variant: variants("grounding_dino") = "tiny",
                   threshold: float = 0.3) -> Detections:
    """Grounding DINO by IDEA -- open-vocabulary detection from a phrase."""
    return mozo.get_model("grounding_dino", variant).predict(
        image, text=comma_separated(text), threshold=threshold)


# --- Segmentation --------------------------------------------------------------------------------

@node(category="Segment")
def sam3(image: Image, text: str = "person", variant: variants("sam3") = "sam3",
         threshold: float = 0.5) -> Detections:
    """SAM 3 by Meta -- masks for every instance of a concept you name."""
    return mozo.get_model("sam3", variant).predict(
        image, text=comma_separated(text), threshold=threshold)


# --- Pose ------------------------------------------------------------------------------------------

@node(category="Pose")
def vitpose(image: Image, detections: Detections,
            variant: variants("vitpose") = "small") -> Detections:
    """ViTPose++ -- the joints of everyone you point it at. Wire a detector into ``detections``."""
    return mozo.get_model("vitpose", variant).predict(image, detections)


# --- Classification ------------------------------------------------------------------------------

@node(category="Classify")
def clip(image: Image, text: str = "a photo of a cat, a photo of a dog",
         variant: variants("clip") = "base") -> Classifications:
    """CLIP by OpenAI -- score an image against phrases you make up."""
    return mozo.get_model("clip", variant).predict(image, text=comma_separated(text))


@node(category="Classify")
def siglip2(image: Image, text: str = "a photo of a cat, a photo of a dog",
            variant: variants("siglip2") = "base-224") -> Classifications:
    """SigLIP 2 by Google -- zero-shot classification against phrases you make up."""
    return mozo.get_model("siglip2", variant).predict(image, text=comma_separated(text))


# --- Reading -------------------------------------------------------------------------------------

@node(category="Read")
def easyocr(image: Image, variant: variants("easyocr") = "english") -> Detections:
    """EasyOCR by JaidedAI -- find text in an image and read it."""
    return mozo.get_model("easyocr", variant).predict(image)


# --- Depth ---------------------------------------------------------------------------------------

@node(category="Depth")
def depth_anything_v2(image: Image, variant: variants("depth_anything_v2") = "small") -> Depth:
    """Depth Anything V2 -- how far away everything is, from one photograph."""
    return mozo.get_model("depth_anything_v2", variant).predict(image)
