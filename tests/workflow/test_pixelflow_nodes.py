"""Every node that draws or transforms, run for real, against what it says it produces.

These need no weights. A detection is a few arrays, so the whole set runs on a synthetic image with
synthetic boxes, masks and named keypoints -- which means this sweep always runs rather than
skipping on a machine that has fetched nothing, and it covers the keypoint nodes, which no model
mozo publishes could exercise (there is no pose family in the registry).

Nodes are called through :class:`~mozo.workflow.node.NodeSpec`, which is the same path the engine
takes, because what is under test is the node's contract and not the wiring around it --
``test_graph.py`` holds the wiring.
"""

from __future__ import annotations

import numpy as np
import pytest

import pixelflow as pf
from conftest import port_types_of
from mozo.workflow import PortType, get
from workflow_nodes import shipped

WIDTH, HEIGHT = 64, 48

#: Nodes whose defaults do not fit the fixture, with the parameters that make them run. Small on
#: purpose: a node needing much setting up to work at all is usually a node with a bad default.
SETTINGS = {
    "align_by_keypoints": {"first": "left_eye", "second": "right_eye"},
    "crop": {"right": 40, "bottom": 30},
    "crop_with_detections": {"right": 40, "bottom": 30},
}

def _inputs(spec, image, detections) -> dict:
    """What to feed each of a node's inputs.

    One value per port type is enough: a node declares what it takes, so supplying it is a lookup
    rather than a table of node names.
    """
    supply = {PortType.IMAGE: image, PortType.DETECTIONS: detections}
    return {port.name: supply[port.type] for port in spec.inputs}


#: The nodes this file is responsible for: everything mozo ships that runs without weights. Chosen
#: by what they are rather than by naming the two modules they happen to live in today, so a third
#: drawing module cannot slip past a sweep that still reads green.
DRAWN = shipped(without=("model", "io"))


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    """A synthetic photograph: a gradient, so a transform that does nothing is visible."""
    rows = np.linspace(0, 255, HEIGHT, dtype=np.uint8)[:, None]
    columns = np.linspace(0, 255, WIDTH, dtype=np.uint8)[None, :]
    return np.dstack([np.broadcast_to(rows, (HEIGHT, WIDTH)),
                      np.broadcast_to(columns, (HEIGHT, WIDTH)),
                      np.full((HEIGHT, WIDTH), 128, np.uint8)]).copy()


@pytest.fixture(scope="module")
def detections() -> "pf.Detections":
    """Two detections carrying everything the annotators can draw: box, mask and named keypoints."""
    masks = np.zeros((2, HEIGHT, WIDTH), bool)
    masks[0, 8:20, 8:20] = True
    masks[1, 12:30, 32:50] = True

    keypoints = np.array([
        [[12.0, 12.0, 0.9], [18.0, 12.0, 0.9], [15.0, 18.0, 0.8]],
        [[36.0, 16.0, 0.9], [44.0, 16.0, 0.9], [40.0, 24.0, 0.8]],
    ], np.float32)

    found = pf.detections.from_arrays(
        boxes=np.array([[5.0, 5.0, 25.0, 25.0], [30.0, 10.0, 55.0, 35.0]], np.float32),
        scores=np.array([0.9, 0.8], np.float32),
        class_ids=np.array([0, 1]),
        labels=["person", "dog"],
        masks=masks,
        keypoints=keypoints,
    )
    for detection in found:
        for point, name in zip(detection.keypoints, ("left_eye", "right_eye", "nose")):
            point.name = name
    return found


class TestEveryNodeReturnsWhatItDeclares:
    """The invariant the editor's connection check depends on."""

    @pytest.mark.parametrize("name", DRAWN)
    def test_a_node_s_outputs_are_the_types_its_ports_claim(self, name, image, detections):
        spec = get(name)
        wires = spec(**_inputs(spec, image, detections), **SETTINGS.get(name, {}))

        assert set(wires) == {port.name for port in spec.outputs}
        for port in spec.outputs:
            assert port.type in port_types_of(wires[port.name]), (
                f"{name}.{port.name} declares {port.type.value} but produced "
                f"{type(wires[port.name]).__name__}")

    @pytest.mark.parametrize("name", [n for n in DRAWN
                                      if any(p.type is PortType.IMAGE for p in get(n).inputs)])
    def test_a_node_does_not_alter_the_image_it_was_given(self, name, image, detections):
        """A node hands its result on; the image it was given may still be wired elsewhere.

        A diamond -- detect, then draw boxes on one branch and blur on the other -- is the ordinary
        shape of a workflow, and an annotator that painted in place would have the second branch
        drawing over the first's work.
        """
        spec = get(name)
        before = image.copy()
        spec(**_inputs(spec, image, detections), **SETTINGS.get(name, {}))
        assert np.array_equal(image, before), f"{name} wrote into the image it was handed"

    def test_the_sweep_covers_the_nodes_that_exist(self):
        """Derived, so an empty list would make every case above vanish and read green."""
        assert len(DRAWN) >= 24
        assert "draw_boxes" in DRAWN and "rotate_with_detections" in DRAWN


class TestSeveralOutputsInPractice:
    """The transforms that move an image and its detections together."""

    def test_rotating_moves_the_detections_with_the_image(self, image, detections):
        wires = get("rotate_with_detections")(image=image, detections=detections, angle=90.0)
        assert not np.allclose(wires["detections"][0].bbox, detections[0].bbox), (
            "the boxes did not move, so the image and its detections have parted company")

    def test_flipping_twice_puts_everything_back(self, image, detections):
        once = get("flip_horizontal_with_detections")(image=image, detections=detections)
        twice = get("flip_horizontal_with_detections")(
            image=once["image"], detections=once["detections"])
        assert np.array_equal(twice["image"], image)
        assert np.allclose(twice["detections"][0].bbox, detections[0].bbox)


class TestFanningOutFromOneImage:
    """A node may turn one image into several, which the engine reads as a batch."""

    def test_cropping_around_detections_produces_one_image_per_detection(self, image, detections):
        wires = get("crop_around_detections")(image=image, detections=detections)
        assert len(wires["image"]) == len(detections)
        assert all(crop.ndim == 3 for crop in wires["image"])


class TestColours:
    """The one piece of arithmetic this package does rather than delegates."""

    def test_a_hex_colour_reaches_the_image_unswapped(self, image, detections):
        """RGB in, RGB out. A swapped red and blue is invisible until someone looks at a result."""
        drawn = get("draw_boxes")(image=image, detections=detections,
                                  thickness=2, color="#FF0000")["image"]
        painted = drawn[np.any(drawn != image, axis=2)]
        assert len(painted), "nothing was drawn"
        assert (painted == (255, 0, 0)).all(axis=1).any(), "the box is not red"

    def test_a_malformed_colour_says_what_was_expected(self, image, detections):
        with pytest.raises(ValueError, match="RRGGBB"):
            get("draw_boxes")(image=image, detections=detections, color="red")
