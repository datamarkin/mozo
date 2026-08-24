"""What a function says about itself, and what is refused when it says nothing.

The point of reading nodes off their signatures is that the catalogue cannot disagree with what
runs. These tests hold that: everything the editor is told comes from the function, and a function
that leaves something unsaid is refused at import rather than at run time.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pytest

from mozo.workflow import Color, Depth, Detections, Embedding, Image, NodeSpec, PortType


def described(function, category="Test", outputs=None) -> NodeSpec:
    """Describe *function* without registering it."""
    return NodeSpec.from_function(function, category, outputs)


class TestReadingTheSignature:
    """Ports, parameters, name and description, all from one function."""

    def test_ports_and_parameters_are_split_by_their_annotation(self):
        def draw(image: Image, detections: Detections, thickness: int = 2) -> Image:
            """Draw boxes."""

        spec = described(draw)
        assert [port.name for port in spec.inputs] == ["image", "detections"]
        assert [parameter.name for parameter in spec.parameters] == ["thickness"]

    def test_the_port_types_are_the_ones_annotated(self):
        def mix(image: Image, depth: Depth, embedding: Embedding) -> Detections:
            """Mix."""

        spec = described(mix)
        assert [port.type for port in spec.inputs] == [
            PortType.IMAGE, PortType.DEPTH, PortType.EMBEDDING]
        assert [port.type for port in spec.outputs] == [PortType.DETECTIONS]

    def test_the_name_is_the_function_s(self):
        def sharpen(image: Image) -> Image:
            """Sharpen."""

        assert described(sharpen).name == "sharpen"

    def test_the_description_is_the_docstring_s_first_line(self):
        def sharpen(image: Image) -> Image:
            """Sharpen an image.

            A second paragraph the editor has no room for.
            """

        assert described(sharpen).description == "Sharpen an image."

    def test_a_node_without_a_docstring_describes_itself_as_nothing(self):
        def bare(image: Image) -> Image:
            return image

        assert described(bare).description == ""

    def test_the_order_of_ports_is_the_order_they_are_written(self):
        def join(second: Image, first: Image) -> Image:
            """Join."""

        assert [port.name for port in described(join).inputs] == ["second", "first"]


class TestTheOutputPort:
    """One output, named after what travels through it unless told otherwise."""

    def test_it_is_named_for_its_type_by_default(self):
        def find(image: Image) -> Detections:
            """Find."""

        assert described(find).outputs[0].name == "detections"

    def test_the_name_can_be_overridden(self):
        def draw(image: Image) -> Image:
            """Draw."""

        assert described(draw, outputs=["annotated"]).outputs[0].name == "annotated"

    def test_returning_none_is_a_node_that_only_consumes(self):
        def save(image: Image) -> None:
            """Save."""

        assert described(save).outputs == ()

    def test_returning_something_that_cannot_travel_is_refused(self):
        def odd(image: Image) -> dict:
            """Odd."""

        with pytest.raises(TypeError, match="not a port type"):
            described(odd)


class TestSeveralOutputs:
    """A node that produces more than one thing at once."""

    def test_a_tuple_return_declares_one_port_per_member(self):
        def rotate(image: Image, angle: float = 0.0) -> tuple[Image, Detections]:
            """Rotate."""

        spec = described(rotate)
        assert [(port.name, port.type) for port in spec.outputs] == [
            ("image", PortType.IMAGE), ("detections", PortType.DETECTIONS)]

    def test_the_names_can_be_overridden_in_order(self):
        def crop(image: Image) -> tuple[Image, Image]:
            """Crop."""

        assert [port.name for port in described(crop, outputs=["inside", "outside"]).outputs] == [
            "inside", "outside"]

    def test_two_outputs_of_one_type_must_be_named_apart(self):
        def crop(image: Image) -> tuple[Image, Image]:
            """Crop."""

        with pytest.raises(TypeError, match="two outputs named the same"):
            described(crop)

    def test_naming_a_different_number_than_it_returns_is_refused(self):
        def rotate(image: Image) -> tuple[Image, Detections]:
            """Rotate."""

        with pytest.raises(TypeError, match="names 1 outputs but returns 2"):
            described(rotate, outputs=["image"])

    def test_a_tuple_of_something_that_cannot_travel_is_refused(self):
        def odd(image: Image) -> tuple[Image, dict]:
            """Odd."""

        with pytest.raises(TypeError, match="cannot travel"):
            described(odd)

    def test_the_catalogue_lists_every_output(self):
        def rotate(image: Image) -> tuple[Image, Detections]:
            """Rotate."""

        assert described(rotate).to_dict()["outputs"] == [
            {"name": "image", "type": "image"},
            {"name": "detections", "type": "detections"},
        ]


class TestRunningASpec:
    """What calling a spec hands back, which is what the engine wires up."""

    def test_one_output_arrives_on_its_port(self):
        def brighten(image: Image) -> Image:
            """Brighten."""
            return image + 1

        assert described(brighten)(image=np.zeros((1, 1, 3), np.uint8))["image"][0, 0, 0] == 1

    def test_several_outputs_arrive_on_their_own_ports(self):
        def two(image: Image) -> tuple[Image, Image]:
            """Two."""
            return image + 1, image + 2

        wires = described(two, outputs=["a", "b"])(image=np.zeros((1, 1, 3), np.uint8))
        assert (wires["a"][0, 0, 0], wires["b"][0, 0, 0]) == (1, 2)

    def test_a_node_that_produces_nothing_wires_nothing(self):
        def save(image: Image) -> None:
            """Save."""

        assert described(save)(image=np.zeros((1, 1, 3), np.uint8)) == {}

    def test_returning_the_wrong_number_of_things_says_so(self):
        def two(image: Image) -> tuple[Image, Image]:
            """Two."""
            return image

        with pytest.raises(TypeError, match="declares 2 outputs"):
            described(two, outputs=["a", "b"])(image=np.zeros((1, 1, 3), np.uint8))

    def test_a_batch_puts_a_list_on_each_port_separately(self):
        def two(image: Image) -> tuple[Image, Image]:
            """Two."""
            return image + 1, image + 2

        spec = described(two, outputs=["a", "b"])
        wires = spec(image=[np.zeros((1, 1, 3), np.uint8), np.ones((1, 1, 3), np.uint8)])
        assert [v[0, 0, 0] for v in wires["a"]] == [1, 2]
        assert [v[0, 0, 0] for v in wires["b"]] == [2, 3]

    def test_the_caller_sees_a_tuple_in_declared_order(self):
        def two(image: Image) -> tuple[Image, Image]:
            """Two."""
            return image + 1, image + 2

        spec = described(two, outputs=["a", "b"])
        first, second = spec.result(spec(image=np.zeros((1, 1, 3), np.uint8)))
        assert (first[0, 0, 0], second[0, 0, 0]) == (1, 2)


class TestParameters:
    """Widget, default and choices, all from the annotation."""

    @pytest.mark.parametrize("annotation, default, kind", [
        (int, 2, "int"), (float, 0.5, "float"), (str, "x", "str"),
        (bool, True, "bool"), (Color, "#00FF00", "color"),
    ])
    def test_the_annotation_chooses_the_widget(self, annotation, default, kind):
        # Set rather than written in the signature: under ``from __future__ import annotations``
        # a parametrised annotation stringifies to the *name* "annotation", which nothing can
        # resolve. One statement of the type, and it is the one that runs.
        def one(image: Image, value=default) -> Image:
            """One."""

        one.__annotations__["value"] = annotation
        assert described(one).parameters[0].kind == kind

    def test_a_literal_is_a_choice_and_carries_its_options(self):
        def save(image: Image, format: Literal["JPEG", "PNG"] = "PNG") -> None:
            """Save."""

        parameter = described(save).parameters[0]
        assert parameter.kind == "select"
        assert parameter.options == ("JPEG", "PNG")
        assert parameter.default == "PNG"

    def test_a_default_outside_the_choices_is_refused(self):
        def save(image: Image, format: Literal["JPEG", "PNG"] = "GIF") -> None:
            """Save."""

        with pytest.raises(TypeError, match="not one of"):
            described(save)

    def test_a_parameter_may_be_left_unset(self):
        def draw(image: Image, thickness: Optional[int] = None) -> Image:
            """Draw."""

        thickness = described(draw).parameters[0]
        assert (thickness.kind, thickness.default, thickness.optional) == ("int", None, True)

    def test_the_catalogue_says_which_parameters_may_be_left_unset(self):
        def draw(image: Image, thickness: Optional[int] = None, padding: int = 6) -> Image:
            """Draw."""

        assert described(draw).to_dict()["parameters"] == [
            {"name": "thickness", "kind": "int", "default": None, "optional": True},
            {"name": "padding", "kind": "int", "default": 6},
        ]

    def test_an_optional_parameter_with_a_value_for_a_default_is_refused(self):
        """A default that means "unset" is the thing optional replaces."""
        def draw(image: Image, thickness: Optional[int] = 0) -> Image:
            """Draw."""

        with pytest.raises(TypeError, match="its default is None"):
            described(draw)

    def test_a_parameter_with_no_default_is_refused(self):
        def scale(image: Image, factor: float) -> Image:
            """Scale."""

        with pytest.raises(TypeError, match="no default"):
            described(scale)

    def test_a_parameter_the_editor_has_no_widget_for_is_refused(self):
        def odd(image: Image, options: dict = {}) -> Image:
            """Odd."""

        with pytest.raises(TypeError, match="no widget"):
            described(odd)


class TestRefusals:
    """What a node may not be."""

    def test_an_unannotated_argument_is_refused(self):
        def vague(image: Image, thing=1) -> Image:
            """Vague."""

        with pytest.raises(TypeError, match="no annotation"):
            described(vague)

    def test_an_input_with_a_default_is_refused(self):
        def odd(image: Image = None) -> Image:
            """Odd."""

        with pytest.raises(TypeError, match="input 'image' has a default"):
            described(odd)


class TestTheCatalogueEntry:
    """What the editor is handed."""

    def test_it_states_the_ports_the_parameters_and_the_choices(self):
        def save(image: Image, path: str = "out.jpg",
                 format: Literal["JPEG", "PNG"] = "PNG") -> None:
            """Write an image to disk."""

        entry = described(save, category="Output").to_dict()
        assert entry["name"] == "save"
        assert entry["category"] == "Output"
        assert entry["description"] == "Write an image to disk."
        assert entry["inputs"] == [{"name": "image", "type": "image"}]
        assert entry["outputs"] == []
        assert entry["parameters"] == [
            {"name": "path", "kind": "str", "default": "out.jpg"},
            {"name": "format", "kind": "select", "default": "PNG", "options": ["JPEG", "PNG"]},
        ]

    def test_it_names_the_output_port_when_there_is_one(self):
        def find(image: Image) -> Detections:
            """Find."""

        assert described(find).to_dict()["outputs"] == [{"name": "detections", "type": "detections"}]


class TestAskingForAPortThatIsNotThere:

    def test_the_message_names_the_ports_that_are(self):
        def draw(image: Image, detections: Detections) -> Image:
            """Draw."""

        with pytest.raises(KeyError, match="image.*detections"):
            described(draw).port("picture")

    def test_a_node_with_no_inputs_says_so(self):
        def source(width: int = 1) -> Image:
            """Source."""

        with pytest.raises(KeyError, match="takes no input"):
            described(source).port("image")
