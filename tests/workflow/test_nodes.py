"""The nodes mozo ships: what they offer, and what they actually return.

Two things are held here. That the choices a node offers come from the registry and cannot drift
from it, which is cheap and always runs. And that a node's declared output port matches what its
implementation really produces, which needs the weights and skips without them -- a declared type
the implementation contradicts would let the editor approve a connection that fails at run time.

The sweeps run over the registry and over the node catalogue, not over a list kept here. A
hand-written list is how a fourteenth family lands with a green suite and no coverage, which is the
argument ``tests/families/test_prompted.py`` already makes for doing it this way.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FIXTURE, document, port_types_of, require_weights
from mozo.registry import MODEL_REGISTRY
from mozo.workflow import PortType, Workflow, get
from workflow_nodes import shipped

#: The families that have no node yet, and why. The one thing here that cannot be derived, because
#: it is a decision. Named so that publishing a family without a node fails rather than going
#: unnoticed.
WITHOUT_A_NODE = {
    "sam2": "prompted with point and box coordinates; no widget for that yet",
    "edgetam": "prompted with point and box coordinates; no widget for that yet",
}

#: What a family's task implies it produces. Seven rules rather than one row per family -- the same
#: fact ``mozo/server.py`` dispatches on. Registry data, independent of the node under test, so a
#: node that returns the wrong thing still fails.
PRODUCES = {
    "object_detection": PortType.DETECTIONS,
    "open_vocabulary_detection": PortType.DETECTIONS,
    "concept_segmentation": PortType.DETECTIONS,
    "promptable_segmentation": PortType.DETECTIONS,
    "text_recognition": PortType.DETECTIONS,
    "zero_shot_classification": PortType.CLASSIFICATIONS,
    "depth_estimation": PortType.DEPTH,
}

#: Families that have a node, in registry order. Derived, so a new family is covered by being
#: published -- or by being listed above as deliberately skipped.
MODELLED = tuple(family for family in MODEL_REGISTRY if family not in WITHOUT_A_NODE)

#: Every node mozo ships, model or not. Registering a node is what enrols it in the sweeps below.
SHIPPED = shipped()


class TestTheCatalogue:
    """What the editor is offered."""

    def test_every_published_family_has_a_node_or_a_stated_reason(self):
        assert set(MODEL_REGISTRY) == set(MODELLED) | set(WITHOUT_A_NODE)

    def test_a_family_named_as_skipped_is_one_that_exists(self):
        """A stale exemption would quietly excuse a family that is no longer published."""
        assert set(WITHOUT_A_NODE) <= set(MODEL_REGISTRY)

    @pytest.mark.parametrize("family", MODELLED)
    def test_a_node_offers_exactly_the_variants_the_registry_publishes(self, family):
        variant = next(p for p in get(family).parameters if p.name == "variant")
        assert list(variant.options) == MODEL_REGISTRY[family]["variants"]

    @pytest.mark.parametrize("family", MODELLED)
    def test_a_model_node_takes_an_image_and_says_what_its_task_produces(self, family):
        spec = get(family)
        assert [port.type for port in spec.inputs] == [PortType.IMAGE]
        assert [port.type for port in spec.outputs] == [PRODUCES[MODEL_REGISTRY[family]["task_type"]]]

    @pytest.mark.parametrize("name", SHIPPED)
    def test_every_node_has_something_to_show_in_the_palette(self, name):
        spec = get(name)
        assert spec.category and spec.description

    def test_the_sweeps_cover_the_nodes_that_exist(self):
        """Every list here is derived, so an empty one would make the file pass over nothing."""
        assert len(MODELLED) >= 12
        assert len(SHIPPED) >= len(MODELLED) + 2, "the model nodes, plus at least load and save"
        assert set(MODELLED) <= set(SHIPPED)


class TestReadingAndWriting:
    """The two ends of a workflow, which need no weights."""

    def test_load_image_produces_what_its_port_claims(self, image):
        loaded = Workflow.from_dict(_loads(str(FIXTURE))).run()["load"]
        assert get("load_image").outputs[0].type in port_types_of(loaded)
        assert np.array_equal(loaded, image)

    def test_the_path_can_be_given_at_run_time(self, image):
        saved = Workflow.from_dict(_loads(None))
        assert np.array_equal(saved.run(image=str(FIXTURE))["load"], image)

    @pytest.mark.parametrize("nothing", [None, ""])
    def test_loading_nothing_says_what_to_do_about_it(self, nothing):
        events = list(Workflow.from_dict(_loads(nothing)).stream())
        assert events[-1].status == "failed"
        assert "run(image=...)" in events[-1].error

    def test_an_image_survives_a_round_trip_through_a_file(self, tmp_path, image):
        from mozo.image import load_image

        written = tmp_path / "out.png"
        Workflow.from_dict(document(
            {"load": ("load_image", {"image": str(FIXTURE)}),
             "save": ("save_image", {"path": str(written)})},
            [("load", "image", "save", "image")])).run()

        assert np.array_equal(load_image(str(written)), image)


class TestWhatTheModelsActuallyReturn:
    """The declared output type, checked against the real thing. Skips without weights."""

    @pytest.mark.parametrize("family", MODELLED)
    def test_a_model_node_returns_what_its_output_port_claims(self, family, ran):
        spec = get(family)
        variant = next(p for p in spec.parameters if p.name == "variant").default
        require_weights(family, variant)

        produced = _run_one(family)
        ran.append(family)
        assert spec.outputs[0].type in port_types_of(produced), (
            f"{family} declares {spec.outputs[0].type.value} but returned "
            f"{type(produced).__name__}")


@pytest.fixture(scope="session")
def ran() -> list:
    """Which model nodes actually executed."""
    return []


@pytest.fixture(scope="session", autouse=True)
def _some_model_ran(ran):
    """``require_weights`` skips case by case, so with no weights the sweep asserts nothing.

    Checked once at the end rather than per case: any single family may legitimately be absent, but
    a run where *none* of them executed has not tested what the class claims to. A suite of skips
    reads green.
    """
    yield
    if not ran:
        pytest.fail("no model node ran, so the output-type sweep proved nothing")


def _loads(path) -> dict:
    """A one-node workflow that loads *path*."""
    return document({"load": ("load_image", {"image": path})})


def _run_one(name: str):
    """Run one model node on the fixture photograph, through a workflow rather than by calling it.

    Drains :meth:`Workflow.stream` rather than calling :meth:`Workflow.run` so that a node which
    fails reports why. ``run`` drops a failed node from its results, which turns a model error into
    a ``KeyError`` naming nothing -- one confusing debugging session was enough.
    """
    workflow = Workflow.from_dict(document(
        {"load": ("load_image", {"image": str(FIXTURE)}), "model": (name, {})},
        [("load", "image", "model", "image")]))

    outputs = {}
    for event in workflow.stream():
        if event.status == "failed":
            raise AssertionError(event.error)
        if event.status == "completed":
            outputs[event.node] = event.output
    return outputs["model"]
