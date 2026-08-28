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

from conftest import (CLIP_FPS, CLIP_FRAMES, FIXTURE, document, port_types_of,
                      weights_are_here)
from mozo.registry import MODEL_REGISTRY
from mozo.workflow import PortType, Workflow, get
from mozo.workflow.node import Context
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
    "pose_estimation": PortType.DETECTIONS,
    # The one task whose answer is a picture. Every other entry here describes an image; this
    # one replaces it, which is why IMAGE appears on both sides of the node.
    "image_inpainting": PortType.IMAGE,
}

#: What to wire into a model node's inputs other than its image, and what produces it. Most
#: families take a photograph and nothing else; a top-down pose model is told where the people are,
#: so running it means running a detector first. Derived from the port type rather than from the
#: family, so a second family that consumes detections needs no entry.
FEEDS = {PortType.DETECTIONS: ("rfdetr", "nano")}

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
        """Every model node reads a photograph first, and declares one output matching its task.

        A node may take more than the image -- ViTPose is told where the people are -- but anything
        further has to be something another node can produce, or the editor would offer an input
        nothing could ever be wired into.
        """
        spec = get(family)
        assert spec.inputs[0].type == PortType.IMAGE
        assert all(port.type in FEEDS for port in spec.inputs[1:])
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

    def test_read_media_produces_what_its_port_claims(self, image):
        loaded = Workflow.from_dict(_loads(str(FIXTURE))).run()["load"]
        assert get("read_media").outputs[0].type in port_types_of(loaded)
        assert np.array_equal(loaded, image)

    def test_the_path_can_be_given_at_run_time(self, image):
        saved = Workflow.from_dict(_loads(None))
        assert np.array_equal(saved.run(source=str(FIXTURE))["load"], image)

    @pytest.mark.parametrize("nothing", [None, ""])
    def test_loading_nothing_says_what_to_do_about_it(self, nothing):
        events = list(Workflow.from_dict(_loads(nothing)).stream())
        assert events[-1].status == "failed"
        assert "run(source=...)" in events[-1].error

    def test_an_image_survives_a_round_trip_through_a_file(self, tmp_path, image):
        from mozo.image import load_image

        written = tmp_path / "out.png"
        Workflow.from_dict(document(
            {"load": ("read_media", {"source": str(FIXTURE)}),
             "save": ("save_image", {"path": str(written)})},
            [("load", "image", "save", "image")])).run()

        assert np.array_equal(load_image(str(written)), image)


class TestOneNodeReadsBothKinds:
    """One node reads an image or a video. :mod:`mozo.workflow.nodes.io` says why."""

    def test_the_same_workflow_takes_either_without_being_rewired(self, clip, image):
        """One saved document, two files, no edit in between. The whole point of the merge."""
        made = Workflow.from_dict(_loads(None))
        stills = list(made.process(source=str(FIXTURE)))
        assert len(stills) == 1
        assert np.array_equal(stills[0][1]["load"], image)
        assert len(list(made.process(source=str(clip)))) == CLIP_FRAMES

    def test_the_kind_is_read_from_the_extension(self, tmp_path, clip):
        """Named rather than sniffed, so being wrong is something a person can see and correct.

        The same bytes under an image's name are decoded as one and say so, rather than quietly
        producing a frame nobody asked for.
        """
        misnamed = tmp_path / "clip.jpg"
        misnamed.write_bytes(clip.read_bytes())
        events = list(Workflow.from_dict(_loads(str(misnamed))).stream())
        assert events[-1].status == "failed"

    def test_an_image_declares_no_rate_rather_than_a_wrong_one(self):
        """A photograph has no frame rate, and saying so is what makes a video sink ask for one."""
        run = Context()
        frame = next(get("read_media").run(run, source=str(FIXTURE)))
        assert run.fps is None and run.frames == 1 and run.is_live is False
        assert (run.height, run.width) == frame.shape[:2]

    def test_a_video_declares_the_rate_it_yields_at(self, clip):
        """Strided, so the declared rate is the yielded one rather than the file's."""
        run = Context()
        frames = list(get("read_media").run(run, source=str(clip), stride=2))
        assert run.fps == CLIP_FPS / 2
        assert len(frames) == (CLIP_FRAMES + 1) // 2


class TestWhatTheRunSaysItWillDo:
    """The count a sink reads to decide whether it can take one filename."""

    def test_the_count_it_declares_is_the_count_it_yields(self, clip):
        """A sink refuses "many images, one filename" by reading this, so an estimate that ignored
        ``count`` would refuse runs that were going to be fine."""
        run = Context()
        frames = list(get("read_media").run(run, source=str(clip), count=3))
        assert run.frames == 3 == len(frames)


class TestOneImagePerItem:
    """``save_image`` used to take one filename and write it once per item."""

    def test_every_item_gets_its_own_file(self, clip, tmp_path):
        """Ten frames in produced one file, overwritten nine times, with nothing raised."""
        out = tmp_path / "frames"
        made = Workflow.from_dict(document(
            {"load": ("read_media", {"source": str(clip)}),
             "save": ("save_image", {"path": str(out)})},
            [("load", "image", "save", "image")]))
        assert len(list(made.process())) == CLIP_FRAMES
        assert len(list(out.iterdir())) == CLIP_FRAMES

    def test_one_filename_and_many_images_is_refused_before_anything_is_written(
            self, clip, tmp_path):
        """Refused on what the source says it will produce, and in a preview too -- ``save_image``
        says why."""
        target = tmp_path / "only.jpg"
        made = Workflow.from_dict(document(
            {"load": ("read_media", {"source": str(clip)}),
             "save": ("save_image", {"path": str(target)})},
            [("load", "image", "save", "image")]))

        assert list(made.stream())[-1].status == "failed"
        with pytest.raises(RuntimeError, match="one filename"):
            list(made.process())
        assert not target.exists()

    def test_one_filename_is_right_for_a_run_of_one(self, tmp_path):
        """A single photograph and a named file is the ordinary case and stays ordinary."""
        target = tmp_path / "only.jpg"
        made = Workflow.from_dict(document(
            {"load": ("read_media", {"source": str(FIXTURE)}),
             "save": ("save_image", {"path": str(target)})},
            [("load", "image", "save", "image")]))
        assert len(list(made.process())) == 1
        assert target.exists()

    def test_the_failure_names_the_item_rather_than_printing_it(self, clip, tmp_path):
        """The item is the frame in a ``process`` run, so the message used to be the repr of a
        720x1280 array with the reason buried under it."""
        made = Workflow.from_dict(document(
            {"load": ("read_media", {"source": str(clip)}),
             "save": ("save_image", {"path": str(tmp_path / "only.jpg")})},
            [("load", "image", "save", "image")]))
        with pytest.raises(RuntimeError, match=r"^a \d+x\d+ image: "):
            list(made.process())


class TestWhatTheModelsActuallyReturn:
    """The declared output type, checked against the real thing. Skips without weights."""

    @pytest.mark.parametrize("family", MODELLED)
    def test_a_model_node_returns_what_its_output_port_claims(self, family, ran, absent):
        spec = get(family)
        variant = next(p for p in spec.parameters if p.name == "variant").default
        needed = [(family, variant)] + [FEEDS[port.type] for port in spec.inputs[1:]]
        missing = [f"{f}/{v}" for f, v in needed if not weights_are_here(f, v)]
        if missing:
            absent.append(family)
            pytest.skip(f"weights are not here: {', '.join(missing)}")

        produced = _run_one(family)
        ran.append(family)
        assert spec.outputs[0].type in port_types_of(produced), (
            f"{family} declares {spec.outputs[0].type.value} but returned "
            f"{type(produced).__name__}")


@pytest.fixture(scope="session")
def ran() -> list:
    """Which model nodes actually executed."""
    return []


@pytest.fixture(scope="session")
def absent() -> list:
    """Which of them skipped because their weights are not on this machine."""
    return []


@pytest.fixture(scope="session", autouse=True)
def _some_model_ran(ran, absent):
    """The sweep skips case by case, so with no weights it asserts nothing.

    Checked once at the end rather than per case: any single family may legitimately be absent, but
    a run where *none* of them executed has not tested what the class claims to. A suite of skips
    reads green.

    Unless every family was absent, which is not a failure but a fact about the machine -- a clean
    checkout, or CI, which runs offline against an empty cache on purpose. The guard is for a sweep
    that *could* have run something and did not; demanding weights that were never going to be
    there would only mean the gate could never be green.
    """
    yield
    if not ran and len(absent) < len(MODELLED):
        pytest.fail("no model node ran, so the output-type sweep proved nothing")


def _loads(path) -> dict:
    """A one-node workflow that reads *path*."""
    return document({"load": ("read_media", {"source": path})})


def _run_one(name: str):
    """Run one model node on the fixture photograph, through a workflow rather than by calling it.

    Drains :meth:`Workflow.stream` rather than calling :meth:`Workflow.run` so that a node which
    fails reports why. ``run`` drops a failed node from its results, which turns a model error into
    a ``KeyError`` naming nothing -- one confusing debugging session was enough.
    """
    nodes = {"load": ("read_media", {"source": str(FIXTURE)}), "model": (name, {})}
    edges = [("load", "image", "model", "image")]
    # Anything the node needs beyond the photograph is produced by another node, wired up here.
    # Feeding it a hand-built value instead would test the node against a fixture rather than
    # against what the editor can actually connect to it.
    for port in get(name).inputs[1:]:
        feeder, variant = FEEDS[port.type]
        nodes[feeder] = (feeder, {"variant": variant})
        edges += [("load", "image", feeder, "image"),
                  (feeder, get(feeder).outputs[0].name, "model", port.name)]

    workflow = Workflow.from_dict(document(nodes, edges))

    outputs = {}
    for event in workflow.stream():
        if event.status == "failed":
            raise AssertionError(event.error)
        if event.status == "completed":
            outputs[event.node] = event.output
    return outputs["model"]
