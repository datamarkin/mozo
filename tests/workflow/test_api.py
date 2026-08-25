"""The workflow runtime over HTTP, and the two rules the HTTP layer has to keep.

The endpoints are thin on purpose -- building a :class:`~mozo.workflow.graph.Workflow` is what
validates it, so there is no second opinion here to hold against the first. What these tests cover
is the part that is genuinely the API's own: what a refusal looks like, what a result looks like as
JSON, and that mounting all of this changed nothing about the model server it hangs off.
"""

from __future__ import annotations

import json

import pytest

import mozo
from conftest import FIXTURE, document, require_present
import workflow_nodes  # noqa: F401 -- imported to register the port-type test nodes


def as_json(nodes: dict, edges: list = ()) -> str:
    """A workflow document, as the form field carries it."""
    return json.dumps(document(nodes, edges))


GRAYSCALE = as_json(
    {"load": ("load_image", {"image": ""}), "gray": ("to_grayscale", {})},
    [("load", "image", "gray", "image")])


class TestTheCatalogue:
    """What the editor asks for before it can draw a palette."""

    def test_it_lists_every_node(self, client):
        offered = client.get("/workflow/nodes").json()["nodes"]
        assert {node["name"] for node in offered} == set(mozo.workflow.names())

    def test_each_entry_carries_what_the_editor_needs_to_draw_it(self, client):
        offered = {node["name"]: node for node in client.get("/workflow/nodes").json()["nodes"]}
        boxes = offered["draw_boxes"]
        assert boxes["category"] == "Annotate" and boxes["description"]
        assert [port["type"] for port in boxes["inputs"]] == ["image", "detections"]
        assert boxes["outputs"] == [{"name": "image", "type": "image"}]
        assert {p["name"]: p["kind"] for p in boxes["parameters"]} == {
            "thickness": "int", "color": "color"}

    def test_a_choice_arrives_with_its_options(self, client):
        offered = {node["name"]: node for node in client.get("/workflow/nodes").json()["nodes"]}
        variant = next(p for p in offered["yolov26"]["parameters"] if p["name"] == "variant")
        assert variant["kind"] == "select"
        assert "seg-nano" in variant["options"]


class TestValidating:
    """Whether a document is a workflow, without running it."""

    def test_a_good_workflow_reports_its_order_and_its_ends(self, client):
        answer = client.post("/workflow/validate", data={"workflow": GRAYSCALE}).json()
        assert answer == {"valid": True, "order": ["load", "gray"], "terminals": ["gray"]}

    @pytest.mark.parametrize("document, because", [
        (as_json({"a": ("load_image", {}), "b": ("invent", {})}), "unknown node"),
        (as_json({"a": ("to_grayscale", {})}), "nothing connected"),
        (as_json({"a": ("load_image", {}), "b": ("draw_boxes", {})},
                 [("a", "image", "b", "detections")]), "takes detections"),
        ("{not json", "not JSON"),
    ])
    def test_a_bad_one_says_what_is_wrong_with_it(self, client, document, because):
        answer = client.post("/workflow/validate", data={"workflow": document}).json()
        assert answer["valid"] is False
        assert because in answer["error"]

    def test_a_cycle_is_caught_here_rather_than_half_way_through_a_run(self, client):
        cycle = as_json(
            {"a": ("to_grayscale", {}), "b": ("to_grayscale", {})},
            [("a", "image", "b", "image"), ("b", "image", "a", "image")])
        assert "cycle" in client.post("/workflow/validate", data={"workflow": cycle}).json()["error"]


class TestRunning:
    """What a run gives back."""

    def test_an_uploaded_image_is_what_the_workflow_runs_on(self, client, payload):
        answer = _run(client, GRAYSCALE, payload)
        assert answer["terminals"] == ["gray"]
        assert answer["results"]["gray"].startswith("data:image/png;base64,")

    def test_an_image_on_disk_needs_no_upload(self, client):
        answer = _run(client, GRAYSCALE, inputs=json.dumps({"image": str(FIXTURE)}))
        assert answer["results"]["gray"].startswith("data:image/png;base64,")

    def test_only_the_ends_come_back_unless_you_ask_for_more(self, client, payload):
        """Encoding an intermediate costs 60 ms and 4 MB a piece; the editor asks, a batch job
        should not have to."""
        assert set(_run(client, GRAYSCALE, payload)["results"]) == {"gray"}
        assert set(_run(client, GRAYSCALE, payload, include="all")["results"]) == {"load", "gray"}

    def test_a_node_that_fails_is_reported_rather_than_answered_with_nothing(self, client):
        """200 with an empty results dict is indistinguishable from a workflow that produced
        nothing, and throws away the reason the node gave."""
        response = client.post("/workflow/run", data={"workflow": GRAYSCALE})
        assert response.status_code == 422
        assert "run(image=...)" in response.json()["detail"]

    def test_a_workflow_that_is_not_one_is_refused_before_anything_runs(self, client):
        response = client.post("/workflow/run", data={"workflow": as_json({"a": ("invent", {})})})
        assert response.status_code == 400
        assert "unknown node" in response.json()["detail"]

    def test_an_override_that_names_nothing_is_a_refusal_not_a_crash(self, client, payload):
        response = client.post("/workflow/run", data={
            "workflow": GRAYSCALE, "inputs": json.dumps({"nonsense": 1})},
            files={"image": ("photo.jpg", payload, "image/jpeg")})
        assert response.status_code == 400
        assert "no parameter" in response.json()["detail"]

    def test_inputs_that_are_not_a_json_object_are_refused(self, client):
        response = client.post("/workflow/run",
                               data={"workflow": GRAYSCALE, "inputs": "[1, 2]"})
        assert response.status_code == 400
        assert "must be a JSON object" in response.json()["detail"]


class TestStreaming:
    """The same run, reported as it happens."""

    def test_each_node_is_reported_starting_and_finishing(self, client, payload):
        events = _stream(client, GRAYSCALE, payload)
        assert [(e.get("node"), e.get("status")) for e in events[:-1]] == [
            ("load", "running"), ("load", "completed"),
            ("gray", "running"), ("gray", "completed")]
        assert events[-1] == {"done": True}

    def test_a_completed_node_carries_its_output_when_it_was_asked_for(self, client, payload):
        """Every node completes; only the ones `include` covers carry a result."""
        events = _stream(client, GRAYSCALE, payload, include="all")
        finished = [e for e in events if e.get("status") == "completed"]
        assert len(finished) == 2
        assert all(e["output"].startswith("data:image/png") for e in finished)

    def test_a_failing_node_is_reported_and_the_stream_stops(self, client):
        events = _stream(client, GRAYSCALE, None)  # nothing to load
        assert events[-1]["status"] == "failed"
        assert "run(image=...)" in events[-1]["error"]
        assert not any(e.get("node") == "gray" for e in events)
        assert not any("done" in event for event in events), (
            "a stream that failed should not also report that it finished")

    def test_only_the_ends_are_sent_unless_you_ask_for_more(self, client, payload):
        """`include` means here what it means on /run. It was declared on neither at one point,
        so the editor's request for every node was accepted and silently discarded."""
        ends = _stream(client, GRAYSCALE, payload)
        everything = _stream(client, GRAYSCALE, payload, include="all")
        assert [e["node"] for e in ends if "output" in e] == ["gray"]
        assert [e["node"] for e in everything if "output" in e] == ["load", "gray"]

    def test_an_override_that_names_nothing_is_refused_before_the_stream_opens(self, client,
                                                                               payload):
        """A stream that has already claimed success is the wrong place to learn it was refused."""
        response = client.post("/workflow/stream", data={
            "workflow": GRAYSCALE, "inputs": json.dumps({"nonsense": 1})},
            files={"image": ("photo.jpg", payload, "image/jpeg")})
        assert response.status_code == 400
        assert "no parameter" in response.json()["detail"]

    def test_a_document_that_is_not_a_workflow_fails_before_the_stream_opens(self, client):
        """A stream that has already claimed success is the wrong place to learn it was refused."""
        response = client.post("/workflow/stream",
                               data={"workflow": as_json({"a": ("invent", {})})})
        assert response.status_code == 400


class TestEveryPortTypeSurvivesJson:
    """Serialising by the declared port, on nodes that need no weights.

    These use the test nodes from ``workflow_nodes`` rather than models, so the branches are
    covered on a machine that has fetched nothing -- and so that ``EMBEDDING`` is covered at all,
    since no shipped node produces one yet. The embedding row is what stops a depth map and an
    embedding being confused: both are two-dimensional float arrays, and asserting the exact list
    fails against the ``{"depth": ...}`` shape a guess would have produced.
    """

    @pytest.mark.parametrize("producer, check", [
        ("fake_depth", lambda v: v["depth"].startswith("data:image/png") and v["max"] > v["min"]),
        ("fake_scores", lambda v: isinstance(v, list) and v[0]["class_name"] == "cat"),
        ("fake_embedding", lambda v: v == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]),
    ])
    def test_a_port_is_sent_as_what_it_declared(self, client, payload, producer, check):
        made = as_json({"load": ("load_image", {}), "out": (producer, {})},
                       [("load", "image", "out", "image")])
        assert check(_run(client, made, payload)["results"]["out"])


class TestSerialisingResults:
    """The same, on real models. Skips without weights."""

    def test_detections_arrive_as_data_rather_than_a_picture(self, client, payload):
        pytest.importorskip("torch")
        require_present("yolov26", "nano")

        found = as_json(
            {"load": ("load_image", {}), "detect": ("yolov26", {"threshold": 0.5})},
            [("load", "image", "detect", "image")])
        answer = _run(client, found, payload)
        found = answer["results"]["detect"]
        assert isinstance(found, list) and found, "detections came back as nothing"
        assert {"bbox", "class_name", "confidence"} <= set(found[0])

    def test_a_depth_map_keeps_the_numbers_that_make_it_a_measurement(self, client, payload):
        require_present("depth_anything_v2", "small")

        far = as_json(
            {"load": ("load_image", {}), "depth": ("depth_anything_v2", {})},
            [("load", "image", "depth", "image")])
        result = _run(client, far, payload)["results"]["depth"]
        assert result["depth"].startswith("data:image/png;base64,")
        assert result["max"] > result["min"], "a depth map without its endpoints is a picture"

    def test_a_batch_arrives_as_a_list(self, client, payload):
        """One image in, several out: the fan-out has to survive the wire too."""
        require_present("yolov26", "nano")

        crops = as_json(
            {"load": ("load_image", {}), "detect": ("yolov26", {}),
             "crop": ("crop_around_detections", {})},
            [("load", "image", "detect", "image"),
             ("load", "image", "crop", "image"),
             ("detect", "detections", "crop", "detections")])
        answer = _run(client, crops, payload)
        assert isinstance(answer["results"]["crop"], list)


class TestTheModelServerIsUnaffected:
    """Mounting the workflow runtime changed nothing about what mozo already served."""

    @pytest.mark.parametrize("path", ["/", "/models", "/test-ui"])
    def test_the_endpoints_that_were_there_still_answer(self, client, path):
        assert client.get(path).status_code == 200

    def test_the_catalogue_still_names_every_family(self, client):
        assert set(client.get("/models").json()) == set(mozo.MODEL_REGISTRY)

    def test_the_workflow_routes_are_all_under_one_prefix(self, client):
        from mozo.server import app

        added = {path for path in _every_path(app.routes) if "workflow" in path}
        assert added == {"/workflow", "/workflow/assets/{name}", "/workflow/nodes",
                         "/workflow/run", "/workflow/stream", "/workflow/validate"}


def _every_path(routes) -> set:
    """Every path in *routes*, reaching into routers that were included rather than flattened.

    Up to FastAPI 0.136 ``include_router`` copied a sub-router's routes into ``app.routes``, so
    every entry there had a ``.path``. Since 0.137 it appends one ``_IncludedRouter`` standing in
    for the whole router, which has no ``.path`` at all -- reading the attribute unconditionally
    raises, and skipping the entries that lack it would find no workflow routes and assert that
    mozo mounts nothing.

    Both shapes are walked, so this asserts where mozo puts its routes rather than which FastAPI
    happens to be installed. The private name is unavoidable: the object is not part of the public
    API, and neither is the flattening it replaced.
    """
    found = set()
    for route in routes:
        path = getattr(route, "path", None)
        if path is not None:
            found.add(path)
        nested = getattr(route, "original_router", None)
        if nested is None:
            nested = getattr(route, "routes", None)
        if nested is not None:
            found |= _every_path(getattr(nested, "routes", nested))
    return found


def _post(client, path: str, workflow: str, payload: bytes = None, **fields):
    """POST a workflow, with an image if there is one."""
    response = client.post(
        f"/workflow/{path}", data={"workflow": workflow, **fields},
        files={"image": ("photo.jpg", payload, "image/jpeg")} if payload else None)
    assert response.status_code == 200, response.text
    return response


def _run(client, workflow: str, payload: bytes = None, **fields) -> dict:
    """POST a workflow to /run and return the parsed answer."""
    return _post(client, "run", workflow, payload, **fields).json()


def _stream(client, workflow: str, payload: bytes | None, **fields) -> list:
    """Run a workflow through the streaming endpoint and collect its events."""
    response = _post(client, "stream", workflow, payload, **fields)
    return [json.loads(line[len("data: "):])
            for line in response.text.splitlines() if line.startswith("data: ")]
