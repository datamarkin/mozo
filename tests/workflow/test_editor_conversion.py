"""The editor's canvas-to-document conversion, run outside a browser.

`toDocument` and `fromDocument` are the only pieces of the editor that decide what mozo receives,
and they are pure functions -- no DOM, no fetch. So they can be exercised with node, and the
document they produce can be fed straight to the server. That covers everything about the editor
except the clicking.

Skipped where node is absent; the editor cannot be built there either.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "ui" / "src" / "lib" / "api.js"

#: A canvas as Svelte Flow holds it: every node typed `custom`, the real kind in `data.nodeType`.
CANVAS = """
import { toDocument, fromDocument } from '%s';

const nodes = [
  { id: 'load-1', type: 'custom', position: { x: 10, y: 20 },
    data: { nodeType: 'load_image', parameters: { image: 'a.jpg' } } },
  { id: 'det-1', type: 'custom', position: { x: 200, y: 20 },
    data: { nodeType: 'yolov26', parameters: { variant: 'seg-nano' } } },
];
const edges = [
  { source: 'load-1', sourceHandle: 'image', target: 'det-1', targetHandle: 'image' },
];

const document = toDocument(nodes, edges);
const back = fromDocument(document, {});
const again = toDocument(back.nodes, back.edges);

console.log(JSON.stringify({
  document,
  stable: JSON.stringify(document) === JSON.stringify(again),
  positions: back.nodes[0].position,
  parameters: back.nodes[1].data.parameters,
  edgeIds: back.edges.map(e => e.id),
}));
""" % API


@pytest.fixture(scope="module")
def converted() -> dict:
    """Run the conversion in node and hand back what it said."""
    if not shutil.which("node"):
        pytest.skip("node is not installed, so the editor cannot be built here either")
    if not API.is_file():
        pytest.skip("the editor source is not present")

    script = ROOT / "ui" / ".conversion-check.mjs"
    script.write_text(CANVAS)
    try:
        finished = subprocess.run(["node", str(script)], capture_output=True, text=True)
    finally:
        script.unlink(missing_ok=True)
    assert finished.returncode == 0, finished.stderr
    return json.loads(finished.stdout)


class TestTheConversion:
    """What the canvas becomes, and what it becomes again."""

    def test_a_canvas_becomes_mozo_s_document_format(self, converted):
        document = converted["document"]
        assert [node["type"] for node in document["nodes"]] == ["load_image", "yolov26"]
        assert document["edges"][0]["sourceHandle"] == "image"

    def test_loading_a_document_and_saving_it_gives_the_same_document(self, converted):
        assert converted["stable"], "a save-load-save cycle changed the file"

    def test_the_canvas_layout_survives(self, converted):
        assert converted["positions"] == {"x": 10, "y": 20}

    def test_the_parameters_survive(self, converted):
        assert converted["parameters"] == {"variant": "seg-nano"}

    def test_every_edge_gets_its_own_id(self, converted):
        ids = converted["edgeIds"]
        assert len(set(ids)) == len(ids) and all(ids)


class TestTheServerAcceptsIt:
    """The document the editor emits, taken at its word by the server."""

    def test_it_is_a_valid_workflow(self, converted, client):
        answer = client.post("/workflow/validate",
                             data={"workflow": json.dumps(converted["document"])}).json()
        assert answer["valid"], answer
        assert answer["order"] == ["load-1", "det-1"]
        assert answer["terminals"] == ["det-1"]
