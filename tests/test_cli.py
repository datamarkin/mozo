"""The command line, which had no test at all until a change broke it silently.

``mozo run wf.json --image photo.jpg`` wrote the uploaded path to a parameter named ``image``,
which was one node's vocabulary copied into a second place. Renaming that parameter fixed the HTTP
endpoint, left this one raising ``KeyError``, and the suite stayed green because nothing here ran
the command. So what this file holds is the entry point itself, not the engine under it.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from conftest import FIXTURE
from mozo.cli import cli


@pytest.fixture
def workflow(tmp_path):
    """A one-node workflow on disk, with no file chosen, waiting to be given one."""
    path = tmp_path / "wf.json"
    path.write_text(json.dumps({
        "nodes": [{"id": "load", "type": "read_media", "data": {"parameters": {}}}],
        "edges": [],
    }))
    return path


def test_a_file_given_on_the_command_line_is_what_it_runs_on(workflow):
    result = CliRunner().invoke(cli, ["run", str(workflow), "--file", str(FIXTURE)])
    assert result.exit_code == 0, result.output
    assert "1 item in" in result.output


def test_run_is_the_whole_source_not_one_item_of_it(tmp_path):
    """``mozo run`` says why this is the whole source rather than one item of it."""
    photos, out = tmp_path / "photos", tmp_path / "out"
    photos.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (photos / name).write_bytes(FIXTURE.read_bytes())
    document = tmp_path / "wf.json"
    document.write_text(json.dumps({
        "nodes": [{"id": "load", "type": "read_media",
                   "data": {"parameters": {"source": str(photos)}}},
                  {"id": "save", "type": "save_image",
                   "data": {"parameters": {"path": str(out)}}}],
        "edges": [{"source": "load", "sourceHandle": "image",
                   "target": "save", "targetHandle": "image"}]}))

    result = CliRunner().invoke(cli, ["run", str(document)])
    assert result.exit_code == 0, result.output
    assert "3 items in" in result.output
    assert sorted(p.name for p in out.iterdir()) == ["a.jpg", "b.jpg", "c.jpg"]


def test_test_describes_every_node_for_one_item(workflow):
    """The older behaviour, kept under its own name: what you want before pointing it at 10,000."""
    result = CliRunner().invoke(cli, ["run", str(workflow), "--file", str(FIXTURE), "--test"])
    assert result.exit_code == 0, result.output
    assert "1920x1281 image" in result.output


def test_the_file_is_bound_by_kind_not_by_a_name_written_here(workflow):
    """The parameter is found from its ``Source`` annotation, so this keeps working when it is
    renamed. The literal that used to be here did not."""
    from mozo.workflow import Workflow

    assert Workflow.load(str(workflow)).file_parameter == "source"
    assert CliRunner().invoke(
        cli, ["run", str(workflow), "--file", str(FIXTURE)]).exit_code == 0
