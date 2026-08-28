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
        "nodes": [{"id": "load", "type": "load_image", "data": {"parameters": {}}}],
        "edges": [],
    }))
    return path


def test_a_file_given_on_the_command_line_is_what_it_runs_on(workflow):
    result = CliRunner().invoke(cli, ["run", str(workflow), "--file", str(FIXTURE)])
    assert result.exit_code == 0, result.output
    assert "1920x1281 image" in result.output


def test_the_file_is_bound_by_kind_not_by_a_name_written_here(workflow):
    """The parameter is found from its ``Source`` annotation, so this keeps working when it is
    renamed. The literal that used to be here did not."""
    from mozo.workflow import Workflow

    assert Workflow.load(str(workflow)).file_parameter == "image"
    assert CliRunner().invoke(
        cli, ["run", str(workflow), "--file", str(FIXTURE)]).exit_code == 0
