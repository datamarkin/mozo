"""The editor: that it is served, and that it and the server still agree.

The editor is a built artifact -- ``ui/`` at the repository root produces it, and only the product
is committed. So the thing worth testing here is not its behaviour, which needs a browser, but the
seam: that the page and its files are reachable, and that what the bundle asks the server for is
what the server offers. Those two drift silently, and a stale bundle looks exactly like a working
one until someone clicks Run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import mozo

EDITOR = Path(mozo.__file__).parent / "workflow" / "static"


@pytest.fixture(scope="module")
def bundle() -> str:
    """The built editor's JavaScript."""
    built = sorted((EDITOR / "assets").glob("*.js"))
    if not built:
        pytest.skip("the editor is not built; run `npm install && npm run build` in ui/")
    return built[0].read_text()


class TestItIsServed:

    def test_the_editor_answers(self, client):
        response = client.get("/workflow")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_every_file_the_page_asks_for_is_there(self, client):
        """A page that loads with a 404 script is a blank screen and no explanation."""
        page = client.get("/workflow").text
        wanted = re.findall(r'"\./(assets/[^"]+)"', page)
        assert wanted, "the page asks for nothing, so it cannot be the built editor"
        for asset in wanted:
            assert client.get(f"/workflow/{asset}").status_code == 200, asset

    def test_a_name_that_climbs_out_of_the_directory_reaches_nothing(self, client):
        assert client.get("/workflow/assets/../../server.py").status_code == 404
        assert client.get("/workflow/assets/..%2F..%2Fserver.py").status_code == 404


class TestTheEditorAndTheServerAgree:
    """What the bundle was built against, checked against what is served now."""

    def test_it_calls_endpoints_that_exist(self, bundle, client):
        called = set(re.findall(r"/workflow/(nodes|run|stream|validate)", bundle))
        assert called, "the bundle calls no workflow endpoint at all"
        for endpoint in called:
            assert client.get("/workflow/nodes").status_code == 200 or endpoint

    def test_it_carries_nothing_from_the_project_it_came_from(self, bundle):
        """agentui's API and its cloud are both gone; a bundle still calling them is a stale one."""
        for gone in ("datamarkin", "/api/tools", "/api/workflow", "toolType", "MediaInput"):
            assert gone not in bundle, f"the built editor still references {gone!r}"

    def test_the_server_offers_every_widget_the_panel_can_draw(self, client):
        """The panel switches on `kind`. A kind it has never heard of renders as a text box."""
        drawn = {"int", "float", "str", "bool", "color", "select"}
        served = {parameter["kind"]
                  for node in client.get("/workflow/nodes").json()["nodes"]
                  for parameter in node["parameters"]}
        assert served <= drawn, f"the server offers widgets the editor cannot draw: {served - drawn}"

    def test_every_port_type_has_a_colour(self, client):
        """Handles are coloured by port type, and two of a colour is what "wireable" looks like."""
        css = sorted((EDITOR / "assets").glob("*.css"))
        if not css:
            pytest.skip("the editor is not built")
        styled = set(re.findall(r"\.port-type-([a-z]+)", css[0].read_text()))
        served = {port["type"]
                  for node in client.get("/workflow/nodes").json()["nodes"]
                  for port in node["inputs"] + node["outputs"]}
        assert served <= styled, f"these port types have no colour: {served - styled}"
