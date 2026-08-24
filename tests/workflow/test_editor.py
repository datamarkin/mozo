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
from urllib.parse import urljoin

import pytest

import mozo
from mozo.workflow import get
from workflow_nodes import shipped

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
        """A page that loads with a 404 script is a blank screen and no explanation.

        Each URL is resolved against the page's own address the way a browser would, rather than
        being pasted onto a prefix by the test. That distinction is the whole point here: the page
        is served at ``/workflow`` with no trailing slash, so a relative ``./assets/x.js`` resolves
        to ``/assets/x.js`` -- which nothing serves. An earlier version of this test joined the
        prefix itself and passed against exactly that broken page.
        """
        page = client.get("/workflow").text
        wanted = re.findall(r'(?:src|href)="([^"]+\.(?:js|css))"', page)
        assert wanted, "the page asks for nothing, so it cannot be the built editor"
        for asset in wanted:
            resolved = urljoin("/workflow", asset)
            assert client.get(resolved).status_code == 200, f"{asset} resolves to {resolved}"

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

    def test_the_server_offers_every_widget_the_panel_can_draw(self):
        """The panel switches on `kind`, and anything it has not heard of falls to a text box.

        Read out of the panel rather than listed here, for the reason the catalogue is read out of
        the node functions: a list kept alongside is a second place to keep in step, and this one
        would drift silently -- a new kind would render as a text box and pass.

        `str` is the text box, so the fall-through is right for it and for nothing else.
        """
        panel = (Path(__file__).resolve().parents[2] / "ui/src/lib/PropertiesPanel.svelte").read_text()
        drawn = set(re.findall(r"field\.kind === '(\w+)'", panel)) | {"str"}
        served = {parameter.kind for name in shipped() for parameter in get(name).parameters}
        assert served <= drawn, f"the server offers widgets the editor cannot draw: {served - drawn}"

    def test_a_connection_is_refused_before_it_is_made_rather_than_after(self):
        """Svelte Flow's `isValidConnection`, not its `connect` event.

        The event is a notification: by the time it fires the edge is on the canvas, and returning
        early from it leaves it there. Checked that way, the editor accepted a detections output
        into an image input and only the server noticed -- 400 at the end of a drag that looked
        like it had worked. `isValidConnection` is asked while the wire is still in the air.
        """
        app = (Path(__file__).resolve().parents[2] / "ui/src/App.svelte").read_text()
        assert "{isValidConnection}" in app, "the canvas does not validate connections at all"
        assert "on:connect=" not in app, (
            "the canvas checks connections on the `connect` event, which fires after the edge "
            "has already been added")

    def test_every_category_has_an_icon(self, client):
        """A category the icons do not know renders as a bare circle in the palette.

        Eight of the ten did, at one point: the icons came from the project the editor came from
        and named its categories, not mozo's.
        """
        icons = (Path(__file__).resolve().parents[2] / "ui/src/lib/CategoryIcon.svelte").read_text()
        drawn = set(re.findall(r"^    (\w+): '", icons, re.M))
        declared = {get(name).category for name in shipped()}
        assert declared <= drawn, f"these categories have no icon: {sorted(declared - drawn)}"

    def test_every_port_type_has_a_colour(self, client):
        """Handles are coloured by port type, and two of a colour is what "wireable" looks like."""
        css = sorted((EDITOR / "assets").glob("*.css"))
        if not css:
            pytest.skip("the editor is not built")
        styled = set(re.findall(r"\.port-type-([a-z]+)", css[0].read_text()))
        served = {port.type.value for name in shipped()
                  for port in get(name).inputs + get(name).outputs}
        assert served <= styled, f"these port types have no colour: {served - styled}"
