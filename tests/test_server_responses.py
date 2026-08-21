"""What each family's ``/predict`` response actually contains.

The server is slated for a rewrite, so this covers the parts with real logic rather than the
whole surface: how a depth map crosses an HTTP boundary without losing the measurement, and
that naming a model which does not exist is answered as such. Six of the nine Depth Anything V2
variants predict metres, and an 8-bit PNG -- which is what the old adapter returned -- quantises
those to 256 levels and calls it a picture.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
import pytest

from conftest import FIXTURE, require_weights


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from mozo.server import app

    with TestClient(app) as running:
        yield running


def post(client, payload, family: str, variant: str):
    """Ask for a prediction, skipping only when the weights were never published."""
    require_weights(family, variant)
    response = client.post(f"/predict/{family}/{variant}",
                           files={"file": ("image.jpg", payload, "image/jpeg")})
    assert response.status_code == 200, response.json().get("detail", response.text)[:200]
    return response


def png(response) -> np.ndarray:
    """The response body decoded as an image, at whatever depth it was encoded."""
    return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)


@pytest.fixture(scope="module")
def depth_response(client, payload):
    """One metric-depth request, reused. Each one costs an inference plus a ~40 ms PNG encode."""
    return post(client, payload, "depth_anything_v2", "indoor-small")


def decode(response) -> np.ndarray:
    """Recover the original depth map from the response, as a client would."""
    low = float(response.headers["X-Depth-Min"])
    high = float(response.headers["X-Depth-Max"])
    return low + png(response).astype(np.float64) / 65535.0 * (high - low)


class TestDepthResponse:
    def test_metric_depth_survives_the_round_trip(self, client, depth_response):
        assert depth_response.headers["content-type"].startswith("image/png")
        assert depth_response.headers["X-Depth-Unit"] == "metres"

        # Ground truth from the model the server already has loaded, rather than a second copy
        # of the same 99 MB checkpoint.
        model = client.app.state.model_manager.get_model("depth_anything_v2", "indoor-small")
        truth = model.predict(FIXTURE)

        # 16 bits over the map's own range: sub-millimetre, against depths of metres.
        assert np.abs(decode(depth_response) - truth).max() < 1e-3

    def test_the_png_is_sixteen_bit(self, depth_response):
        assert png(depth_response).dtype == np.uint16

    def test_a_relative_variant_is_not_called_metres(self, client, payload):
        """The transport does not invent a unit the model never claimed."""
        response = post(client, payload, "depth_anything_v2", "small")
        assert response.headers["X-Depth-Unit"] == "none"


class TestUnknownModels:
    """Naming something that does not exist is a 404, not a server error.

    Five families were removed from mozo; requests for them must read as "no such model" rather
    than "mozo broke", and the same holds for a variant a surviving family does not have. The
    registry settles all of this before anything is decoded or loaded.
    """

    @pytest.mark.parametrize("path", [
        "/predict/paddleocr/mobile",           # a removed family
        "/predict/florence2/ocr",              # another
        "/predict/rfdetr/does-not-exist",      # a real family, an invented variant
        "/predict/depth_anything_v2/giant",    # ditto -- upstream never published vitg
    ])
    def test_unknown_family_or_variant_is_404(self, client, payload, path):
        response = client.post(path, files={"file": ("image.jpg", payload, "image/jpeg")})
        assert response.status_code == 404
        assert "detail" in response.json()

    def test_an_unknown_model_is_rejected_before_the_image_is_decoded(self, client):
        """Which is why the body here is not an image at all, and still gets a 404."""
        response = client.post("/predict/paddleocr/mobile",
                               files={"file": ("image.jpg", b"not an image", "image/jpeg")})
        assert response.status_code == 404


class TestCatalogue:
    """The shape ``/models`` promises, because something already reads it.

    ``mozo/static/test_ui.html`` builds its dropdowns from this response. Dropping a field it
    reads renders "undefined variants" in the browser and nothing anywhere fails -- which is
    what happened to ``num_variants``. Pinning the fields the page actually uses is cheaper
    than finding out by looking.
    """

    def test_every_family_carries_what_the_page_reads(self, client):
        from mozo.registry import MODEL_REGISTRY

        body = client.get("/models").json()
        assert set(body) == set(MODEL_REGISTRY)
        for family, info in body.items():
            assert info["task_type"] and info["description"]
            assert isinstance(info["variants"], list) and info["variants"]

    def test_the_catalogue_matches_the_registry(self, client):
        """It is a projection of the registry, so it must not drift from one."""
        from mozo.registry import MODEL_REGISTRY

        body = client.get("/models").json()
        for family, entry in MODEL_REGISTRY.items():
            assert body[family]["variants"] == entry["variants"]

    def test_residency_is_reported_by_its_own_endpoint(self, client):
        body = client.get("/models/loaded").json()
        assert body["models"] == client.app.state.model_manager.loaded()

    def test_the_catalogue_costs_no_model_loads(self, client):
        """It is answered from the registry alone -- no adapter import, no torch, no weights."""
        before = client.app.state.model_manager.loaded()
        client.get("/models")
        assert client.app.state.model_manager.loaded() == before


class TestDetectionResponse:
    def test_detections_come_back_as_json(self, client, payload):
        response = post(client, payload, "rfdetr", "nano")
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body and all("bbox" in d and "confidence" in d for d in body)


def test_every_registered_task_type_has_a_dispatch_arm():
    """The endpoint encodes one task per arm, and a task with no arm returns 501 at request
    time. Nothing else holds the two in step: a family can be registered, tested and shipped
    while being unserveable, and the first person to find out is a caller. This is the guard
    that made ``text_recognition`` a two-line change rather than a two-line change plus a bug.
    """
    import inspect

    from mozo.registry import MODEL_REGISTRY
    from mozo import server

    source = inspect.getsource(server.predict)
    registered = {entry["task_type"] for entry in MODEL_REGISTRY.values()}
    # PROMPTED is dispatched as a set rather than by name, so its members are served by the
    # ``task in PROMPTED`` arm and will not appear as string literals.
    served = set(re.findall(r'task == "([a-z_]+)"', source)) | server.PROMPTED

    assert registered <= served, (
        f"registered but not served: {sorted(registered - served)}. Add an arm to "
        f"mozo.server.predict, or the endpoint answers 501 for it."
    )
