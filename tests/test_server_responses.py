"""What each family's ``/predict`` response actually contains.

The server is slated for a rewrite, so this covers the parts with real logic rather than the
whole surface: how a depth map crosses an HTTP boundary without losing the measurement, and
that naming a model which does not exist is answered as such. Six of the nine Depth Anything V2
variants predict metres, and an 8-bit PNG -- which is what the old adapter returned -- quantises
those to 256 levels and calls it a picture.
"""

from __future__ import annotations

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


class TestDetectionResponse:
    def test_detections_come_back_as_json(self, client, payload):
        response = post(client, payload, "rfdetr", "nano")
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body and all("bbox" in d and "confidence" in d for d in body)
