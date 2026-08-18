"""Does RF-DETR actually work.

Unlike the module tests, these load real checkpoints and run real inference, so they are slow
and need the published artifacts. They are skipped rather than failed when those are absent --
a laptop without 3 GB of weights should still be able to run the rest of the suite.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families -q

What they protect is the promise that the artifact you pick does not change the answer: torch
and ONNX must return the same detections, with the same names, for every variant.
"""

from __future__ import annotations

import cv2
import pytest

from mozo.runtimes import select_runtime
from mozo.weights import WeightsError, artifacts

FIXTURE = "tests/fixtures/images/example.jpg"
THRESHOLD = 0.5

DETECTION = ["nano", "small", "medium", "large"]
SEGMENTATION = ["seg-nano", "seg-small", "seg-medium", "seg-large"]
ALL = DETECTION + SEGMENTATION

#: The scene in the fixture, as ids in COCO's original space. A model that reads this photograph
#: as anything else has either regressed or been given the wrong vocabulary.
EXPECTED_NAMES = {"person", "cup", "dining table", "laptop", "cell phone"}


def _published(variant: str) -> list[str]:
    try:
        return artifacts("rfdetr", variant)
    except WeightsError:
        return []


def _require(variant: str, runtime: str) -> None:
    if runtime not in _published(variant):
        pytest.skip(f"rfdetr/{variant} does not publish {runtime}")


@pytest.fixture(scope="module")
def image():
    return cv2.imread(FIXTURE)


@pytest.fixture(scope="module")
def predictor_for():
    """Build predictors once per (variant, runtime) -- loading a checkpoint is the slow part."""
    from mozo.adapters.rfdetr import RFDETRPredictor

    cache: dict[tuple[str, str], object] = {}

    def build(variant: str, runtime: str):
        if (variant, runtime) not in cache:
            cache[(variant, runtime)] = RFDETRPredictor(variant, device="cpu", runtime=runtime)
        return cache[(variant, runtime)]

    return build


class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_every_variant_publishes_torch_and_onnx(self, variant):
        published = _published(variant)
        if not published:
            pytest.skip(f"rfdetr/{variant} is not in the manifest")
        assert "torch-fp32" in published
        assert "onnx-fp32" in published
        assert "labels" in published


class TestDetections:
    @pytest.mark.parametrize("variant", ALL)
    def test_finds_the_scene(self, predictor_for, image, variant):
        _require(variant, "torch-fp32")
        detections = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        assert len(detections) > 0
        names = {d.class_name for d in detections}
        assert names <= EXPECTED_NAMES, f"unexpected classes: {names - EXPECTED_NAMES}"
        assert "person" in names

    @pytest.mark.parametrize("variant", ALL)
    def test_names_come_from_the_published_vocabulary(self, predictor_for, image, variant):
        """RF-DETR emits COCO's original ids. The contiguous list would say "bicycle" here."""
        _require(variant, "torch-fp32")
        detections = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        person = next(d for d in detections if d.class_name == "person")
        assert person.class_id == 1

    @pytest.mark.parametrize("variant", SEGMENTATION)
    def test_segmentation_variants_return_masks(self, predictor_for, image, variant):
        _require(variant, "torch-fp32")
        detections = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        assert detections[0].masks is not None
        assert detections[0].masks[0].shape[:2] == image.shape[:2]

    @pytest.mark.parametrize("variant", DETECTION)
    def test_detection_variants_return_no_masks(self, predictor_for, image, variant):
        _require(variant, "torch-fp32")
        assert predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)[0].masks is None


class TestRuntimeAgreement:
    """The artifact you pick must not change the answer."""

    @pytest.mark.parametrize("variant", ALL)
    def test_torch_and_onnx_agree(self, predictor_for, image, variant):
        _require(variant, "torch-fp32")
        _require(variant, "onnx-fp32")

        torch_out = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        onnx_out = predictor_for(variant, "onnx-fp32").predict(image, threshold=THRESHOLD)

        assert len(torch_out) == len(onnx_out)
        assert [d.class_name for d in torch_out] == [d.class_name for d in onnx_out]

        # Boxes are stored as integers, so a sub-pixel float difference between the runtimes
        # can always straddle a rounding boundary and move one edge by one. Anything larger is
        # the model disagreeing with itself. The float-level check lives in tools/export.
        worst = max(
            (max(abs(a - b) for a, b in zip(x.bbox, y.bbox)) for x, y in zip(torch_out, onnx_out)),
            default=0,
        )
        assert worst <= 1, f"boxes moved {worst} px between runtimes"

    def test_auto_picks_torch(self):
        """Asserted against the real manifest rather than by loading a duplicate predictor."""
        _require("small", "torch-fp32")
        assert select_runtime("cpu", artifacts("rfdetr", "small")) == "torch-fp32"


class TestCallerSuppliedNames:
    def test_caller_labels_override_the_published_ones(self, predictor_for, image):
        _require("small", "torch-fp32")
        detections = predictor_for("small", "torch-fp32").predict(
            image, threshold=THRESHOLD, labels={1: "human"}
        )
        assert any(d.class_name == "human" for d in detections)
        assert not any(d.class_name == "person" for d in detections)


class TestAgreesWithUpstream:
    """Cross-checks against the ``rfdetr`` package, when it happens to be installed.

    Everything else in this suite compares mozo against itself, which cannot catch mozo being
    consistently wrong. These compare against the implementation the weights were published for.
    """

    @pytest.fixture(scope="class")
    def upstream(self):
        rfdetr = pytest.importorskip("rfdetr", reason="upstream comparison needs the rfdetr package")
        return rfdetr.RFDETRSmall(device="cpu")

    def test_preprocessing_matches_upstream_on_a_heavy_downscale(self, upstream, predictor_for):
        """Resizing must not antialias, because upstream's does not.

        This is the one preprocessing choice that changes results rather than rounding them: on a
        2000px photograph downscaled to 384, antialiasing turns 81 detections into 56.

        The version of ``rfdetr`` installed matters. Releases before 1.7 antialiased, so testing
        against an old one asserts the opposite of the truth -- convincingly, because everything
        matches to four decimals right up until the images get large. The vendor was extracted
        from 1.10.0.dev; a baseline older than that is not a baseline.
        """
        _require("small", "torch-fp32")
        from PIL import Image

        image = Image.open(FIXTURE).convert("RGB")
        want = upstream.predict(image, threshold=0.1)
        got = predictor_for("small", "torch-fp32").predict(cv2.imread(FIXTURE), threshold=0.1)

        assert len(got) == len(want.xyxy)
        assert abs(float(max(want.confidence)) - max(d.confidence for d in got)) < 0.01
