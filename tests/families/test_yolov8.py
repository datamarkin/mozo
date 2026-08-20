"""Does YOLOv8 actually work.

Unlike the module tests, these load real checkpoints and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families -q

Two promises are protected here. The artifact you pick must not change the answer -- torch and
ONNX must find the same objects. And mozo must not change the answer either: the vendored package
run directly, with none of mozo between it and the weights, has to agree with what a mozo user
receives. ``tools/verify/yolov8.py`` is the same check as a standalone script over your own images.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import COORD_STEP, FIXTURE, as_pixelflow_reports, published, require_weights
from mozo.runtimes import executable, select_runtime
from mozo.weights import WeightsError, artifacts, companions, resolve

#: Where the recorded reference lives, beside the photograph it was recorded on.
FIXTURES = FIXTURE.parent.parent

THRESHOLD = 0.25

#: What a graph may move against torch: what the YOLO exporter already held it to at full
#: precision, plus one step of PixelFlow's rounding on top.
BOX_TOLERANCE = 1e-2 + COORD_STEP

ALL = ["nano", "small", "medium", "large", "xlarge"]

#: Everything any variant of this family reports on the fixture photograph, which is a desk scene.
#: The union rather than one variant's answer: capacity differs across five sizes and a larger
#: model legitimately finds more -- xlarge sees a clock that nano does not. What this pins is that
#: the family keeps reading the photograph as this scene, not that every size agrees on its
#: contents. Numeric drift is pinned by TestAgainstRecordedDetections, which is exact.
EXPECTED_NAMES = {"person", "cup", "dining table", "laptop", "cell phone", "clock"}


@pytest.fixture(scope="module")
def predictor_for():
    """Build predictors once per runtime, for one variant at a time.

    Both runtimes of a variant have to coexist for the agreement test, but nothing needs two
    variants at once and xlarge is 137 MB of weights, so changing variant drops the cache. The
    parametrized classes therefore rebuild each variant once per class -- about 2.6 s across the
    file, which is the price of not holding five models.
    """
    from mozo.adapters.yolov8 import YOLOv8Predictor

    cache: dict[str, object] = {}
    loaded = None

    def build(variant: str, runtime: str):
        nonlocal loaded
        if variant != loaded:
            cache.clear()
            loaded = variant
        if runtime not in cache:
            try:
                cache[runtime] = YOLOv8Predictor(variant, device="cpu", runtime=runtime)
            except WeightsError as error:
                pytest.skip(f"yolov8/{variant} weights unavailable: {error}")
        return cache[runtime]

    return build


class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_every_published_variant_carries_its_licence(self, variant):
        """These weights are AGPL-3.0, so the licence has to travel with them, not near them.

        Asked of :func:`~mozo.weights.companions` rather than of :func:`~mozo.weights.artifacts`,
        which omits these on purpose -- they accompany whatever you asked for instead of being a
        thing you can run.

        Asserted per family rather than in ``tools/generate_manifest.py`` because it is a fact
        about *these* terms: the manifest generator requires a LICENSE of everyone, and a NOTICE
        of no one, since a model may have nothing to attribute. AGPL-3.0 does have something to
        attribute -- the source these weights correspond to.
        """
        if not published("yolov8", variant):
            pytest.skip(f"yolov8/{variant} is not in the manifest")

        assert "torch-fp32" in published("yolov8", variant)
        accompanying = companions("yolov8", variant)
        assert "LICENSE" in accompanying, "AGPL-3.0 weights published without their licence text"
        assert "NOTICE" in accompanying, "AGPL-3.0 weights published without a source pointer"

    @pytest.mark.parametrize("variant", ALL)
    def test_a_graph_is_published_with_the_names_it_cannot_carry(self, variant):
        """A graph records no class names, so publishing one without labels leaves ids unnamed."""
        keys = published("yolov8", variant)
        if not [k for k in keys if k.split("-")[0] in {"onnx", "coreml"}]:
            pytest.skip(f"yolov8/{variant} publishes no graph artifact")
        assert "labels" in keys

    @pytest.mark.parametrize("variant", ALL)
    def test_no_fp16_is_published(self, variant):
        """Measured across four variants and a loss every time, so mozo does not ship it.

        torch fp16 on MPS is *slower* than fp32 and moves boxes 0.76 px; ONNX fp16 is slower too.
        CoreML fp16 is genuinely faster, about 1.4x, and costs 1.4 to 7.4 px depending on variant
        -- it finds every object fp32 finds, and puts them in slightly the wrong place. Boxes are
        what a detector is for, so mozo does not publish it. See tools/export/yolov8.py, whose
        docstring also records how an earlier and much worse-looking set of numbers was wrong.
        """
        assert not [k for k in published("yolov8", variant) if k.endswith("fp16")]


class TestDetections:
    @pytest.mark.parametrize("variant", ALL)
    def test_finds_the_scene(self, predictor_for, image, variant):
        require_weights("yolov8", variant)
        detections = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        assert len(detections) > 0
        names = {d.class_name for d in detections}
        assert names <= EXPECTED_NAMES, f"unexpected classes: {names - EXPECTED_NAMES}"
        assert "person" in names

    def test_names_come_from_the_checkpoint(self, predictor_for, image):
        """YOLOv8 emits contiguous COCO ids, where RF-DETR emits the original sparse ones.

        Nothing derives one from the other: each family's names are whatever its own weights
        recorded. If mozo ever defaulted to a shared COCO list, one of these two would be wrong.
        """
        require_weights("yolov8", "nano")
        detections = predictor_for("nano", "torch-fp32").predict(image, threshold=THRESHOLD)
        person = next(d for d in detections if d.class_name == "person")
        assert person.class_id == 0


class TestAgainstRecordedDetections:
    """A reference captured once, so a silent numeric change has something to fail against.

    Every other test here compares mozo to the vendored package, and the two share the
    letterboxing, the suppression and the coordinate mapping -- so a change to any of those moves
    both sides together and is invisible. That is not hypothetical: re-introducing the BGR channel
    flip the vendor arrived with turns 26 detections into 22 on this photograph, and before this
    test existed nothing in the suite noticed, including ``tools/verify/yolov8.py``.

    The recorded values come from the state that was verified detection-for-detection against the
    package exactly as it was harvested, whose own parity against the original implementation is
    recorded in the vendor's PROVENANCE.md. Regenerate them only when a change to the numbers is
    understood and intended.
    """

    def test_nano_finds_what_it_found_before(self, predictor_for, image):
        require_weights("yolov8", "nano")
        recorded = json.loads((FIXTURES / "yolov8_nano_example.json").read_text())
        got = predictor_for("nano", "torch-fp32").predict(image, threshold=THRESHOLD)

        assert len(got) == len(recorded), "the number of detections changed"
        assert [d.class_id for d in got] == [row["class_id"] for row in recorded]
        assert [d.class_name for d in got] == [row["class_name"] for row in recorded]
        assert [list(d.bbox) for d in got] == [row["bbox"] for row in recorded]
        assert [d.confidence for d in got] == pytest.approx([row["confidence"] for row in recorded])


class TestChannelOrder:
    """The vendor was BGR when it arrived; mozo is RGB. Nothing raises if that regresses."""

    def test_feeding_bgr_changes_the_answer(self, predictor_for, image):
        require_weights("yolov8", "nano")
        model = predictor_for("nano", "torch-fp32")
        rgb = model.predict(image, threshold=THRESHOLD)
        bgr = model.predict(np.ascontiguousarray(image[..., ::-1]), threshold=THRESHOLD)
        assert [d.bbox for d in rgb] != [d.bbox for d in bgr], (
            "RGB and BGR gave the same detections, so this test proves nothing about either"
        )


class TestMatchesTheVendor:
    """mozo must return what the vendored package returns, not merely something like it."""

    @pytest.mark.parametrize("variant", ALL)
    def test_mozo_reports_exactly_what_the_vendor_found(self, predictor_for, image, variant):
        require_weights("yolov8", variant)
        from mozo.vendors.yolov8_deploy import Detector

        try:
            checkpoint = resolve("yolov8", variant, "torch-fp32")
        except WeightsError as error:
            pytest.skip(f"yolov8/{variant} weights unavailable: {error}")

        want = Detector(checkpoint, device="cpu").predict(image, conf=THRESHOLD, iou=0.7, max_det=300)
        got = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)

        assert len(want) == len(got)
        assert [int(i) for i in want.class_ids] == [d.class_id for d in got]
        assert want.names == [d.class_name for d in got]

        boxes, scores = as_pixelflow_reports(want.boxes, want.scores, want.class_ids)
        assert boxes.tolist() == [[float(v) for v in d.bbox] for d in got]
        assert scores.tolist() == pytest.approx([d.confidence for d in got])


class TestRuntimeAgreement:
    """The artifact you pick must not change the answer."""

    @pytest.mark.parametrize("variant", ALL)
    @pytest.mark.parametrize("runtime", ["onnx-fp32", "coreml-fp32"])
    def test_every_published_graph_agrees_with_torch(self, predictor_for, image, variant, runtime):
        """Parametrized over runtimes rather than naming one, so a new artifact is covered by
        publishing it rather than by remembering to write another test."""
        require_weights("yolov8", variant, "torch-fp32")
        require_weights("yolov8", variant, runtime)
        if runtime not in executable(published("yolov8", variant)):
            pytest.skip(f"{runtime} is published but not runnable here")

        torch_out = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        other = predictor_for(variant, runtime).predict(image, threshold=THRESHOLD)

        assert len(torch_out) == len(other)
        assert [d.class_name for d in torch_out] == [d.class_name for d in other]

        # A rounding boundary can move one edge by a step of PixelFlow's coordinate precision;
        # anything larger is the model disagreeing with itself. The float-level check, before any
        # rounding, lives in tools/export.
        worst = max(
            (max(abs(a - b) for a, b in zip(x.bbox, y.bbox)) for x, y in zip(torch_out, other)),
            default=0,
        )
        assert worst <= BOX_TOLERANCE, f"boxes moved {worst} px between torch-fp32 and {runtime}"

    def test_auto_picks_the_runtime_that_was_measured_fastest(self):
        """Asserted against the real manifest rather than by loading duplicate predictors.

        CoreML on Apple silicon is 8-12x torch CPU and 1.2-1.9x torch MPS at 0.0004 px, which is
        why it leads there; ONNX beats torch on CPU for nano and loses for every larger variant,
        so the shared preference table keeps torch first.
        """
        require_weights("yolov8", "nano", "torch-fp32")
        assert select_runtime("cpu", artifacts("yolov8", "nano")) == "torch-fp32"
        if "coreml-fp32" in executable(published("yolov8", "nano")):
            assert select_runtime("mps", artifacts("yolov8", "nano")) == "coreml-fp32"

    @pytest.mark.parametrize("runtime", ["onnx-fp32", "coreml-fp32"])
    def test_a_graph_runtime_takes_its_size_from_the_artifact(self, predictor_for, runtime):
        """Letterboxing to anything else would feed the runtime a shape it cannot accept.

        Both runners report ``input_shape``, so the adapter asks rather than assuming, and does
        not need to know which kind of artifact it is holding.
        """
        require_weights("yolov8", "nano", runtime)
        if runtime not in executable(published("yolov8", "nano")):
            pytest.skip(f"{runtime} is published but not runnable here")
        assert predictor_for("nano", runtime).imgsz == 640


class TestCallerSuppliedNames:
    def test_caller_labels_override_the_checkpoint(self, predictor_for, image):
        require_weights("yolov8", "nano")
        detections = predictor_for("nano", "torch-fp32").predict(
            image, threshold=THRESHOLD, labels={0: "human"}
        )
        assert any(d.class_name == "human" for d in detections)
        assert not any(d.class_name == "person" for d in detections)


class TestRegistry:
    def test_registry_agrees_with_the_adapter(self):
        """The variant list is written twice on purpose, so something has to hold it together.

        mozo.registry must answer /models without importing an adapter, and every adapter pulls
        torch in -- so the registry cannot derive its list from the adapter. This test is what
        makes the duplication safe rather than a latent drift.
        """
        from mozo.adapters.yolov8 import YOLOv8Predictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY["yolov8"]
        assert entry["adapter_class"] == YOLOv8Predictor.__name__
        assert entry["module"] == "mozo.adapters.yolov8"
        assert set(entry["variants"]) == set(YOLOv8Predictor.VARIANTS)
