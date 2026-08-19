"""Does YOLO26 actually work.

Unlike the module tests, these load real checkpoints and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families -q

Two promises are protected here. The artifact you pick must not change the answer -- torch and
ONNX must find the same objects. And mozo must not change the answer either: the vendored package
run directly, with none of mozo between it and the weights, has to agree with what a mozo user
receives. ``tools/verify/yolov26.py`` is the same check as a standalone script over your own
images.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import FIXTURE, as_pixelflow_reports, published, require_weights
from mozo.runtimes import executable, select_runtime
from mozo.weights import WeightsError, artifacts, companions, resolve

#: Where the recorded reference lives, beside the photograph it was recorded on.
FIXTURES = FIXTURE.parent.parent

THRESHOLD = 0.25

ALL = ["nano", "small", "medium", "large", "xlarge"]

#: Everything any variant of this family reports on the fixture photograph, which is a desk scene.
#: The union rather than one variant's answer: capacity differs across five sizes and a larger
#: model legitimately finds more -- everything from small up sees a dining table that nano does
#: not, and large alone sees a chair. What this pins is that the family keeps reading the
#: photograph as this scene, not that every size agrees on its contents. Numeric drift is pinned
#: by TestAgainstRecordedDetections, which is exact.
EXPECTED_NAMES = {"person", "cup", "dining table", "laptop", "cell phone", "chair"}


@pytest.fixture(scope="module")
def predictor_for():
    """Build predictors once per runtime, for one variant at a time.

    Both runtimes of a variant have to coexist for the agreement test, but nothing needs two
    variants at once and xlarge is 115 MB of weights, so changing variant drops the cache.
    """
    from mozo.adapters.yolov26 import YOLOv26Predictor

    cache: dict[str, object] = {}
    loaded = None

    def build(variant: str, runtime: str):
        nonlocal loaded
        if variant != loaded:
            cache.clear()
            loaded = variant
        if runtime not in cache:
            try:
                cache[runtime] = YOLOv26Predictor(variant, device="cpu", runtime=runtime)
            except WeightsError as error:
                pytest.skip(f"yolov26/{variant} weights unavailable: {error}")
        return cache[runtime]

    return build


class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_every_published_variant_carries_its_licence(self, variant):
        """These weights are AGPL-3.0, so the licence has to travel with them, not near them.

        Asked of :func:`~mozo.weights.companions` rather than of :func:`~mozo.weights.artifacts`,
        which omits these on purpose -- they accompany whatever you asked for instead of being a
        thing you can run.
        """
        if not published("yolov26", variant):
            pytest.skip(f"yolov26/{variant} is not in the manifest")

        assert "torch-fp32" in published("yolov26", variant)
        accompanying = companions("yolov26", variant)
        assert "LICENSE" in accompanying, "AGPL-3.0 weights published without their licence text"
        assert "NOTICE" in accompanying, "AGPL-3.0 weights published without a source pointer"

    @pytest.mark.parametrize("variant", ALL)
    def test_a_graph_is_published_with_the_names_it_cannot_carry(self, variant):
        """A graph records no class names, so publishing one without labels leaves ids unnamed."""
        keys = published("yolov26", variant)
        if not [k for k in keys if k.split("-")[0] in {"onnx", "coreml"}]:
            pytest.skip(f"yolov26/{variant} publishes no graph artifact")
        assert "labels" in keys

    @pytest.mark.parametrize("variant", ALL)
    def test_no_coreml_is_published(self, variant):
        """Two separate things stop CoreML for this family, and one of them kills the process.

        The converter refuses the in-graph top-k's gather indices outright. Casting them to int32
        gets past that, and then the Metal compiler aborts with ``MLIR pass manager failed`` --
        an assertion, not an exception, so nothing catches it and the interpreter dies. Off the
        GPU it runs accurately at 22.6 ms against 13.1 ms for torch on MPS, so there is nothing to
        recover either.

        This is the test that keeps ``auto`` safe. :func:`~mozo.runtimes.select_runtime` chooses
        only among what a variant publishes, so *not publishing* is the entire mechanism, and
        publishing one by accident is all it would take to hand Apple-silicon users a crash.
        """
        assert not [k for k in published("yolov26", variant) if k.startswith("coreml")]

    @pytest.mark.parametrize("variant", ALL)
    def test_no_fp16_is_published(self, variant):
        """Faster, and puts the objects it finds in slightly the wrong place.

        Measured on YOLOv8 and recorded in tools/export/yolov8.py, which is the family that has
        a CoreML path to measure it on: fp16 finds every object fp32 finds and puts them in
        slightly the wrong place. Boxes are what a detector is for.
        """
        assert not [k for k in published("yolov26", variant) if k.endswith("fp16")]


class TestDetections:
    @pytest.mark.parametrize("variant", ALL)
    def test_finds_the_scene(self, predictor_for, image, variant):
        require_weights("yolov26", variant)
        detections = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        assert len(detections) > 0
        names = {d.class_name for d in detections}
        assert names <= EXPECTED_NAMES, f"unexpected classes: {names - EXPECTED_NAMES}"
        assert "person" in names

    def test_names_come_from_the_checkpoint(self, predictor_for, image):
        """Read from this family's own weights, not shared with any other family's vocabulary.

        YOLO26 happens to emit the same contiguous COCO ids YOLOv8 does, which is exactly why it
        has to be read rather than assumed: a fine-tuned checkpoint would not, and mozo never
        invents a class name.
        """
        require_weights("yolov26", "nano")
        detections = predictor_for("nano", "torch-fp32").predict(image, threshold=THRESHOLD)
        person = next(d for d in detections if d.class_name == "person")
        assert person.class_id == 0


class TestAgainstRecordedDetections:
    """A reference captured once, so a silent numeric change has something to fail against.

    It carries more here than for the siblings. This family's decode and top-k are inside the
    network rather than in the vendor's post-processing, so there is less shared code between the
    two sides of the vendor comparison and correspondingly more that only a recorded answer sees.

    Every other test here compares mozo to the vendored package, and the two share the
    letterboxing and the coordinate mapping -- so a change to any of those moves
    both sides together and is invisible. On the sibling family that was not hypothetical:
    re-introducing the BGR channel flip the vendor arrived with turned 26 detections into 22 and
    nothing in the suite noticed.

    Recorded from the reshaped vendor, whose pre-processing was verified to produce a tensor
    bit-identical to the harvest's (``max|delta| = 0.0``) and whose parity against the original
    implementation is recorded in the vendor's PROVENANCE.md. Regenerate only when a change to the
    numbers is understood and intended.
    """

    def test_nano_finds_what_it_found_before(self, predictor_for, image):
        require_weights("yolov26", "nano")
        recorded = json.loads((FIXTURES / "yolov26_nano_example.json").read_text())
        got = predictor_for("nano", "torch-fp32").predict(image, threshold=THRESHOLD)

        assert len(got) == len(recorded), "the number of detections changed"
        assert [d.class_id for d in got] == [row["class_id"] for row in recorded]
        assert [d.class_name for d in got] == [row["class_name"] for row in recorded]
        assert [list(d.bbox) for d in got] == [row["bbox"] for row in recorded]
        assert [d.confidence for d in got] == pytest.approx([row["confidence"] for row in recorded])


class TestNoSuppression:
    """The head fires once per object, and the package has no suppression anywhere."""

    def test_the_vendor_exposes_no_suppress(self):
        """If one ever appears, tests/test_vendor_agreement.py stops skipping this family.

        That skip is currently the only thing saying "NMS-free" out loud in the shared suite, and
        it is keyed on this absence. A suppression function added here without updating the family
        would be silently exempted from the invariants every other vendor is held to.
        """
        from mozo.vendors import yolov26_deploy

        assert not hasattr(yolov26_deploy.image, "suppress")

    def test_detect_takes_no_overlap_threshold(self):
        """There is nothing to overlap, so there is no knob -- and no default to get wrong."""
        import inspect

        from mozo.vendors.yolov26_deploy import detect

        parameters = inspect.signature(detect).parameters
        assert "iou" not in parameters and "max_det" not in parameters


class TestChannelOrder:
    """The vendor was BGR when it arrived; mozo is RGB. Nothing raises if that regresses."""

    def test_feeding_bgr_changes_the_answer(self, predictor_for, image):
        require_weights("yolov26", "nano")
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
        require_weights("yolov26", variant)
        from mozo.vendors.yolov26_deploy import Detector

        try:
            checkpoint = resolve("yolov26", variant, "torch-fp32")
        except WeightsError as error:
            pytest.skip(f"yolov26/{variant} weights unavailable: {error}")

        # No overlap or detection cap to pass: this family's network returns the list itself.
        want = Detector(checkpoint, device="cpu").predict(image, conf=THRESHOLD)
        got = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)

        assert len(want) == len(got)
        assert [int(i) for i in want.class_ids] == [d.class_id for d in got]
        assert want.names == [d.class_name for d in got]

        boxes, scores = as_pixelflow_reports(want.boxes, want.scores)
        assert boxes.tolist() == [[float(v) for v in d.bbox] for d in got]
        assert scores.tolist() == pytest.approx([d.confidence for d in got])


class TestRuntimeAgreement:
    """The artifact you pick must not change the answer."""

    @pytest.mark.parametrize("variant", ALL)
    def test_the_published_graph_agrees_with_torch(self, predictor_for, image, variant):
        """ONNX is the only graph this family publishes; see ``test_no_coreml_is_published``."""
        runtime = "onnx-fp32"
        require_weights("yolov26", variant, "torch-fp32")
        require_weights("yolov26", variant, runtime)
        if runtime not in executable(published("yolov26", variant)):
            pytest.skip(f"{runtime} is published but not runnable here")

        torch_out = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        other = predictor_for(variant, runtime).predict(image, threshold=THRESHOLD)

        assert len(torch_out) == len(other)
        assert [d.class_name for d in torch_out] == [d.class_name for d in other]

        # Boxes are truncated to integers, so a sub-pixel float difference between the runtimes
        # can always straddle the boundary and move one edge by one. Anything larger is the model
        # disagreeing with itself. The float-level check lives in tools/export.
        worst = max(
            (max(abs(a - b) for a, b in zip(x.bbox, y.bbox)) for x, y in zip(torch_out, other)),
            default=0,
        )
        assert worst <= 1, f"boxes moved {worst} px between torch-fp32 and {runtime}"

    def test_auto_takes_torch_on_apple_silicon(self):
        """No per-family preference table exists, and this is why none is needed.

        The global table puts CoreML first on ``mps``, which is right for YOLOv8 and would be a
        crash for this family. It never fires here because ``select_runtime`` chooses only among
        published keys and this family publishes no CoreML -- so the absence of the artifact is
        what makes the shared table safe. If CoreML for YOLO26 is ever published, this fails
        alongside ``test_no_coreml_is_published``, and *then* a per-family table is the fix.
        """
        require_weights("yolov26", "nano", "torch-fp32")
        assert select_runtime("cpu", artifacts("yolov26", "nano")) == "torch-fp32"
        assert select_runtime("mps", artifacts("yolov26", "nano")) == "torch-fp32"
        assert select_runtime("cuda", artifacts("yolov26", "nano")) == "torch-fp32"

    def test_a_graph_runtime_takes_its_size_from_the_artifact(self, predictor_for):
        """Letterboxing to anything else would feed the runtime a shape it cannot accept.

        The runner reports ``input_shape``, so the adapter asks rather than assuming -- which is
        also what lets one adapter body serve a runtime it was not written against.
        """
        runtime = "onnx-fp32"
        require_weights("yolov26", "nano", runtime)
        if runtime not in executable(published("yolov26", "nano")):
            pytest.skip(f"{runtime} is published but not runnable here")
        assert predictor_for("nano", runtime).imgsz == 640


class TestCallerSuppliedNames:
    def test_caller_labels_override_the_checkpoint(self, predictor_for, image):
        require_weights("yolov26", "nano")
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
        from mozo.adapters.yolov26 import YOLOv26Predictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY["yolov26"]
        assert entry["adapter_class"] == YOLOv26Predictor.__name__
        assert entry["module"] == "mozo.adapters.yolov26"
        assert set(entry["variants"]) == set(YOLOv26Predictor.VARIANTS)
