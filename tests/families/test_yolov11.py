"""Does YOLO11 actually work.

Unlike the module tests, these load real checkpoints and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families -q

Two promises are protected here. The artifact you pick must not change the answer -- torch and
ONNX must find the same objects. And mozo must not change the answer either: the vendored package
run directly, with none of mozo between it and the weights, has to agree with what a mozo user
receives. ``tools/verify/yolov11.py`` is the same check as a standalone script over your own
images.
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

#: The segmentation variants: the same backbone and neck with a ``Segment`` head. Listed apart
#: from :data:`ALL` because most of this file asks about detection, which they also do -- what
#: they add is a mask, and only :class:`TestSegmentation` is about that.
SEGMENTING = ["seg-nano", "seg-small", "seg-medium", "seg-large", "seg-xlarge"]

#: Everything any variant of this family reports on the fixture photograph, which is a desk scene.
#: The union rather than one variant's answer: capacity differs across five sizes and a larger
#: model legitimately finds more -- everything from small up sees a dining table that nano does
#: not. What this pins is that the family keeps reading the photograph as this scene, not that
#: every size agrees on its contents. Numeric drift is pinned by TestAgainstRecordedDetections,
#: which is exact.
EXPECTED_NAMES = {"person", "cup", "dining table", "laptop", "cell phone"}


def vendor_answer(variant: str, image):
    """What the vendored package returns for *variant*, or skip if its weights are absent.

    The resolve-and-skip preamble is here rather than in each test because three of them need the
    same four lines, and a test that quietly stopped skipping would look like a pass.
    """
    from mozo.vendors.yolov11_deploy import Detector

    return Detector(vendor_checkpoint(variant), device="cpu").predict(
        image, conf=THRESHOLD, iou=0.7, max_det=300)


def vendor_checkpoint(variant: str):
    """The published torch checkpoint for *variant*, or skip if it is not there."""
    require_weights("yolov11", variant)
    try:
        return resolve("yolov11", variant, "torch-fp32")
    except WeightsError as error:
        pytest.skip(f"yolov11/{variant} weights unavailable: {error}")


@pytest.fixture(scope="module")
def predictor_for():
    """Build predictors once per runtime, for one variant at a time.

    Both runtimes of a variant have to coexist for the agreement test, but nothing needs two
    variants at once and xlarge is 115 MB of weights, so changing variant drops the cache.
    """
    from mozo.adapters.yolov11 import YOLOv11Predictor

    cache: dict[str, object] = {}
    loaded = None

    def build(variant: str, runtime: str):
        nonlocal loaded
        if variant != loaded:
            cache.clear()
            loaded = variant
        if runtime not in cache:
            try:
                cache[runtime] = YOLOv11Predictor(variant, device="cpu", runtime=runtime)
            except WeightsError as error:
                pytest.skip(f"yolov11/{variant} weights unavailable: {error}")
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
        if not published("yolov11", variant):
            pytest.skip(f"yolov11/{variant} is not in the manifest")

        assert "torch-fp32" in published("yolov11", variant)
        accompanying = companions("yolov11", variant)
        assert "LICENSE" in accompanying, "AGPL-3.0 weights published without their licence text"
        assert "NOTICE" in accompanying, "AGPL-3.0 weights published without a source pointer"

    @pytest.mark.parametrize("variant", ALL)
    def test_a_graph_is_published_with_the_names_it_cannot_carry(self, variant):
        """A graph records no class names, so publishing one without labels leaves ids unnamed."""
        keys = published("yolov11", variant)
        if not [k for k in keys if k.split("-")[0] in {"onnx", "coreml"}]:
            pytest.skip(f"yolov11/{variant} publishes no graph artifact")
        assert "labels" in keys

    @pytest.mark.parametrize("variant", ALL)
    def test_no_coreml_is_published(self, variant):
        """CoreML does not merely underperform for this family -- it takes the process down.

        Converting the full network yields a package whose first prediction fails inside Apple's
        Metal graph compiler with ``MLIR pass manager failed``, and that is an assertion, not an
        exception: nothing catches it and the interpreter dies. It was bisected to layer 10, the
        ``C2PSA`` attention block. The configuration that does work -- CPU and Neural Engine only
        -- is accurate but slower than torch on MPS, so there is nothing to recover.

        This is the test that keeps ``auto`` safe. :func:`~mozo.runtimes.select_runtime` chooses
        only among what a variant publishes, so *not publishing* is the entire mechanism, and
        publishing one by accident is all it would take to hand Apple-silicon users a crash.
        """
        assert not [k for k in published("yolov11", variant) if k.startswith("coreml")]

    @pytest.mark.parametrize("variant", ALL)
    def test_no_fp16_is_published(self, variant):
        """Faster, and puts the objects it finds in slightly the wrong place.

        Measured on the sibling family by pairing detections by IoU at a serving threshold: fp16
        finds every object fp32 finds, same count, all paired, and moves boxes 1.4 to 7.4 px for
        about 1.4x. Boxes are what a detector is for. See tools/export/yolov11.py.
        """
        assert not [k for k in published("yolov11", variant) if k.endswith("fp16")]


class TestDetections:
    @pytest.mark.parametrize("variant", ALL)
    def test_finds_the_scene(self, predictor_for, image, variant):
        require_weights("yolov11", variant)
        detections = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        assert len(detections) > 0
        names = {d.class_name for d in detections}
        assert names <= EXPECTED_NAMES, f"unexpected classes: {names - EXPECTED_NAMES}"
        assert "person" in names

    def test_names_come_from_the_checkpoint(self, predictor_for, image):
        """Read from this family's own weights, not shared with any other family's vocabulary.

        YOLO11 happens to emit the same contiguous COCO ids YOLOv8 does, which is exactly why it
        has to be read rather than assumed: a fine-tuned checkpoint would not, and mozo never
        invents a class name.
        """
        require_weights("yolov11", "nano")
        detections = predictor_for("nano", "torch-fp32").predict(image, threshold=THRESHOLD)
        person = next(d for d in detections if d.class_name == "person")
        assert person.class_id == 0


class TestAgainstRecordedDetections:
    """A reference captured once, so a silent numeric change has something to fail against.

    Every other test here compares mozo to the vendored package, and the two share the
    letterboxing, the suppression and the coordinate mapping -- so a change to any of those moves
    both sides together and is invisible. On the sibling family that was not hypothetical:
    re-introducing the BGR channel flip the vendor arrived with turned 26 detections into 22 and
    nothing in the suite noticed.

    Recorded from the reshaped vendor, whose pre-processing was verified to produce a tensor
    bit-identical to the harvest's (``max|delta| = 0.0``) and whose parity against the original
    implementation is recorded in the vendor's PROVENANCE.md. Regenerate only when a change to the
    numbers is understood and intended.
    """

    def test_nano_finds_what_it_found_before(self, predictor_for, image):
        require_weights("yolov11", "nano")
        recorded = json.loads((FIXTURES / "yolov11_nano_example.json").read_text())
        got = predictor_for("nano", "torch-fp32").predict(image, threshold=THRESHOLD)

        assert len(got) == len(recorded), "the number of detections changed"
        assert [d.class_id for d in got] == [row["class_id"] for row in recorded]
        assert [d.class_name for d in got] == [row["class_name"] for row in recorded]
        assert [list(d.bbox) for d in got] == [row["bbox"] for row in recorded]
        assert [d.confidence for d in got] == pytest.approx([row["confidence"] for row in recorded])


class TestChannelOrder:
    """The vendor was BGR when it arrived; mozo is RGB. Nothing raises if that regresses."""

    def test_feeding_bgr_changes_the_answer(self, predictor_for, image):
        require_weights("yolov11", "nano")
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
        want = vendor_answer(variant, image)
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
    def test_the_published_graph_agrees_with_torch(self, predictor_for, image, variant):
        """ONNX is the only graph this family publishes; see ``test_no_coreml_is_published``."""
        runtime = "onnx-fp32"
        require_weights("yolov11", variant, "torch-fp32")
        require_weights("yolov11", variant, runtime)
        if runtime not in executable(published("yolov11", variant)):
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

    def test_auto_takes_torch_on_apple_silicon(self):
        """No per-family preference table exists, and this is why none is needed.

        The global table puts CoreML first on ``mps``, which is right for YOLOv8 and would be a
        crash for this family. It never fires here because ``select_runtime`` chooses only among
        published keys and this family publishes no CoreML -- so the absence of the artifact is
        what makes the shared table safe. If CoreML for YOLO11 is ever published, this fails
        alongside ``test_no_coreml_is_published``, and *then* a per-family table is the fix.
        """
        require_weights("yolov11", "nano", "torch-fp32")
        assert select_runtime("cpu", artifacts("yolov11", "nano")) == "torch-fp32"
        assert select_runtime("mps", artifacts("yolov11", "nano")) == "torch-fp32"
        assert select_runtime("cuda", artifacts("yolov11", "nano")) == "torch-fp32"

    def test_a_graph_runtime_takes_its_size_from_the_artifact(self, predictor_for):
        """Letterboxing to anything else would feed the runtime a shape it cannot accept.

        The runner reports ``input_shape``, so the adapter asks rather than assuming -- which is
        also what lets one adapter body serve a runtime it was not written against.
        """
        runtime = "onnx-fp32"
        require_weights("yolov11", "nano", runtime)
        if runtime not in executable(published("yolov11", "nano")):
            pytest.skip(f"{runtime} is published but not runnable here")
        assert predictor_for("nano", runtime).imgsz == 640


class TestSegmentation:
    """The ``seg-`` variants answer with a mask per detection, and the others with none."""

    def test_a_segmentation_variant_returns_one_mask_per_detection(self, predictor_for, image):
        """The count is the point: a mask that does not pair one-to-one with a box is unusable,
        and nothing about the array's own shape would reveal it."""
        require_weights("yolov11", "seg-nano")
        found = predictor_for("seg-nano", "torch-fp32").predict(image, threshold=THRESHOLD)

        assert len(found) > 0, "the fixture photograph has objects in it"
        for detection in found:
            mask = np.asarray(detection.masks)
            assert mask.squeeze().shape == image.shape[:2], (
                "a mask is at the source image's resolution, so it pairs with the box beside it"
            )
            assert mask.dtype == np.bool_, "a mask is a yes or no per pixel"

    def test_a_detection_variant_returns_no_masks_at_all(self, predictor_for, image):
        """``None`` rather than an empty array, so the two kinds are distinguishable without
        measuring a length -- which is what lets one adapter serve both."""
        require_weights("yolov11", "nano")
        found = predictor_for("nano", "torch-fp32").predict(image, threshold=THRESHOLD)

        assert len(found) > 0
        assert all(detection.masks is None for detection in found)

    def test_a_threshold_nothing_clears_returns_nothing(self, predictor_for, image):
        """An empty answer is an answer, not a crash.

        Mask assembly resizes the whole stack at once, and ``interpolate`` refuses a tensor with
        no channels -- so a threshold that keeps nothing raised from inside the resize, which over
        HTTP is a 500 on a query parameter a caller is entitled to send.
        """
        require_weights("yolov11", "seg-nano")
        found = predictor_for("seg-nano", "torch-fp32").predict(image, threshold=0.999)

        assert len(found) == 0

    def test_every_mask_lies_inside_its_own_box(self, predictor_for, image):
        """The crop is the last step of assembly and the one that pairs a mask with a box.

        Gathering the coefficients with a second pass over the scores, rather than with the index
        the suppression already chose, would put each mask on a neighbouring object -- plausible
        everywhere except here, because a mask cropped to box A cannot have pixels outside box A.
        """
        require_weights("yolov11", "seg-nano")
        found = predictor_for("seg-nano", "torch-fp32").predict(image, threshold=THRESHOLD)

        for detection in found:
            mask = np.asarray(detection.masks).squeeze()
            rows, columns = np.nonzero(mask)
            if not len(rows):
                continue
            x1, y1, x2, y2 = detection.bbox
            assert columns.min() >= int(x1) - 1 and columns.max() <= int(x2) + 1
            assert rows.min() >= int(y1) - 1 and rows.max() <= int(y2) + 1

    @pytest.mark.parametrize("variant", SEGMENTING)
    def test_mozo_reports_the_masks_the_vendor_found(self, predictor_for, image, variant):
        """The same promise :class:`TestMatchesTheVendor` makes about boxes, made about masks.

        Every step after the head is the vendor's -- the suppression, the coordinate mapping and
        the whole of the assembly -- so a mask mozo returns that the vendor did not produce could
        only have come from the adapter, which is the one place that is meant to hold no model
        maths at all.
        """
        want = vendor_answer(variant, image)
        got = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)

        assert want.masks is not None, "a segmentation checkpoint must produce masks"
        assert len(want) == len(got)
        for index, detection in enumerate(got):
            assert np.array_equal(np.asarray(detection.masks).squeeze(), want.masks[index])


class TestTheHeadsThisPackageServes:
    """Which checkpoints the vendor will build, and which it refuses.

    The refusals matter as much as the acceptances: this package reads a checkpoint's own record
    of itself, so the way it stays honest about a head it does not implement is to say so rather
    than to build something plausible out of the parts it recognises.
    """

    def test_the_head_table_names_a_task_and_a_coefficient_count_for_each(self):
        from mozo.vendors.yolov11_deploy.build import HEADS

        assert HEADS["Detect"] == ("detect", None), "a detection head has no mask coefficients"
        assert HEADS["Segment"] == ("segment", "nm"), "a segmentation head records them as 'nm'"

    def test_a_head_with_no_dataflow_is_refused_by_name(self):
        """``Pose`` and ``OBB`` are published for this family and are not implemented here.

        Refused by the dataflow table rather than by :data:`HEADS`, and that is the order: a class
        nothing knows how to route tensors through cannot be built at all, so the head table never
        sees it. Which is why the next test exists -- the head table's own gate needs a class that
        *does* build.
        """
        from mozo.vendors.yolov11_deploy.build import build_module

        with pytest.raises(NotImplementedError, match="Pose"):
            build_module(type("Pose", (), {})())

    def test_a_final_layer_that_is_not_a_head_is_refused(self):
        """The gate :data:`HEADS` is actually for: a class that builds but does not end a network.

        This is what would catch a head given a dataflow and then forgotten in :data:`HEADS` --
        the failure that would otherwise reach ``head.cv2`` and raise something about a missing
        attribute, three steps from the cause.
        """
        from mozo.vendors.yolov11_deploy.build import build_network

        layer = type("Concat", (), {})()
        layer.__dict__.update({"i": 0, "f": -1})
        layers = type("Layers", (), {})()
        layers.__dict__["_modules"] = {"0": layer}
        record = type("Model", (), {})()
        record.__dict__["_modules"] = {"model": layers}

        with pytest.raises(NotImplementedError, match="Concat"):
            build_network(record)

    def test_a_head_whose_task_disagrees_with_the_checkpoint_is_refused(self):
        """The task is recorded on the root and the head is the last layer, independently.

        A checkpoint where they disagree is one this package has misread, not one it can serve --
        and reading a ``Segment`` head as detection would drop the mask branch and answer boxes,
        which is the failure that produces plausible output.
        """
        from mozo.vendors.yolov11_deploy.build import build_network
        from mozo.vendors.yolov11_deploy.reader import load_model_record

        record = load_model_record(vendor_checkpoint("seg-nano"))
        record.__dict__["task"] = "detect"
        with pytest.raises(ValueError, match="task 'detect' for a Segment head"):
            build_network(record)


class TestCallerSuppliedNames:
    def test_caller_labels_override_the_checkpoint(self, predictor_for, image):
        require_weights("yolov11", "nano")
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
        from mozo.adapters.yolov11 import YOLOv11Predictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY["yolov11"]
        assert entry["adapter_class"] == YOLOv11Predictor.__name__
        assert entry["module"] == "mozo.adapters.yolov11"
        assert set(entry["variants"]) == set(YOLOv11Predictor.VARIANTS)
