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

import pytest

from mozo.runtimes import select_runtime
from conftest import COORD_STEP, FIXTURE, published, require_weights
from mozo.weights import WeightsError, artifacts

THRESHOLD = 0.5

#: What the graph may move against torch. Measured across all eight variants rather than assumed:
#: five agree exactly, seg-medium and seg-large differ by one step of PixelFlow's rounding, and
#: nano by 0.09. So nine steps, which covers the worst measured case with room and still fails on
#: anything an order of magnitude larger.
#:
#: ``tools/export/rfdetr.py`` allows a whole pixel at full precision, and that is not evidence
#: this needs to: it is a bound nobody measured, eleven times the worst real disagreement. A
#: transformer's top-k can reorder where a convolutional head's cannot, which is the reason to
#: check detections rather than raw tensors -- it is not a reason to accept a pixel of drift.
BOX_TOLERANCE = 9 * COORD_STEP

DETECTION = ["nano", "small", "medium", "large"]
SEGMENTATION = ["seg-nano", "seg-small", "seg-medium", "seg-large"]
#: Deliberately outside ``ALL``. The keypoint preview publishes no graph artifact, and its head
#: has its own two-slot class space rather than COCO's -- so the sweeps that assert torch/ONNX
#: agreement and COCO's vocabulary would both be asking it the wrong question.
KEYPOINT = ["keypoint-preview"]
ALL = DETECTION + SEGMENTATION

#: COCO's person keypoints, in the order the published ``labels.json`` names them.
JOINTS = 17

#: The scene in the fixture, as ids in COCO's original space. A model that reads this photograph
#: as anything else has either regressed or been given the wrong vocabulary.
EXPECTED_NAMES = {"person", "cup", "dining table", "laptop", "cell phone"}


@pytest.fixture(scope="module")
def predictor_for():
    """Build predictors once per (variant, runtime) -- loading a checkpoint is the slow part."""
    from mozo.adapters.rfdetr import RFDETRPredictor

    # Keyed by runtime only, and dropped when the variant changes. Both runtimes of one
    # variant have to coexist for the agreement test, but nothing needs two variants at once --
    # and holding all sixteen took peak RSS from 2.6 GB to 6.1 GB.
    cache: dict[str, object] = {}
    loaded = None

    def build(variant: str, runtime: str):
        nonlocal loaded
        if variant != loaded:
            cache.clear()
            loaded = variant
        if runtime not in cache:
            try:
                cache[runtime] = RFDETRPredictor(variant, device="cpu", runtime=runtime)
            except WeightsError as error:
                pytest.skip(f"rfdetr/{variant} weights unavailable: {error}")
        return cache[runtime]

    return build


class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_every_variant_publishes_torch_and_onnx(self, variant):
        keys = published("rfdetr", variant)
        if not keys:
            pytest.skip(f"rfdetr/{variant} is not in the manifest")
        assert "torch-fp32" in keys
        assert "onnx-fp32" in keys
        assert "labels" in keys


class TestDetections:
    @pytest.mark.parametrize("variant", ALL)
    def test_finds_the_scene(self, predictor_for, image, variant):
        require_weights("rfdetr", variant, "torch-fp32")
        detections = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        assert len(detections) > 0
        names = {d.class_name for d in detections}
        assert names <= EXPECTED_NAMES, f"unexpected classes: {names - EXPECTED_NAMES}"
        assert "person" in names

    @pytest.mark.parametrize("variant", ALL)
    def test_names_come_from_the_published_vocabulary(self, predictor_for, image, variant):
        """RF-DETR emits COCO's original ids. The contiguous list would say "bicycle" here."""
        require_weights("rfdetr", variant, "torch-fp32")
        detections = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        person = next(d for d in detections if d.class_name == "person")
        assert person.class_id == 1

    @pytest.mark.parametrize("variant", SEGMENTATION)
    def test_segmentation_variants_return_masks(self, predictor_for, image, variant):
        require_weights("rfdetr", variant, "torch-fp32")
        detections = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        assert detections[0].masks is not None
        assert detections[0].masks[0].shape[:2] == image.shape[:2]
        assert not detections[0].keypoints

    @pytest.mark.parametrize("variant", DETECTION)
    def test_detection_variants_return_no_masks(self, predictor_for, image, variant):
        """And no joints either -- the uniformity that lets one result type serve the family.

        A variant without a head for something returns that field unset rather than empty or
        zero-filled, so a caller reads one type and asks what is on it. Asserted here and in the
        segmentation sweep above rather than in a third sweep of its own: ``predictor_for`` drops
        its cache whenever the variant changes, so a separate parametrisation over the same eight
        variants would reload eight checkpoints to assert one more thing about them.
        """
        require_weights("rfdetr", variant, "torch-fp32")
        detections = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        assert detections[0].masks is None
        assert not detections[0].keypoints


class TestKeypoints:
    """The one variant that answers where a person's joints are."""

    @pytest.mark.parametrize("variant", KEYPOINT)
    def test_publishes_torch_only(self, variant):
        """No graph artifact, and ``EXECUTES`` is not the reason -- the adapter runs ONNX happily
        for its siblings. The dual-projector graph has simply never been exported, so ``auto``
        must not be able to offer one. If an export ever lands, this is the reminder that the
        runtime-agreement sweep should gain this variant at the same time.
        """
        keys = published("rfdetr", variant)
        if not keys:
            pytest.skip(f"rfdetr/{variant} is not in the manifest")
        assert sorted(keys) == ["labels", "torch-fp32"]

    def test_returns_seventeen_named_joints(self, predictor_for, image):
        require_weights("rfdetr", "keypoint-preview", "torch-fp32")
        detections = predictor_for("keypoint-preview", "torch-fp32").predict(image, threshold=THRESHOLD)
        assert len(detections) > 0
        for detection in detections:
            assert len(detection.keypoints) == JOINTS
        first = detections[0].keypoints
        assert [k.id for k in first] == list(range(JOINTS))
        assert first[0].name == "nose"
        assert first[-1].name == "right_ankle"

    def test_its_class_space_is_its_own(self, predictor_for, image):
        """Two slots -- background at 0, person at 1 -- not COCO's sparse ids running to 90.

        The number happens to coincide with COCO's ``person``, which is exactly why it is worth
        pinning: publishing the detection vocabulary here would still name id 1 "person" and be
        wrong about every other id it claimed to know.
        """
        require_weights("rfdetr", "keypoint-preview", "torch-fp32")
        detections = predictor_for("keypoint-preview", "torch-fp32").predict(image, threshold=THRESHOLD)
        assert {d.class_id for d in detections} == {1}
        assert {d.class_name for d in detections} == {"person"}

    def test_an_unseen_joint_keeps_its_slot(self, predictor_for, image):
        """Upstream's behaviour, reproduced rather than tidied.

        The fixture crops at the hips, so the lower-body joints are not in the photograph. They
        come back anyway, at a confidence near zero and at coordinates that mean nothing -- because
        a joint's index *is* its identity, and dropping the invisible ones would renumber the rest.
        """
        require_weights("rfdetr", "keypoint-preview", "torch-fp32")
        detections = predictor_for("keypoint-preview", "torch-fp32").predict(image, threshold=THRESHOLD)
        joints = detections[0].keypoints
        assert len(joints) == JOINTS
        by_name = {k.name: k for k in joints}
        assert by_name["nose"].confidence > 0.5
        assert by_name["left_ankle"].confidence < 0.1


class TestRuntimeAgreement:
    """The artifact you pick must not change the answer."""

    @pytest.mark.parametrize("variant", ALL)
    def test_torch_and_onnx_agree(self, predictor_for, image, variant):
        require_weights("rfdetr", variant, "torch-fp32")
        require_weights("rfdetr", variant, "onnx-fp32")

        torch_out = predictor_for(variant, "torch-fp32").predict(image, threshold=THRESHOLD)
        onnx_out = predictor_for(variant, "onnx-fp32").predict(image, threshold=THRESHOLD)

        assert len(torch_out) == len(onnx_out)
        assert [d.class_name for d in torch_out] == [d.class_name for d in onnx_out]

        # A rounding boundary can move one edge by a step of PixelFlow's coordinate precision;
        # anything larger is the model disagreeing with itself. The float-level check, before any
        # rounding, lives in tools/export.
        worst = max(
            (max(abs(a - b) for a, b in zip(x.bbox, y.bbox)) for x, y in zip(torch_out, onnx_out)),
            default=0,
        )
        assert worst <= BOX_TOLERANCE, f"boxes moved {worst} px between torch-fp32 and onnx-fp32"

    def test_auto_picks_torch(self):
        """Asserted against the real manifest rather than by loading a duplicate predictor."""
        require_weights("rfdetr", "small", "torch-fp32")
        assert select_runtime("cpu", artifacts("rfdetr", "small")) == "torch-fp32"


class TestCallerSuppliedNames:
    def test_caller_labels_override_the_published_ones(self, predictor_for, image):
        require_weights("rfdetr", "small", "torch-fp32")
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
        require_weights("rfdetr", "small", "torch-fp32")
        from PIL import Image

        image = Image.open(FIXTURE).convert("RGB")
        want = upstream.predict(image, threshold=0.1)
        got = predictor_for("small", "torch-fp32").predict(str(FIXTURE), threshold=0.1)

        assert len(got) == len(want.xyxy)
        assert abs(float(max(want.confidence)) - max(d.confidence for d in got)) < 0.01

class TestRegistry:
    def test_registry_agrees_with_the_adapter(self):
        """The variant list is written twice on purpose, so something has to hold it together.

        mozo.registry must answer /models without importing an adapter, and every adapter pulls
        torch in -- so the registry cannot derive its list from the adapter. This test is what
        makes the duplication safe rather than a latent drift.
        """
        from mozo.adapters.rfdetr import RFDETRPredictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY["rfdetr"]
        assert entry["adapter_class"] == RFDETRPredictor.__name__
        assert entry["module"] == "mozo.adapters.rfdetr"
        assert set(entry["variants"]) == set(RFDETRPredictor.VARIANTS)

    def test_the_sweeps_between_them_cover_every_variant(self):
        """``ALL`` is a hand-kept subset now, so something has to notice what falls outside it.

        It used to be the whole family, and a variant added to the adapter was swept by everything
        in this file for free. ``KEYPOINT`` is deliberately outside it -- the torch/ONNX agreement
        and COCO-vocabulary sweeps would be asking that variant the wrong question -- but the cost
        of splitting the list is that a *tenth* variant could now be tested by nothing at all and
        no test would fail. This is what makes the split safe rather than a hole.
        """
        from mozo.adapters.rfdetr import RFDETRPredictor

        assert set(DETECTION + SEGMENTATION + KEYPOINT) == set(RFDETRPredictor.VARIANTS)
