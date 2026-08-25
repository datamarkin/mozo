"""ViTPose: the two rewritten halves, and what taking detections as input has to preserve.

Bit-exactness against ``transformers`` is recorded in ``vitpose_deploy/PROVENANCE.md`` and
reproduced by ``tools/bench/vitpose.py``, both of which need the weights. What this file pins is
everything that would still be green if the model had quietly changed:

* the two operations rewritten to avoid a SciPy dependency -- the affine warp and the DARK
  blur -- held against SciPy itself, which is what makes reimplementing them a dependency choice
  rather than a different operation;
* the expert selection, held against the mask form upstream uses;
* the geometry that has to match the checkpoints' shapes;
* and the one thing unique to this family: it is handed detections and must give them back.

The tests that load a checkpoint are skipped rather than failed when the weights are absent.
Point them at a local tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_vitpose.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import COORD_STEP, published, require_weights
from mozo.vendors.vitpose_deploy.config import SPECS, get_spec
from mozo.vendors.vitpose_deploy.image import (
    ASPECT, NORMALIZE_FACTOR, PADDING_FACTOR, box_to_center_and_scale, warp, warp_matrix)
from mozo.vendors.vitpose_deploy.layers import MoeMLP
from mozo.vendors.vitpose_deploy.network import VitPose
from mozo.vendors.vitpose_deploy.postprocess import KERNEL, SIGMA, blur, peaks, to_frame
from mozo.weights import WeightsError

FAMILY = "vitpose"
ALL = ["small", "base", "large", "huge"]

#: The variants that also publish an ONNX graph. ``huge`` does not: its graph is 2.5 GB, past
#: protobuf's single-file ceiling, so ONNX writes the weights beside it instead of inside it and
#: mozo publishes one file per artifact. ``tools/export/vitpose.py`` refuses rather than writing
#: the few hundred kilobytes of stub that leaves.
ONNX = ["small", "base", "large"]

#: COCO's person keypoints, in the order the published ``labels.json`` names them.
JOINTS = 17

scipy = pytest.importorskip("scipy", reason="the SciPy comparisons need SciPy to compare against")


@pytest.fixture(scope="module")
def predictor_for():
    """Build predictors once per (variant, runtime) -- loading a checkpoint is the slow part."""
    from mozo.adapters.vitpose import ViTPosePredictor

    # Keyed by runtime only, and dropped when the variant changes. Both runtimes of one variant
    # have to coexist for the agreement test, but nothing needs two variants at once -- and
    # ``huge`` alone is 3.6 GB.
    cache: dict[str, object] = {}
    loaded = None

    def build(variant: str, runtime: str = "torch-fp32"):
        nonlocal loaded
        if variant != loaded:
            cache.clear()
            loaded = variant
        if runtime not in cache:
            try:
                cache[runtime] = ViTPosePredictor(variant, device="cpu", runtime=runtime)
            except WeightsError as error:
                pytest.skip(f"vitpose/{variant} weights unavailable: {error}")
        return cache[runtime]

    return build


@pytest.fixture(scope="module")
def people(image):
    """Person boxes for the fixture photograph, from a detector rather than typed in.

    Typed-in boxes would drift from the picture the moment the fixture changed, and would not
    exercise the thing this family is for: taking another model's output as input.
    """
    from mozo.adapters.rfdetr import RFDETRPredictor

    try:
        found = RFDETRPredictor("medium", device="cpu").predict(image, threshold=0.5)
    except WeightsError as error:
        pytest.skip(f"rfdetr/medium weights unavailable, so there are no boxes: {error}")
    return found.filter_by_class_id(1)


# --- what was rewritten to avoid a dependency ---------------------------------------------------

def test_the_warp_is_the_one_scipy_would_have_done(image):
    """mozo does not depend on SciPy, so ``image.warp`` is written out. This is what makes that a
    dependency choice rather than a different warp: bit-identical on a real photograph, at three
    box shapes that exercise widening, heightening and a crop that runs off the frame."""
    from scipy.ndimage import affine_transform

    def reference(source, matrix, height, width):
        inverse = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))
        inverse = np.array([[inverse[1, 1], inverse[1, 0], inverse[1, 2]],
                            [inverse[0, 1], inverse[0, 0], inverse[0, 2]],
                            [0, 0, 1]])
        return np.stack([affine_transform(source[..., c], inverse, output_shape=(height, width),
                                          order=1) for c in range(3)], axis=-1)

    for box in ([50, 60, 170, 360], [10, 10, 410, 130], [-40, -40, 200, 500]):
        center, scale = box_to_center_and_scale(np.array(box, dtype=np.float64))
        matrix = warp_matrix(center, scale, 256, 192)
        assert np.array_equal(warp(image, matrix, 256, 192), reference(image, matrix, 256, 192))


def test_the_dark_blur_repeats_the_edge_sample_as_scipy_does():
    """``torch``'s reflect padding skips the edge sample and SciPy's repeats it. At sigma 0.8 the
    first neighbour carries about a fifth of the weight, so the wrong one is not a rounding
    error -- it is wrong at every border cell, which is where joints at the edge of a crop live.
    """
    from scipy.ndimage import gaussian_filter

    heatmaps = np.random.default_rng(0).random((2, JOINTS, 64, 48)).astype(np.float32) * 3
    radius = (KERNEL - 1) // 2
    reference = np.array([[gaussian_filter(one, sigma=SIGMA, radius=(radius, radius), axes=(0, 1))
                           for one in each] for each in heatmaps])
    assert np.abs(reference - blur(heatmaps)).max() < 1e-6

    skipped = torch.nn.functional.pad(
        torch.from_numpy(heatmaps.astype(np.float64)), (radius,) * 4, mode="reflect").numpy()
    repeated = np.pad(heatmaps.astype(np.float64), ((0, 0), (0, 0), (radius,) * 2, (radius,) * 2),
                      mode="symmetric")
    assert not np.array_equal(skipped, repeated), "the two paddings agree -- this test is vacuous"


def test_running_one_expert_is_running_all_six_and_masking_five():
    """Upstream evaluates every expert and multiplies five by zero, because a training batch can
    mix datasets. Inference asks one question, so ``MoeMLP`` indexes. Same arithmetic."""
    spec = get_spec("small")
    mlp = MoeMLP(spec).eval()
    hidden = torch.randn(2, 12, spec.hidden)

    with torch.inference_mode():
        indexed = mlp(hidden, expert=0)

        shared = torch.nn.functional.gelu(mlp.fc1(hidden))
        masked = torch.zeros_like(shared[:, :, -spec.part_features:])
        for index, expert in enumerate(mlp.experts):
            masked = masked + expert(shared) * (torch.zeros(2, 1, 1) == index)
        assert torch.equal(indexed, torch.cat([mlp.fc2(shared), masked], dim=-1))


# --- the geometry -------------------------------------------------------------------------------

def test_the_crop_reaches_outside_the_box():
    """The reason ``predict`` takes the whole frame. A 50x140 person is cropped at roughly
    131x175 -- pixels a tight crop has already thrown away."""
    _, scale = box_to_center_and_scale(np.array([0.0, 0.0, 50.0, 140.0]))
    width, height = scale * NORMALIZE_FACTOR

    assert width > 50 and height > 140
    assert width == pytest.approx(140 * ASPECT * PADDING_FACTOR, abs=0.01)
    assert height == pytest.approx(140 * PADDING_FACTOR, abs=0.01)
    assert width / height == pytest.approx(ASPECT, abs=1e-6)


def test_a_box_already_at_the_right_aspect_is_only_padded():
    """The widening is conditional; the padding is not."""
    _, scale = box_to_center_and_scale(np.array([0.0, 0.0, 192.0, 256.0]))
    assert (scale * NORMALIZE_FACTOR) == pytest.approx([192 * 1.25, 256 * 1.25], abs=0.01)


def test_the_corners_of_the_heatmap_map_back_to_the_corners_of_the_crop():
    """``to_frame`` divides by ``size - 1`` rather than ``size``. Off by one there moves every
    joint by half a cell, which is a couple of pixels in the frame and looks plausible."""
    center = np.array([100.0, 200.0], dtype=np.float32)
    scale = np.array([1.0, 2.0], dtype=np.float32)
    corners = np.array([[0.0, 0.0], [47.0, 63.0]])

    mapped = to_frame(corners, center, scale, (64, 48))
    span = scale * NORMALIZE_FACTOR
    assert mapped[0] == pytest.approx(center - span * 0.5)
    assert mapped[1] == pytest.approx(center + span * 0.5)


def test_a_dead_channel_is_marked_rather_than_placed_at_the_origin():
    """A heatmap with nothing positive in it has no peak. Upstream returns -1; returning 0 would
    put a confident-looking joint in the corner of the crop."""
    heatmaps = np.zeros((1, 2, 8, 8), dtype=np.float32)
    heatmaps[0, 1, 3, 5] = 2.0
    coordinates, scores = peaks(heatmaps)

    assert coordinates[0, 0].tolist() == [-1, -1] and scores[0, 0] == 0
    assert coordinates[0, 1].tolist() == [5, 3] and scores[0, 1] == 2


@pytest.mark.parametrize("variant", ALL)
def test_the_spec_describes_the_published_checkpoint(variant):
    """Geometry is written out here because mozo publishes checkpoints without their configs.
    Building the network is what checks it: a wrong width is a wrong tensor shape."""
    spec = get_spec(variant)
    model = VitPose(variant)

    assert spec.experts == 6, "every published variant is ViTPose++"
    assert spec.keypoints == JOINTS
    assert model.backbone.embeddings.position_embeddings.shape == (1, 16 * 12 + 1, spec.hidden)
    assert len(model.backbone.encoder.layer) == spec.layers
    assert model.head.conv.out_channels == JOINTS
    assert spec.heatmap == (64, 48)


def test_the_patch_embedding_pads_by_two():
    """One of the two things separating this trunk from a plain ViT. Upstream says so in its own
    docstring, and a 0 here would change the token count and fail the position embedding."""
    assert VitPose("small").backbone.embeddings.patch_embeddings.projection.padding == (2, 2)


# --- what the family publishes ------------------------------------------------------------------

class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_publishes_torch_and_a_graph_where_one_fits(self, variant):
        """No CoreML anywhere -- nothing has measured one, and mozo does not publish an artifact
        whose agreement nobody has checked."""
        keys = published(FAMILY, variant)
        if not keys:
            pytest.skip(f"{FAMILY}/{variant} is not in the manifest")
        assert "torch-fp32" in keys
        assert "labels" in keys
        assert ("onnx-fp32" in keys) == (variant in ONNX)
        assert not [key for key in keys if key.startswith("coreml")]

    def test_the_graphs_and_the_checkpoints_cover_every_variant(self):
        """Both lists are written here, so something has to hold them to what exists."""
        from mozo.adapters.vitpose import ViTPosePredictor

        assert set(ONNX) < set(ALL) == set(ViTPosePredictor.VARIANTS)

    def test_registry_agrees_with_the_adapter(self):
        """The variant list is written twice on purpose, so something has to hold it together.

        mozo.registry must answer /models without importing an adapter, and every adapter pulls
        torch in -- so the registry cannot derive its list from the adapter.
        """
        from mozo.adapters.vitpose import ViTPosePredictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY[FAMILY]
        assert entry["adapter_class"] == ViTPosePredictor.__name__
        assert entry["module"] == "mozo.adapters.vitpose"
        assert entry["variants"] == list(ViTPosePredictor.VARIANTS) == ALL
        assert set(SPECS) == set(ViTPosePredictor.VARIANTS)


# --- taking detections as input -------------------------------------------------------------------

class TestAnnotatesWhatItIsGiven:
    @pytest.mark.parametrize("variant", ALL)
    def test_returns_the_same_rows_with_joints_added(self, predictor_for, image, people, variant):
        """The whole contract of this family. Every other model produces detections; this one is
        handed them and gives them back -- same boxes, same class ids, same names, same scores.
        Rebuilding them from arrays would silently drop tracker ids and rename the classes."""
        require_weights(FAMILY, variant)
        posed = predictor_for(variant).predict(image, people)

        assert len(posed) == len(people)
        for before, after in zip(people, posed):
            assert after.bbox == before.bbox
            assert after.class_id == before.class_id
            assert after.class_name == before.class_name
            assert after.confidence == before.confidence
            assert len(after.keypoints) == JOINTS

    @pytest.mark.parametrize("variant", ALL)
    def test_the_joints_are_named_from_the_published_vocabulary(
            self, predictor_for, image, people, variant):
        """Named through the incoming class id would leave them unnamed for a detector whose
        person is 0 rather than 1. They hang off this checkpoint's own category instead."""
        require_weights(FAMILY, variant)
        joints = predictor_for(variant).predict(image, people)[0].keypoints

        assert [joint.name for joint in joints[:3]] == ["nose", "left_eye", "right_eye"]
        assert [joint.id for joint in joints] == list(range(JOINTS))

    @pytest.mark.parametrize("variant", ONNX)
    def test_the_graph_returns_what_the_checkpoint_does(
            self, predictor_for, image, people, variant):
        """The promise the export exists to keep: which artifact you pick does not change the
        answer. Both runtimes run the vendor's own crop and decode -- only the forward pass
        differs -- so a disagreement here is the graph, not a reimplementation around it."""
        require_weights(FAMILY, variant, "torch-fp32")
        require_weights(FAMILY, variant, "onnx-fp32")

        def joints(runtime):
            return np.array([[[joint.x, joint.y, joint.confidence] for joint in row.keypoints]
                             for row in predictor_for(variant, runtime).predict(image, people)])

        moved = np.abs(joints("torch-fp32") - joints("onnx-fp32"))
        # One step of PixelFlow's coordinate rounding, and no more. Measured at exactly one step
        # on ``small`` and zero on the other two, so this fails on a real divergence rather than
        # on the last float bit. The slack is because 0.01 has no exact float64 representation:
        # the difference of two values rounded to it lands a few ULPs either side of a step.
        assert moved[..., :2].max() <= COORD_STEP + 1e-9
        assert moved[..., 2].max() < 1e-4

    @pytest.mark.parametrize("variant", ALL)
    def test_the_joints_land_inside_the_frame(self, predictor_for, image, people, variant):
        """Coordinates come back in the frame's pixels, not the crop's. A joint in crop space
        would sit in the top-left corner of the picture and look almost right."""
        require_weights(FAMILY, variant)
        height, width = image.shape[:2]
        for row in predictor_for(variant).predict(image, people):
            seen = [joint for joint in row.keypoints if joint.confidence > 0.5]
            assert seen, "no joint at all was found on a person a detector was confident about"
            for joint in seen:
                assert -width <= joint.x <= 2 * width and -height <= joint.y <= 2 * height

    def test_nobody_in_the_frame_is_an_answer_rather_than_an_error(
            self, predictor_for, image, people):
        """Most frames of most videos. Raising here would make the common case the exception."""
        require_weights(FAMILY, "small")
        posed = predictor_for("small").predict(image, people.filter_by_class_id(-1))
        assert len(posed) == 0

    def test_it_does_not_filter_what_it_is_given(self, predictor_for, image):
        """Deliberate: which boxes are people is the caller's fact. This pins the behaviour so
        that adding a filter later is a decision someone makes rather than a quiet change."""
        require_weights(FAMILY, "small")
        from mozo.adapters.rfdetr import RFDETRPredictor

        try:
            found = RFDETRPredictor("medium", device="cpu").predict(image, threshold=0.5)
        except WeightsError as error:
            pytest.skip(f"rfdetr/medium weights unavailable: {error}")

        assert len(predictor_for("small").predict(image, found)) == len(found)
