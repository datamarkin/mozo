"""Depth Anything V2, from the variant table down to a depth map on a real photograph.

The tests that need weights skip when the artifact cannot be obtained -- not published, not
cached, or offline -- so this file is honest on a clean checkout and thorough on a machine that
has run ``tools/fetch/depth_anything_v2.py``. The manifest saying an artifact exists is not the
same as having its bytes, and only the second one lets a depth map be computed.

What is deliberately *not* tested here is agreement with upstream. That needs a checkout of the
authors' repository pinned to the extracted commit, and it lives in
``tools/verify/depth_anything_v2.py`` where the commit check can refuse to run against the wrong
baseline. A test that quietly compared against whatever upstream happened to be installed would
be worse than no test at all.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mozo.vendors.depth_anything_v2_deploy import MODEL_SPECS, get_spec
from mozo.weights import WeightsError, artifacts

FAMILY = "depth_anything_v2"
FIXTURE = "tests/fixtures/images/example.jpg"

RELATIVE = ("small", "base", "large")
METRIC = ("indoor-small", "indoor-base", "indoor-large",
          "outdoor-small", "outdoor-base", "outdoor-large")
ALL = RELATIVE + METRIC

#: One small variant per group is enough for the tests that actually load weights; the encoder
#: sizes differ in width, not in behaviour, and vitl costs 1.3 GB to prove the same point.
SAMPLED = ("small", "indoor-small", "outdoor-small")


def _published(variant: str) -> list[str]:
    try:
        return artifacts(FAMILY, variant)
    except WeightsError:
        return []


def _require(variant: str, runtime: str = "torch-fp32") -> None:
    if runtime not in _published(variant):
        pytest.skip(f"{FAMILY}/{variant} does not publish {runtime}")


@pytest.fixture(scope="module")
def image():
    return cv2.imread(FIXTURE)


@pytest.fixture(scope="module")
def predictor_for():
    """Build predictors once per variant -- loading a checkpoint is the slow part."""
    from mozo.adapters.depth_anything_v2 import DepthAnythingV2Predictor

    cache: dict[str, object] = {}

    def build(variant: str):
        if variant not in cache:
            try:
                cache[variant] = DepthAnythingV2Predictor(variant, device="cpu")
            except WeightsError as error:
                pytest.skip(f"{FAMILY}/{variant} weights unavailable: {error}")
        return cache[variant]

    return build


class TestVariantTable:
    def test_nine_variants(self):
        assert set(MODEL_SPECS) == set(ALL)

    @pytest.mark.parametrize("variant", RELATIVE)
    def test_relative_variants_have_no_unit(self, variant):
        spec = get_spec(variant)
        assert spec.unit is None
        assert spec.max_depth is None
        assert spec.relative

    @pytest.mark.parametrize("variant", METRIC)
    def test_metric_variants_are_in_metres(self, variant):
        spec = get_spec(variant)
        assert spec.unit == "metres"
        assert spec.max_depth == (20.0 if variant.startswith("indoor") else 80.0)
        assert not spec.relative

    @pytest.mark.parametrize("size,encoder", [("small", "vits"), ("base", "vitb"), ("large", "vitl")])
    def test_the_three_regimes_share_one_backbone_per_size(self, size, encoder):
        specs = [get_spec(size), get_spec(f"indoor-{size}"), get_spec(f"outdoor-{size}")]
        assert {s.encoder for s in specs} == {encoder}
        assert len({(s.features, s.out_channels) for s in specs}) == 1

    def test_unknown_variant_names_the_alternatives(self):
        with pytest.raises(KeyError, match="outdoor-large"):
            get_spec("giant")


class TestPublished:
    @pytest.mark.parametrize("variant", ALL)
    def test_every_variant_publishes_torch(self, variant):
        published = _published(variant)
        if not published:
            pytest.skip(f"{FAMILY}/{variant} is not published locally")
        assert "torch-fp32" in published

    @pytest.mark.parametrize("variant", ALL)
    def test_licence_and_notice_are_not_offered_as_runtimes(self, variant):
        # They travel with whatever you asked for; they are never a thing you ask for.
        assert not ({"LICENSE", "NOTICE"} & set(_published(variant)))


class TestPreprocessing:
    """The input keeps its aspect ratio, which is what upstream's published numbers used."""

    @pytest.mark.parametrize("shape", [(1281, 1920, 3), (480, 640, 3), (900, 600, 3), (518, 518, 3)])
    def test_shorter_side_is_518_and_both_sides_are_multiples_of_14(self, predictor_for, shape):
        _require("small")
        predictor = predictor_for("small")._predictor
        tensor, size = predictor.preprocess(np.zeros(shape, dtype=np.uint8))

        height, width = tensor.shape[-2:]
        assert size == shape[:2]
        assert height % 14 == 0 and width % 14 == 0
        assert min(height, width) == 518

    def test_a_wide_image_is_not_squashed_to_a_square(self, predictor_for):
        _require("small")
        predictor = predictor_for("small")._predictor
        tensor, _ = predictor.preprocess(np.zeros((1000, 2000, 3), dtype=np.uint8))
        height, width = tensor.shape[-2:]
        assert width > height


class TestDepthMaps:
    @pytest.mark.parametrize("variant", SAMPLED)
    def test_output_matches_the_input_resolution(self, predictor_for, image, variant):
        _require(variant)
        depth = predictor_for(variant).predict(image)
        assert depth.shape == image.shape[:2]
        assert depth.dtype == np.float32

    @pytest.mark.parametrize("variant", SAMPLED)
    def test_accepts_a_path_as_well_as_an_array(self, predictor_for, image, variant):
        _require(variant)
        from_path = predictor_for(variant).predict(FIXTURE)
        assert np.array_equal(from_path, predictor_for(variant).predict(image))

    def test_relative_depth_is_unitless(self, predictor_for, image):
        _require("small")
        model = predictor_for("small")
        assert model.unit is None
        assert model.max_depth is None
        depth = model.predict(image)
        # Larger means nearer, and that is the whole contract. Not a distance, so nothing here
        # asserts a range -- only that the map varies, i.e. the model said something.
        assert depth.min() >= 0.0
        assert depth.max() > depth.min()

    @pytest.mark.parametrize("variant,ceiling", [("indoor-small", 20.0), ("outdoor-small", 80.0)])
    def test_metric_depth_is_metres_within_the_variants_ceiling(self, predictor_for, image,
                                                                variant, ceiling):
        _require(variant)
        model = predictor_for(variant)
        assert model.unit == "metres"
        assert model.max_depth == ceiling
        depth = model.predict(image)
        assert 0.0 <= depth.min()
        assert depth.max() <= ceiling

    def test_indoor_and_outdoor_disagree_about_the_same_scene(self, predictor_for, image):
        """Two metric variants are two different models, not one with a scale factor."""
        _require("indoor-small")
        _require("outdoor-small")
        indoor = predictor_for("indoor-small").predict(image)
        outdoor = predictor_for("outdoor-small").predict(image)
        assert not np.allclose(indoor, outdoor)


class TestAdapter:
    def test_variants_come_from_the_vendor(self):
        from mozo.adapters.depth_anything_v2 import DepthAnythingV2Predictor

        assert set(DepthAnythingV2Predictor.VARIANTS) == set(MODEL_SPECS)

    def test_unknown_variant_is_rejected_before_any_download(self):
        from mozo.adapters.depth_anything_v2 import DepthAnythingV2Predictor

        with pytest.raises(ValueError, match="Unsupported variant"):
            DepthAnythingV2Predictor("giant")

    def test_registry_agrees_with_the_adapter(self):
        from mozo.adapters.depth_anything_v2 import DepthAnythingV2Predictor
        from mozo.registry import MODEL_REGISTRY

        entry = MODEL_REGISTRY[FAMILY]
        assert entry["adapter_class"] == DepthAnythingV2Predictor.__name__
        assert entry["module"] == "mozo.adapters.depth_anything_v2"
        assert set(entry["variants"]) == set(DepthAnythingV2Predictor.VARIANTS)
