"""BEN2, from the variant table down to an alpha matte on a real photograph.

The tests that need weights skip when the artifact cannot be obtained -- not published, not
cached, or offline -- so this file is honest on a clean checkout and thorough on a machine that
has run ``tools/fetch/ben2.py``.

What is deliberately *not* tested here is agreement with upstream. That needs a checkout of
``PramaLLC/BEN2`` pinned to the extracted commit, and it lives in ``tools/verify/ben2.py`` where
the commit check can refuse to run against the wrong baseline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import FIXTURE, published
from mozo.registry import BOXED, ENCODES, MODEL_REGISTRY, PROMPTED
from mozo.vendors.ben2_deploy import BACKBONE, DECODER, INPUT
from mozo.vendors.ben2_deploy.image import ALPHA_EPSILON, postprocess
from mozo.weights import WeightsError

FAMILY = "ben2"


@pytest.fixture(scope="module")
def predictor():
    from mozo.adapters.ben2 import Ben2Predictor

    try:
        return Ben2Predictor("base", device="cpu")
    except WeightsError as error:
        pytest.skip(f"{FAMILY}/base weights unavailable: {error}")


class TestRegistry:
    def test_registry_agrees_with_the_adapter(self):
        from mozo.adapters.ben2 import Ben2Predictor

        entry = MODEL_REGISTRY[FAMILY]
        assert tuple(entry["variants"]) == Ben2Predictor.VARIANTS
        assert entry["adapter_class"] == "Ben2Predictor"
        assert entry["module"] == "mozo.adapters.ben2"

    def test_one_variant_named_rather_than_a_wildcard(self):
        # An empty list means "accepts any variant name", which is a different promise from
        # "publishes exactly one".
        assert MODEL_REGISTRY[FAMILY]["variants"] == ["base"]

    def test_it_is_asked_nothing_and_told_nothing(self):
        """The whole answer is the picture: no prompt, no box, no embeddings."""
        task = MODEL_REGISTRY[FAMILY]["task_type"]
        assert task == "image_matting"
        assert task not in PROMPTED
        assert task not in BOXED
        assert FAMILY not in ENCODES

    def test_the_task_has_a_branch_in_the_server(self):
        import inspect

        from mozo import server

        assert '"image_matting"' in inspect.getsource(server.predict)


class TestPublished:
    def test_it_publishes_torch(self):
        keys = published(FAMILY, "base")
        if not keys:
            pytest.skip(f"{FAMILY}/base is not published locally")
        assert "torch-fp32" in keys

    def test_licence_and_notice_are_not_offered_as_runtimes(self):
        assert not ({"LICENSE", "NOTICE"} & set(published(FAMILY, "base")))

    def test_the_published_onnx_is_not_republished(self):
        """Upstream ships an ONNX; it is float16 and cannot hold parity. PROVENANCE says so."""
        assert "onnx-fp32" not in published(FAMILY, "base")


class TestGeometry:
    def test_the_backbone_is_swin_b(self):
        assert BACKBONE.embed_dim == 128
        assert BACKBONE.depths == (2, 2, 18, 2)
        assert BACKBONE.num_heads == (4, 8, 16, 32)
        assert BACKBONE.window_size == 12

    def test_head_dimension_is_32_at_every_stage(self):
        widths = [BACKBONE.embed_dim * 2 ** i for i in range(len(BACKBONE.depths))]
        assert {w // h for w, h in zip(widths, BACKBONE.num_heads)} == {32}

    def test_five_feature_maps_not_four(self):
        """The backbone seeds its output list with the patch embedding before the stages run."""
        assert BACKBONE.channels == (128, 128, 256, 512, 1024)
        assert len(BACKBONE.channels) == len(BACKBONE.depths) + 1

    def test_the_input_is_square_and_divides_into_the_patch_grid(self):
        assert INPUT == 1024
        # Four quadrants, each halved again by the 0.5 global downscale, then patch-embedded.
        assert INPUT % 2 == 0
        assert (INPUT // 2) % BACKBONE.patch_size == 0

    def test_decoder_pools_differ_between_the_global_and_refinement_blocks(self):
        assert DECODER.mclm_pools == (1, 4, 8)
        assert DECODER.mcrm_pools == (2, 4, 8)
        assert DECODER.num_heads == 1


class TestEinopsRewrites:
    """The nine reshape rewrites are the riskiest change in the vendor. Two invariants here;
    the full comparison against ``einops`` itself is in ``tools/verify/ben2.py``."""

    def test_patches2image_inverts_image2patches(self):
        from mozo.vendors.ben2_deploy.blocks import image2patches, patches2image

        x = torch.arange(2 * 3 * 8 * 12, dtype=torch.float32).reshape(2, 3, 8, 12)
        assert torch.equal(patches2image(image2patches(x)), x)

    def test_image2patches_puts_each_quadrant_in_its_own_row(self):
        from mozo.vendors.ben2_deploy.blocks import image2patches

        # A frame whose four quadrants are constant 0, 1, 2, 3 -- so a wrong axis order shows up
        # as a row that is not constant rather than as a number that is merely different.
        x = torch.zeros(1, 1, 4, 4)
        x[..., :2, 2:], x[..., 2:, :2], x[..., 2:, 2:] = 1, 2, 3
        rows = image2patches(x).flatten(1)
        assert [float(r.unique().item()) for r in rows] == [0.0, 1.0, 2.0, 3.0]


class TestThePoolingSubstitution:
    """``_pool`` uses ``avg_pool2d`` where the division is exact, which is what lets the model
    export. It is only sound because every ratio in this model divides evenly, so the equivalence
    is pinned here rather than argued in a comment."""

    #: Every (input shape, target) pair the model reaches, from the frozen 1024 input.
    PAIRS = [
        ((1, 128, 32, 32), (16, 16)), ((1, 128, 32, 32), (4, 4)), ((1, 128, 32, 32), (2, 2)),
        ((4, 128, 16, 16), (16, 16)), ((4, 128, 16, 16), (8, 8)), ((4, 128, 16, 16), (4, 4)),
        ((4, 128, 32, 32), (32, 32)), ((4, 128, 32, 32), (16, 16)), ((4, 128, 32, 32), (8, 8)),
        ((4, 128, 64, 64), (64, 64)), ((4, 128, 64, 64), (32, 32)), ((4, 128, 64, 64), (16, 16)),
    ]

    @pytest.mark.parametrize("shape,target", PAIRS)
    def test_it_is_bit_identical_to_adaptive_pooling(self, shape, target):
        from mozo.vendors.ben2_deploy.blocks import _pool

        torch.manual_seed(0)
        x = torch.randn(*shape)
        assert torch.equal(_pool(x, target), F.adaptive_avg_pool2d(x, target))

    def test_it_falls_back_when_the_division_is_not_exact(self):
        """A ratio that does not divide evenly must keep upstream's operator, not guess."""
        from mozo.vendors.ben2_deploy.blocks import _pool

        torch.manual_seed(0)
        x = torch.randn(1, 4, 30, 30)          # 30 is not a multiple of 7
        assert torch.equal(_pool(x, (7, 7)), F.adaptive_avg_pool2d(x, (7, 7)))


class TestTheStretchContract:
    """The default alpha is a per-image contrast stretch, not a probability."""

    def test_stretch_reaches_both_ends_of_the_range(self):
        matte = torch.full((1, 1, 8, 8), 0.4)
        matte[..., 0, 0] = 0.6
        out = postprocess(matte, (8, 8), stretch=True)
        assert out.min() == 0 and out.max() == 255

    def test_without_stretch_the_sigmoid_survives(self):
        matte = torch.full((1, 1, 8, 8), 0.4)
        matte[..., 0, 0] = 0.6
        out = postprocess(matte, (8, 8), stretch=False)
        assert out.max() == int(0.6 * 255)
        assert out.min() == int(0.4 * 255)

    def test_a_flat_matte_does_not_become_nan(self):
        """Upstream divides by ``max - min`` unguarded; a uniform frame is inf/nan cast to uint8."""
        flat = torch.full((1, 1, 4, 4), 0.3)
        out = postprocess(flat, (4, 4), stretch=True)
        assert out.dtype == np.uint8
        assert np.all(out == int(0.3 * 255))

    def test_the_guard_is_narrower_than_any_real_matte(self):
        assert ALPHA_EPSILON < 1e-6

    def test_a_one_pixel_image_keeps_two_dimensions(self):
        """``np.squeeze`` with no axis collapses a 1x1 matte to 0-d, which upstream then
        hands to ToPILImage and crashes on."""
        out = postprocess(torch.full((1, 1, 4, 4), 0.5), (1, 1), stretch=True)
        assert out.shape == (1, 1)


@pytest.fixture(scope="module")
def alpha(predictor, image):
    """The matte, computed once. Six tests below assert on it and each forward is ~5 s on CPU."""
    return predictor.predict(image)


@pytest.fixture(scope="module")
def cutout(predictor, image):
    """The default cut-out, computed once."""
    return predictor.cutout(image)


class TestMattes:
    def test_output_matches_the_input_resolution(self, alpha, image):
        assert alpha.shape == image.shape[:2]
        assert alpha.dtype == np.uint8

    def test_accepts_a_path_as_well_as_an_array(self, predictor, alpha):
        assert np.array_equal(predictor.predict(FIXTURE), alpha)

    def test_the_model_says_something(self, alpha):
        # Bimodal is the whole point: a matte that never commits is a matte that found nothing.
        assert alpha.max() > 200
        assert alpha.min() < 55

    def test_stretch_and_no_stretch_are_different_answers(self, predictor, image, alpha):
        assert not np.array_equal(alpha, predictor.predict(image, stretch=False))

    def test_cutout_is_rgba_and_carries_the_matte_as_alpha(self, cutout, image, alpha):
        assert cutout.shape == image.shape[:2] + (4,)
        assert np.array_equal(cutout[..., :3], image)
        assert np.array_equal(cutout[..., 3], alpha)

    def test_refine_changes_colour_and_not_opacity_alone(self, predictor, image, cutout):
        refined = predictor.cutout(image, refine=True)
        assert not np.array_equal(cutout[..., :3], refined[..., :3])

    @pytest.mark.parametrize("shape", [(64, 256, 3), (256, 64, 3), (1, 1, 3)])
    def test_odd_aspect_ratios_come_back_at_their_own_size(self, predictor, shape):
        assert predictor.predict(np.full(shape, 128, dtype=np.uint8)).shape == shape[:2]


class TestRefusals:
    def test_an_unknown_variant_names_the_alternatives(self):
        from mozo.adapters.ben2 import Ben2Predictor

        with pytest.raises(ValueError, match="base"):
            Ben2Predictor("large")

    @pytest.mark.parametrize("bad", [
        np.zeros((8, 8), dtype=np.uint8),           # no channel axis
        np.zeros((8, 8, 4), dtype=np.uint8),        # RGBA
        np.zeros((8, 8, 3), dtype=np.float32),      # not uint8
    ])
    def test_the_vendor_refuses_arrays_it_cannot_interpret(self, bad):
        from mozo.vendors.ben2_deploy.image import preprocess

        with pytest.raises(ValueError, match="uint8 RGB"):
            preprocess(bad)
