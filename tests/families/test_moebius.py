"""Moebius: the things that would still be green if the model had quietly changed.

Bit-exactness against ``hustvl/Moebius`` is recorded in ``moebius_deploy/PROVENANCE.md`` and
reproduced by ``tools/verify/moebius.py``, both of which need the weights and an upstream checkout.
What this file pins is everything reachable without either:

* the geometry that has to match the published tensors' shapes, including the asymmetric walk down
  and up that sizes the cross-λ's positional table;
* the two off-by-ones that are upstream's behaviour rather than bugs -- nineteen steps out of
  twenty, and the conditioning ids being unconditional-first;
* the export rewrite, which must be algebraically exact even though it is not bit-exact;
* the composite contract, which is the whole reason this family is safe to call a photo editor:
  every pixel outside the mask comes back byte-identical;
* and the refusals, since a model frozen to one resolution has to say so rather than guess.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mozo.vendors.moebius_deploy.attention import (
    SelfLambda, fold_for_export, fold_positional)
from mozo.vendors.moebius_deploy.config import SPECS, VaeSpec, get_spec
from mozo.vendors.moebius_deploy.image import as_tensor, binarise, composite, dilate, to_pixels
from mozo.vendors.moebius_deploy.scheduler import DDIM, timesteps_for

FAMILY = "moebius"
ALL = ["general", "places2"]


class TestGeometry:
    """The numbers that have to agree with the checkpoint, held against each other."""

    @pytest.mark.parametrize("variant", ALL)
    def test_every_variant_is_the_same_architecture(self, variant):
        # Four checkpoints exist upstream and differ only in what they were tuned on. If this
        # ever stops holding, ``SPECS`` mapping them all to one object stops being right.
        assert get_spec(variant) == get_spec("general")

    def test_the_model_runs_at_one_resolution(self):
        spec = get_spec("general")
        assert spec.image_size == 512
        assert spec.latent == 64
        assert spec.image_size == spec.latent * spec.vae.downsample

    def test_head_dimension_is_the_width_over_eight(self):
        # Eight heads at every level, by way of a documented diffusers naming bug: the config's
        # ``attention_head_dim: 8`` is read as the head *count*.
        spec = get_spec("general")
        assert [spec.head_dim(c) for c in spec.block_out_channels] == [40, 80, 160]

    def test_the_latent_walk_is_not_symmetric(self):
        # The last down level does not downsample and the last up level does not upsample, so
        # this is not ``latent >> i``. Getting it wrong sizes ``rel_pos_emb`` for the wrong level.
        down, up = get_spec("general").latent_sides()
        assert down == (64, 32, 16)
        assert up == (16, 32, 64)
        assert up == tuple(reversed(down))

    def test_unknown_variants_name_the_known_ones(self):
        with pytest.raises(ValueError, match="general, places2"):
            get_spec("celebahq")

    def test_the_autoencoder_carries_sdxl_constants(self):
        # Not decoration: ``scaling_factor`` belongs to the diffusion model that was trained on
        # these latents, and the value identifies whose autoencoder this is.
        vae = VaeSpec()
        assert vae.scaling_factor == pytest.approx(0.13025)
        assert vae.latent_channels == 4
        assert vae.downsample == 8


class TestConditioning:
    """Latent Categories Guidance: twenty learned rows, split in half."""

    def test_unconditional_comes_first(self):
        # The batch is built unconditional-first and ``chunk(2)`` reads it back that way.
        # Reversed, guidance points away from the conditioning and still returns a picture.
        uncond, cond = get_spec("general").conditioning_ids
        assert uncond == tuple(range(10, 20))
        assert cond == tuple(range(10))

    def test_the_unconditional_half_is_not_null(self):
        # Ten trained embeddings, not zeros and not a dropped condition. There is no empty prompt
        # to substitute, which is why the ids are fixed rather than exposed.
        uncond, cond = get_spec("general").conditioning_ids
        assert len(uncond) == len(cond) == 10
        assert not set(uncond) & set(cond)


class TestScheduler:
    """DDIM, and the step that is quietly never run."""

    def test_twenty_steps_runs_nineteen(self):
        # Upstream's ``strength=0.99`` trims one from the front. Asking for twenty and running
        # twenty produces a different image and raises nothing.
        ddim = DDIM()
        assert len(timesteps_for(ddim.schedule(20), 20, 0.99)) == 19

    def test_full_strength_runs_them_all(self):
        ddim = DDIM()
        assert len(timesteps_for(ddim.schedule(20), 20, 1.0)) == 20

    def test_the_trimmed_step_is_the_first(self):
        ddim = DDIM()
        full = ddim.schedule(20)
        assert int(full[0]) == 950
        assert int(timesteps_for(full, 20, 0.99)[0]) == 900

    def test_the_final_step_takes_a_different_branch(self):
        # ``prev_timestep`` goes negative once, at the end, and the alpha comes from
        # ``final_alpha_cumprod`` instead of the table. It is a branch, so it needs its own test.
        ddim = DDIM()
        assert float(ddim.final_alpha_cumprod) == 1.0
        sample = torch.zeros(1, 4, 8, 8)
        pred = torch.ones(1, 4, 8, 8)
        # At t=0 the branch fires; the result must still be finite and must use alpha_prev == 1.
        out = ddim.step(pred, 0, sample, 20)
        assert torch.isfinite(out).all()

    def test_the_schedule_is_scaled_linear_not_linear(self):
        # "Scaled linear" interpolates the square roots and squares the result. Interpolating the
        # betas directly is a different curve and nothing raises.
        ddim = DDIM()
        naive = torch.linspace(0.00085, 0.012, 1000)
        assert not torch.allclose(1.0 - ddim.alphas_cumprod[:1], naive[:1])
        assert ddim.alphas_cumprod.shape == (1000,)
        assert ddim.alphas_cumprod[0] > ddim.alphas_cumprod[-1]


class TestExportRewrite:
    """The Conv3d fold: algebraically the same function, numerically not."""

    def test_the_fold_keeps_the_same_numbers(self):
        layer = SelfLambda(320, 40, 8, kernel=15).eval()
        folded = fold_positional(layer)
        assert folded.kernel_size == (15, 15)
        assert folded.padding == (7, 7)
        assert torch.equal(folded.weight, layer.pos_conv.weight.squeeze(2))
        assert torch.equal(folded.bias, layer.pos_conv.bias)

    def test_the_fold_is_close_but_not_exact(self):
        # This is why the torch path keeps the Conv3d. torch dispatches two- and three-dimensional
        # convolutions to different kernels, so they sum in a different order. If this ever
        # becomes exact, the export path can stop being a separate decision.
        layer = SelfLambda(64, 8, 4, kernel=15).eval()
        folded = fold_positional(layer)
        values = torch.randn(1, 1, 16, 32, 32)
        with torch.no_grad():
            want = layer.pos_conv(values)
            got = folded(values.reshape(16, 1, 32, 32)).reshape(1, 16, 8, 32, 32).transpose(1, 2)
        assert torch.allclose(got, want, atol=1e-4)

    def test_folding_a_model_swaps_every_positional_convolution(self):
        # What the export path calls. In place, and idempotent, because an export script that
        # folded twice would reach for weight.squeeze(2) on a tensor that no longer has that axis.
        import torch.nn as nn

        model = nn.Sequential(SelfLambda(64, 8, 4, kernel=15), SelfLambda(64, 8, 4, kernel=15))
        assert all(m.pos_conv.weight.dim() == 5 for m in model)
        fold_for_export(model)
        assert all(m.pos_conv.weight.dim() == 4 for m in model)
        fold_for_export(model)
        assert all(m.pos_conv.weight.dim() == 4 for m in model)

    def test_a_folded_layer_still_answers(self):
        # The forward branches on the weight's rank, so both forms have to reach the same shape.
        layer = SelfLambda(64, 8, 4, kernel=15).eval()
        x = torch.randn(1, 16 * 16, 64)
        with torch.no_grad():
            before = layer(x, 16, 16)
            after = fold_for_export(layer)(x, 16, 16)
        assert before.shape == after.shape
        assert torch.allclose(before, after, atol=1e-4)

    def test_an_even_kernel_is_refused(self):
        with pytest.raises(ValueError, match="odd"):
            SelfLambda(320, 40, 8, kernel=14)


class TestSampling:
    """The draw, and the one property that makes a seed worth exposing."""

    def test_the_draw_is_device_independent(self):
        # Drawn on the CPU and moved, so seed 0 is the same picture on a Mac and on a CUDA box.
        # A per-device generator would make the seed mean something different on each, which is
        # worse than useless for a knob whose whole job is "give me that one again".
        from mozo.vendors.moebius_deploy.vae import Gaussian

        parameters = torch.zeros(1, 8, 4, 4)
        first = Gaussian(parameters).sample(torch.Generator().manual_seed(0))
        second = Gaussian(parameters).sample(torch.Generator().manual_seed(0))
        third = Gaussian(parameters).sample(torch.Generator().manual_seed(1))
        assert torch.equal(first, second)
        assert not torch.equal(first, third)

    def test_the_mode_is_deterministic_and_is_not_what_upstream_runs(self):
        from mozo.vendors.moebius_deploy.vae import Gaussian

        parameters = torch.randn(1, 8, 4, 4)
        distribution = Gaussian(parameters)
        assert torch.equal(distribution.mode(), distribution.mean)

    def test_the_log_variance_is_clamped(self):
        # Left un-clamped, exp() overflows to inf on a checkpoint stored in half precision.
        from mozo.vendors.moebius_deploy.vae import Gaussian

        wild = torch.cat([torch.zeros(1, 4, 2, 2), torch.full((1, 4, 2, 2), 400.0)], dim=1)
        assert torch.isfinite(Gaussian(wild).std).all()


class TestImage:
    """Preprocessing, and the composite that is the family's actual contract."""

    def test_masks_binarise_by_content_not_by_dtype(self):
        # A float mask thresholded at 127 is empty, and empty raises nothing.
        assert binarise(np.array([[0.0, 1.0]], dtype=np.float32)).tolist() == [[0, 1]]
        assert binarise(np.array([[0, 255]], dtype=np.uint8)).tolist() == [[0, 1]]
        assert binarise(np.array([[False, True]])).tolist() == [[0, 1]]

    def test_a_uint8_mask_of_zeros_and_ones_survives(self):
        # The regression. uint8 holding {0, 1} is what a segmenter's boolean masks become, and
        # what the adapter's own box path writes. Thresholded at 127.5 it empties, the
        # empty-mask shortcut returns the input, and the caller is told everything worked.
        assert binarise(np.array([[0, 1]], dtype=np.uint8)).tolist() == [[0, 1]]
        assert binarise(np.ones((4, 4), dtype=np.uint8)).sum() == 16

    def test_the_hole_is_mid_grey_not_black(self):
        # The image lives in [-1, 1], so zeroing the hole is mid-grey. Doing it in [0, 1] gives
        # black, which is a conditioning signal the model was never trained on.
        image = np.full((8, 8, 3), 255, np.uint8)
        mask = np.zeros((8, 8), np.uint8)
        mask[2:4, 2:4] = 1
        pixels, binary, masked = as_tensor(image, mask)
        assert pixels.min() == pytest.approx(1.0)
        assert masked[0, :, 2, 2].tolist() == [0.0, 0.0, 0.0]
        assert masked[0, :, 0, 0].tolist() == pytest.approx([1.0, 1.0, 1.0])

    def test_decode_shifts_before_it_clamps(self):
        # Clamping in [-1, 1] first would fold the tails in at the wrong place.
        assert to_pixels(torch.tensor([[[[-3.0]], [[0.0]], [[3.0]]]])).tolist() == [[[0, 128, 255]]]

    def test_dilation_of_zero_is_the_identity(self):
        mask = np.zeros((16, 16), np.uint8)
        mask[6:10, 6:10] = 1
        assert np.array_equal(dilate(mask, 0), mask)
        assert dilate(mask, 5).sum() > mask.sum()

    def test_everything_outside_the_mask_is_byte_identical(self):
        # The whole reason this family can be pointed at a photograph. Upstream returns the
        # decoder's reconstruction of the entire frame; mozo returns the caller's own bytes.
        rng = np.random.default_rng(0)
        original = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        generated = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), np.uint8)
        mask[20:40, 20:40] = 1

        out = composite(original, generated, mask, feather=0)
        assert np.array_equal(out[:10, :10], original[:10, :10])
        assert np.array_equal(out[20:40, 20:40], generated[20:40, 20:40])

    def test_the_feather_only_reaches_where_the_blur_reaches(self):
        rng = np.random.default_rng(1)
        original = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        generated = np.zeros((64, 64, 3), np.uint8)
        mask = np.zeros((64, 64), np.uint8)
        mask[20:44, 20:44] = 1

        out = composite(original, generated, mask, feather=3)
        # Far from the hole, untouched to the byte.
        assert np.array_equal(out[:12, :12], original[:12, :12])
        # Well inside it, entirely the model's.
        assert out[32, 32].tolist() == [0, 0, 0]

    def test_the_feather_erodes_a_small_mask(self):
        # Worth pinning rather than discovering: a radius-3 Gaussian pulls the alpha below one
        # several pixels *inside* the selection, so on a mask only a few pixels across the caller
        # gets a blend of original and generated everywhere -- including the centre -- and the
        # thing they selected is still faintly there. The fix is a larger ``dilate``, not a
        # smaller feather, and that is a decision the caller has to be able to see.
        original = np.full((32, 32, 3), 200, np.uint8)
        generated = np.zeros((32, 32, 3), np.uint8)
        mask = np.zeros((32, 32), np.uint8)
        mask[14:18, 14:18] = 1

        blended = composite(original, generated, mask, feather=3)
        sharp = composite(original, generated, mask, feather=0)
        assert sharp[16, 16].tolist() == [0, 0, 0]
        assert blended[16, 16].tolist() != [0, 0, 0]

    def test_the_seam_reaches_outside_the_mask_and_only_so_far(self):
        # The contract stated precisely. "Outside the mask" and "untouched" are the same region
        # only at feather=0; at the default radius of 3 the blend spreads about 8 px past the
        # selection. Pinned because the loose version of this claim was written into four
        # documents before anyone measured it.
        rng = np.random.default_rng(2)
        original = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        generated = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), np.uint8)
        mask[64:192, 64:192] = 1

        moved = (composite(original, generated, mask, feather=3) != original).any(-1)
        outside = mask == 0
        rows, cols = np.where(moved & outside)
        reach = max(64 - cols.min(), cols.max() - 191, 64 - rows.min(), rows.max() - 191)
        assert 0 < reach <= 10, reach

        # At feather=0 nothing outside the selection moves at all.
        sharp = (composite(original, generated, mask, feather=0) != original).any(-1)
        assert not sharp[outside].any()

    def test_a_mismatched_generation_is_refused(self):
        with pytest.raises(ValueError, match="caller's resolution"):
            composite(np.zeros((8, 8, 3), np.uint8), np.zeros((4, 4, 3), np.uint8),
                      np.zeros((8, 8), np.uint8))
