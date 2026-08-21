"""EdgeTAM: the trunk's shape, the settings no checkpoint can vouch for, and the prompt contract.

What EdgeTAM shares with every other promptable family -- the adapter's output shape, the
server's prompt parsing -- is in ``test_promptable.py``, checked against all of them.

Bit-exactness against upstream lives in ``tools/verify/edgetam.py``, which needs a checkout and
cannot run here. What this file pins is everything that would still be green if the model had
quietly changed: geometry that must match the checkpoint's key names, flags that carry no
weights, and the shape of what an adapter hands back.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mozo.registry import get_model_info
from mozo.vendors.edgetam_deploy.backbones.repvit import DEPTHS, WIDTHS, RepViT
from mozo.vendors.edgetam_deploy.config import SETTINGS, STABILITY
from mozo.vendors.edgetam_deploy.image import preprocess, to_model_coords
from mozo.vendors.edgetam_deploy.network import EdgeTam
from mozo.vendors.edgetam_deploy.predictor import BOX_BOTTOM_RIGHT, BOX_TOP_LEFT

FAMILY, VARIANT = "edgetam", "edgetam"


@pytest.fixture(scope="module")
def network():
    """The image-mode network, built but unloaded. Geometry needs no weights."""
    return EdgeTam()


# --- the trunk -------------------------------------------------------------------------------

def test_the_trunk_is_repvit_m1_at_the_published_geometry():
    """Widths and depths are the one thing that must match ``repvit_m1``, because the published
    checkpoint is keyed by timm's names and a different ladder would not load."""
    assert WIDTHS == (48, 96, 192, 384)
    assert DEPTHS == (2, 2, 14, 2)


def test_the_trunk_emits_one_map_per_stage_at_the_expected_strides():
    """The neck indexes these finest-first and the mask decoder skips into the two finest, so a
    trunk that returned three maps, or coarsest-first, would be wrong in a way that still runs."""
    trunk = RepViT().eval()
    with torch.no_grad():
        maps = trunk(torch.zeros(1, 3, 256, 256))
    assert [m.shape[1] for m in maps] == list(WIDTHS)
    assert [m.shape[-1] for m in maps] == [64, 32, 16, 8]  # strides 4, 8, 16, 32


def test_the_trunk_and_the_neck_agree_on_widths(network):
    """``ImageEncoder`` asserts this at construction; this is what makes the assert meaningful
    rather than a line nobody has ever executed with a mismatch."""
    assert network.image_encoder.trunk.channel_list == list(WIDTHS[::-1])
    assert network.image_encoder.neck.backbone_channel_list == list(SETTINGS["backbone_channel_list"])


def test_squeeze_excite_alternates_and_starts_on():
    """timm gates every other block, first one on. The checkpoint carries ``se.*`` for exactly
    those blocks, so getting the phase wrong leaves half the weights with nowhere to load."""
    stage = RepViT().body["stages_2"]
    gated = [not isinstance(block.se, torch.nn.Identity) for block in stage.blocks]
    assert gated == [i % 2 == 0 for i in range(DEPTHS[2])]


# --- the settings a checkpoint cannot vouch for ----------------------------------------------

def test_the_stability_fallback_is_on(network):
    """Carries no weights, is not in upstream's YAML at all -- ``build_sam2`` appends it as a
    Hydra override when ``apply_postprocessing`` is true, which is its default. A strict load
    cannot catch this being wrong, and it only changes the answer for a single-mask prompt."""
    decoder = network.sam_mask_decoder
    assert decoder.dynamic_multimask_via_stability is True
    assert decoder.dynamic_multimask_stability_delta == STABILITY["dynamic_multimask_stability_delta"]
    assert decoder.dynamic_multimask_stability_thresh == STABILITY["dynamic_multimask_stability_thresh"]


def test_asking_for_one_mask_consults_the_candidates(network):
    """The fallback swaps an unstable single mask for the best of the three. Driven directly
    rather than through a photograph, because a real image may simply never be unstable."""
    decoder = network.sam_mask_decoder
    logits = torch.zeros(1, 4, 8, 8)
    logits[:, 0] = 0.001          # token 0: hovering at the threshold, so barely stable
    logits[:, 1:] = 5.0
    iou = torch.tensor([[0.1, 0.2, 0.9, 0.3]])
    masks, scores = decoder._dynamic_multimask_via_stability(logits, iou)
    assert masks.shape == (1, 1, 8, 8)
    assert scores.item() == pytest.approx(0.9)  # the best multimask candidate, not token 0's 0.1


def test_the_video_machinery_is_left_behind(network):
    """EdgeTAM's contribution is the 2-D spatial perceiver, and it is video-only. If any of this
    reappeared, the package would be carrying weights it never runs."""
    names = dict(network.named_modules())
    for absent in ("memory_attention", "memory_encoder", "spatial_perceiver", "obj_ptr_proj"):
        assert absent not in names
    # But the one tensor that looks like tracker state and is not: it means "no memory to attend
    # to", which is exactly the situation a single image is in.
    assert network.no_mem_embed.shape == (1, 1, SETTINGS["hidden_dim"])


# --- preprocessing ---------------------------------------------------------------------------

@pytest.mark.parametrize("height,width", [(1281, 1920), (640, 640), (100, 3000)])
def test_preprocessing_squashes_to_a_square(height, width):
    """EdgeTAM distorts the aspect ratio rather than letterboxing, which is what it was trained
    under. Several families here letterbox, so the instinct to subtract a pad is wrong."""
    batch = preprocess(np.zeros((height, width, 3), np.uint8), 1024)
    assert batch.shape == (1, 3, 1024, 1024)


def test_preprocessing_refuses_anything_that_is_not_rgb():
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10), np.uint8), 1024)
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10, 4), np.uint8), 1024)


def test_prompt_coordinates_squash_with_the_pixels():
    """A click must land on the same feature after the squash, which means scaling x and y by
    independent factors rather than one ratio."""
    coords = to_model_coords(np.array([[[960.0, 640.0]]]), (1280, 1920), 1024)
    assert coords[0, 0, 0].item() == pytest.approx(512.0)
    assert coords[0, 0, 1].item() == pytest.approx(512.0)


# --- the prompt contract ----------------------------------------------------------------------

def test_a_box_is_two_corners_with_reserved_labels():
    """EdgeTAM has no box input; a box is spelled as corners carrying labels 2 and 3."""
    assert (BOX_TOP_LEFT, BOX_BOTTOM_RIGHT) == (2, 3)


def test_prompt_token_order_does_not_change_the_mask(network):
    """The opposite is the natural assumption and it is wrong, so it is pinned.

    Each token's learned embedding is chosen by its *label*, not by its position, and the sparse
    tokens are only read through attention rather than sliced by index -- so corners-then-click
    and click-then-corners are the same multiset. ``predict`` still emits corners first, because
    that is what upstream does and what keeps the parity gate exact.
    """
    encoder = network.sam_prompt_encoder
    box = torch.tensor([[[100.0, 200.0], [400.0, 700.0]]])
    click = torch.tensor([[[300.0, 500.0]]])
    corners_first, _ = encoder(
        points=(torch.cat([box, click], 1), torch.tensor([[2, 3, 1]])), boxes=None, masks=None)
    click_first, _ = encoder(
        points=(torch.cat([click, box], 1), torch.tensor([[1, 2, 3]])), boxes=None, masks=None)
    assert corners_first.shape == click_first.shape
    ours = sorted(tuple(t.tolist()) for t in corners_first[0])
    theirs = sorted(tuple(t.tolist()) for t in click_first[0])
    assert ours == theirs


def test_a_box_prompt_carries_the_padding_token(network):
    """SAM 2 changed this from SAM 1 and EdgeTAM inherits the change: the predictor folds the
    corners into the point list and passes ``boxes=None``, so ``pad`` fires and a box-only
    prompt is three tokens, not two. HuggingFace's independent port reaches the same three."""
    encoder = network.sam_prompt_encoder
    corners = torch.tensor([[[100.0, 200.0], [400.0, 700.0]]])
    folded, _ = encoder(points=(corners, torch.tensor([[2, 3]])), boxes=None, masks=None)
    assert folded.shape[1] == 3
    assert torch.equal(folded[0, 2], encoder.not_a_point_embed.weight[0])


# --- the registry and the adapter --------------------------------------------------------------

def test_registry_agrees_with_the_adapter():
    """The variant list is written twice -- here and in the adapter -- so that answering "what
    exists" needs no torch import. This is what holds the two copies in step."""
    from mozo.adapters.edgetam import EdgeTamPredictor

    entry = get_model_info(FAMILY)
    assert entry["adapter_class"] == "EdgeTamPredictor"
    assert entry["module"] == "mozo.adapters.edgetam"
    assert entry["task_type"] == "promptable_segmentation"
    assert set(entry["variants"]) == set(EdgeTamPredictor.VARIANTS)
