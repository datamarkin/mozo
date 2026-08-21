"""OWLv2: the tokenizer's quirks, the geometry, and the two coordinate rules.

What OWLv2 shares with mozo's other text-prompted family -- how the server refuses a missing
prompt -- is in ``test_prompted.py``, checked against both of them.

Bit-exactness against ``transformers`` lives in ``tools/verify/owlv2.py``, which needs the
weights. What this file pins is everything that would still be green if the model had quietly
changed: the tokenizer settings that decide whether a prompt encodes the way the weights expect,
the geometry that has to match the checkpoint's shapes, and the two places where a coordinate
convention is easy to read the obvious way and be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mozo.registry import get_model_info
from mozo.vendors.owlv2_deploy.checkpoint import DROPPED, translate
from mozo.vendors.owlv2_deploy.config import SPECS
from mozo.vendors.owlv2_deploy.heads import box_bias
from mozo.vendors.owlv2_deploy.image import RESCALE, preprocess, to_original
from mozo.vendors.owlv2_deploy.layers import quick_gelu
from mozo.vendors.owlv2_deploy.network import OwlV2
from mozo.vendors.owlv2_deploy.text.tokenizer import Tokenizer

FAMILY = "owlv2"


@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()


@pytest.fixture(scope="module")
def network():
    """The base geometry, built but unloaded. Shapes need no weights."""
    return OwlV2("base-ensemble")


# --- the tokenizer ------------------------------------------------------------------------------

def test_the_vocabulary_is_clips_and_the_context_is_sixteen(tokenizer):
    """49,408 entries is CLIP's, and 16 is OWLv2's -- every other CLIP model uses 77. The
    published position embedding is ``(16, width)``, so a longer context would not load."""
    assert len(tokenizer.encoder) == 49408
    assert tokenizer.context_length == 16
    assert SPECS["base"].text.context_length == 16


def test_an_exclamation_mark_is_its_own_token_with_id_zero(tokenizer):
    """The published config makes ``!`` the padding token, which puts it in the added-token
    table, which makes the tokenizer split it out before byte-pair encoding sees it. So it is
    id 0 rather than ``!</w>``'s 256 -- and id 0 is also what padding is."""
    ids, _ = tokenizer(["cat!"])
    assert ids[0, :4].tolist() == [49406, 2368, 0, 49407]


def test_padding_is_not_recoverable_from_the_ids(tokenizer):
    """Which is why the mask is returned rather than left to the caller. ``ids != 0`` is the
    obvious reading and it would drop a real token out of the attention on any prompt with an
    exclamation mark in it."""
    ids, mask = tokenizer(["a cat!"])
    assert (ids == 0).sum() > (mask == 0).sum()
    assert mask[0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert ids[0, 3].item() == 0 and mask[0, 3].item() == 1


def test_prompts_are_lowercased_but_entities_are_not_unescaped(tokenizer):
    """Two halves of the same rule: reproduce ``CLIPTokenizer``, which normalises with NFC,
    collapses whitespace and lowercases -- and nothing else. OpenAI's own cleaner additionally
    runs ``ftfy`` and unescapes HTML twice, and a sibling package here does exactly that, which
    is why this is pinned rather than assumed."""
    assert torch.equal(tokenizer(["A Red Hat"])[0], tokenizer(["a red hat"])[0])
    assert torch.equal(tokenizer(["  spaced   out "])[0], tokenizer(["spaced out"])[0])
    assert not torch.equal(tokenizer(["&amp;"])[0], tokenizer(["&"])[0])


def test_a_prompt_that_does_not_fit_is_still_terminated(tokenizer):
    """Truncation drops tokens off the end, which would drop the end-of-text marker -- and the
    text tower pools at the marker, so losing it would pool at whatever id happened to be
    largest."""
    ids, mask = tokenizer(["a photograph of a person sitting at a table with a laptop and a cup"])
    assert ids.shape == (1, 16)
    assert ids[0, -1].item() == 49407
    assert mask[0].tolist() == [1] * 16


# --- the geometry -------------------------------------------------------------------------------

def test_the_two_published_geometries_are_whole_numbers_of_patches():
    """960/16 and 1008/14. Not a detail -- the position embedding has one row per patch plus one
    for the class token, so a resolution that did not divide would not load."""
    assert SPECS["base"].vision.patches == 60
    assert SPECS["large"].vision.patches == 72
    for spec in (SPECS["base"], SPECS["large"]):
        assert spec.vision.image_size % spec.vision.patch_size == 0


def test_the_projection_matches_the_text_width():
    """The class head projects a patch to the text width and dots it against the projected
    prompt. If the two differed the einsum would not contract, which is a loud failure -- but
    only at the first forward, and only with weights."""
    for spec in SPECS.values():
        assert spec.text.projection == spec.text.width


def test_the_ensembles_share_a_geometry_with_the_plain_checkpoints():
    """They differ in training, not in shape: ``-ensemble`` averages the self-trained and
    fine-tuned weights. One spec each is what makes that true rather than duplicated."""
    assert SPECS["base-ensemble"] is SPECS["base"]
    assert SPECS["large-ensemble"] is SPECS["large"]


def test_the_network_builds_neither_of_the_two_dead_tensors(network):
    """``visual_projection`` and ``logit_scale`` are CLIP's contrastive head, which the detector
    never reads. They are in every published checkpoint, so ``checkpoint.translate`` has to drop
    them by name -- and a *third* unexpected key must still fail a strict load."""
    names = dict(network.named_parameters())
    assert not [n for n in names if "visual_projection" in n or "logit_scale" == n]
    assert DROPPED == {"owlv2.visual_projection.weight", "owlv2.logit_scale"}


def test_translate_drops_exactly_the_dead_tensors_and_renames_the_towers():
    """A rename table that silently dropped a live tensor would leave a strict load complaining
    about something missing rather than about the tensor that went astray."""
    published = {
        "owlv2.vision_model.pre_layernorm.weight": torch.zeros(1),
        "owlv2.text_model.final_layer_norm.bias": torch.zeros(1),
        "owlv2.text_projection.weight": torch.zeros(1),
        "owlv2.visual_projection.weight": torch.zeros(1),
        "owlv2.logit_scale": torch.zeros(()),
        "class_head.dense0.weight": torch.zeros(1),
    }
    assert set(translate(published)) == {
        "vision.pre_layernorm.weight",
        "text.final_layer_norm.bias",
        "text_projection.weight",
        "class_head.dense0.weight",
    }


def test_the_towers_activate_differently_from_the_heads(network):
    """``quick_gelu`` in the transformer blocks, plain ``GELU`` in the heads. Upstream's
    arrangement, invisible in the weights, and using one for both loads strictly."""
    x = torch.linspace(-3, 3, 7)
    assert not torch.allclose(quick_gelu(x), torch.nn.functional.gelu(x), atol=1e-4)
    assert isinstance(network.box_head.gelu, torch.nn.GELU)
    assert isinstance(network.objectness_head.gelu, torch.nn.GELU)


# --- the coordinate rules -----------------------------------------------------------------------

@pytest.mark.parametrize("height,width", [(1281, 1920), (640, 640), (100, 3000)])
def test_preprocessing_pads_to_a_square_rather_than_squashing(height, width):
    """A sibling family here squashes and several letterbox; OWLv2 does neither. It pads bottom
    and right to ``max(h, w)`` and resizes that, so the aspect ratio is kept and the padding is
    all on two edges."""
    batch = preprocess(np.zeros((height, width, 3), np.uint8), 960)
    assert batch.shape == (1, 3, 960, 960)


def test_the_pad_is_black_and_lands_bottom_right():
    """Which is checkable without the model: a white image padded to a square has a dark corner
    on exactly two sides. Normalisation moves black off zero, so the test is that the pad is
    darker than the picture rather than that it is any particular number."""
    batch = preprocess(np.full((100, 400, 3), 255, np.uint8), 960)
    assert batch[0, 0, 10, 10] > batch[0, 0, -10, 10]   # bottom is padding
    assert batch[0, 0, 10, 10] == batch[0, 0, 10, -10]  # right is not, at 4:1
    assert batch[0, 0, -10, -10] == batch[0, 0, -10, 10]


def test_boxes_descale_by_the_long_side_on_both_axes():
    """The usual reading is width for x and height for y, and it is wrong here: the pad made the
    coordinate space square. On a 4:3 image the two differ by a third on every y."""
    boxes = torch.tensor([[0.5, 0.5, 0.25, 0.25]])
    corners = to_original(boxes, (768, 1024))
    assert corners.tolist() == [[384.0, 384.0, 640.0, 640.0]]


def test_the_rescale_is_a_multiply_by_a_reciprocal():
    """Upstream carries ``rescale_factor`` as a constant and multiplies. Dividing by 255 is the
    same number and not the same float, and it moved the trunk's input by 9.5e-07."""
    assert RESCALE == 0.00392156862745098
    # The property, not the constant: a byte scaled both ways is not the same float, so swapping
    # the multiply for a divide would fail here rather than staying green.
    bytes_ = torch.arange(256, dtype=torch.float32)
    assert not torch.equal(bytes_ * RESCALE, bytes_ / 255.0)


# --- the box prior ------------------------------------------------------------------------------

def test_the_box_bias_centres_each_patch_on_itself():
    """The head predicts an offset in logit space and this is what it offsets from, so a head
    predicting nothing still emits a box on its own patch and one patch wide. Row-major over the
    grid, which is the order the trunk flattens in -- transposed, every box would be mirrored
    about the diagonal.

    The tolerance is 1e-3 rather than exact because of upstream's ``1e-4``, which keeps the
    logit finite at both ends: the last column's centre comes back as 0.9999 rather than 1.0,
    and without that term the first column's would be negative infinity."""
    bias = box_bias(4)
    assert bias.shape == (16, 4)
    centres = torch.sigmoid(bias[:, :2])
    assert centres[0].tolist() == pytest.approx([0.25, 0.25], abs=1e-3)
    assert centres[3].tolist() == pytest.approx([1.0, 0.25], abs=1e-3)   # x moves first
    assert centres[12].tolist() == pytest.approx([0.25, 1.0], abs=1e-3)
    assert torch.sigmoid(bias[:, 2:]).mean().item() == pytest.approx(0.25, abs=1e-3)
    assert torch.isfinite(bias).all()


# --- the registry and the adapter ----------------------------------------------------------------

def test_registry_agrees_with_the_adapter():
    """The variant list is written twice -- here and in the adapter -- so that answering "what
    exists" needs no torch import. This is what holds the two copies in step."""
    from mozo.adapters.owlv2 import OwlV2Predictor

    entry = get_model_info(FAMILY)
    assert entry["adapter_class"] == "OwlV2Predictor"
    assert entry["module"] == "mozo.adapters.owlv2"
    assert entry["task_type"] == "open_vocabulary_detection"
    assert set(entry["variants"]) == set(OwlV2Predictor.VARIANTS)


def test_the_adapter_publishes_every_variant_the_vendor_can_build():
    """A published variant with no geometry, or a geometry with nothing published, is a mismatch
    nothing else would report."""
    from mozo.adapters.owlv2 import OwlV2Predictor

    assert set(OwlV2Predictor.VARIANTS) == set(SPECS)
