"""SAM 3's contracts, checked without weights.

The numbers SAM 3 produces are guarded by ``tools/verify/sam3.py``, which needs the 3.45 GB
checkpoint. Everything here runs in a second and holds the shapes of the package steady: what the
tokenizer emits, what preprocessing produces, which keys the checkpoint translation expects, and
that the caches bound themselves and evict the right entry.

These are the invariants that a refactor breaks silently and a parity gate only catches if
somebody has the weights to hand.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mozo.registry import get_model_info
from mozo.vendors.sam3_deploy import checkpoint as loader
from mozo.vendors.sam3_deploy.config import SPEC, TEXT
from mozo.vendors.sam3_deploy.click import ClickHead
from mozo.vendors.sam3_deploy.image import preprocess, preprocess_click, to_model_coords
from mozo.vendors.sam3_deploy.predictor import (
    CLICK_CACHE,
    IMAGE_CACHE,
    PROMPT_CACHE,
    Segmenter,
    instances,
)
from mozo.vendors.sam3_deploy.text import Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer()


# --- the prompt ------------------------------------------------------------------------------

#: Recorded from the published model's own tokenizer. These ids are the contract: the text tower's
#: embedding table is indexed by them, so a change here is a change of meaning, not of formatting.
KNOWN_IDS = {
    "cow": [49406, 9706, 49407],
    "yellow school bus": [49406, 4481, 1228, 2840, 49407],
    "dog's tail": [49406, 1929, 568, 4132, 49407],
    "": [49406, 49407],
}


@pytest.mark.parametrize("prompt,expected", KNOWN_IDS.items(), ids=list(KNOWN_IDS))
def test_the_tokenizer_emits_the_ids_the_weights_were_trained_on(tokenizer, prompt, expected):
    ids = tokenizer([prompt])[0]
    assert ids[: len(expected)].tolist() == expected
    assert (ids[len(expected):] == 0).all(), "prompts pad on the right with zero"


def test_case_does_not_change_a_prompt(tokenizer):
    """SAM 3 builds its tokenizer with ``clean="lower"``; uppercase must not re-segment."""
    assert torch.equal(tokenizer(["A Red Hat"]), tokenizer(["a red hat"]))


def test_a_prompt_too_long_is_truncated_but_still_terminated(tokenizer):
    ids = tokenizer([" ".join(["word"] * 200)])[0]
    assert len(ids) == TEXT.context_length
    assert ids[-1] == tokenizer.end_id, "a truncated prompt still has to end"


def test_padding_is_what_marks_padding(tokenizer):
    """``ids == 0`` is the attention mask, so nothing real may tokenize to zero."""
    ids = tokenizer(["cow", "a much longer phrase about a cow"])
    assert (ids[0][:3] != 0).all()
    assert (ids[0][3:] == 0).all()


# --- preprocessing ---------------------------------------------------------------------------

@pytest.mark.parametrize("height,width", [(1281, 1920), (640, 640), (100, 3000)])
def test_preprocessing_squashes_to_a_square(height, width):
    """SAM 3 distorts the aspect ratio rather than letterboxing -- there is no padding to undo."""
    batch = preprocess(np.zeros((height, width, 3), dtype=np.uint8))
    side = SPEC.trunk.image_size
    assert batch.shape == (1, 3, side, side)


def test_preprocessing_normalises_to_minus_one_and_one():
    """Mean and standard deviation are 0.5, not ImageNet's -- black and white map to the ends."""
    black = preprocess(np.zeros((64, 64, 3), dtype=np.uint8))
    white = preprocess(np.full((64, 64, 3), 255, dtype=np.uint8))
    assert torch.allclose(black, torch.full_like(black, -1.0))
    assert torch.allclose(white, torch.full_like(white, 1.0))


def test_preprocessing_refuses_anything_that_is_not_rgb():
    with pytest.raises(ValueError, match="HxWx3"):
        preprocess(np.zeros((64, 64), dtype=np.uint8))


# --- the click path ------------------------------------------------------------------------

def test_the_two_heads_do_not_preprocess_alike():
    """The concept path rounds the resize back to uint8 and multiplies by 1/255; the click path
    does neither. This is the difference that was worth 9e-03 of predicted IoU, so it is pinned
    here rather than left to a gate that needs 3.45 GB."""
    image = np.random.default_rng(0).integers(0, 256, (321, 517, 3), dtype=np.uint8)
    concept, click = preprocess(image), preprocess_click(image)
    assert concept.shape == click.shape
    assert not torch.equal(concept, click), "the two transforms must not collapse into one"
    # Half a grey level in [-1, 1] space, which is what a uint8 round-trip costs.
    assert (concept - click).abs().max() < 2.0 / 255


def test_click_preprocessing_squares_and_normalises_like_its_sibling():
    batch = preprocess_click(np.zeros((100, 3000, 3), dtype=np.uint8))
    side = SPEC.trunk.image_size
    assert batch.shape == (1, 3, side, side)
    assert torch.allclose(batch, torch.full_like(batch, -1.0))


def test_click_preprocessing_refuses_anything_that_is_not_rgb():
    with pytest.raises(ValueError, match="HxWx3"):
        preprocess_click(np.zeros((64, 64), dtype=np.uint8))


def test_prompt_coordinates_squash_with_the_pixels():
    """No letterboxing means x and y scale independently, with no padding offset to undo."""
    side = SPEC.trunk.image_size
    got = to_model_coords(np.array([[100.0, 50.0]]), (200, 400))
    assert torch.allclose(got, torch.tensor([[side * 0.25, side * 0.25]]))


def test_the_click_head_is_built_at_sam3s_geometry():
    """1008 over a 72x72 grid, and a mask input four times the grid, which is where 288 comes
    from. Every value is ``transformers/models/sam3_tracker``'s own config default, and every
    one is confirmed by the published checkpoint loading strict."""
    head = ClickHead()
    assert (head.image_size, head.grid) == (1008, 72)
    assert head.prompt_encoder.grid == 72
    assert head.prompt_encoder.mask_input_size == 288
    assert head.mask_decoder.num_mask_tokens == 4


def test_the_click_geometry_is_the_trunks():
    """``ClickSpec`` restates the trunk's square and the neck's width so the head reads one spec.
    They are the same physical numbers, and nothing about a shape mismatch would catch them
    drifting: a click would simply land on the wrong feature. So they are pinned."""
    from mozo.vendors.sam3_deploy.config import CLICK, SPEC

    assert CLICK.image_size == SPEC.trunk.image_size
    assert CLICK.patch == SPEC.trunk.patch
    assert CLICK.grid == SPEC.trunk.grid
    assert CLICK.hidden == SPEC.fpn_hidden


def test_asking_for_one_mask_consults_the_candidates():
    """Asking for one mask does not mean taking token 0 whatever it looks like. When that token
    is unstable the decoder returns the best candidate instead -- the behaviour that survived 23
    of 24 parity prompts before one image caught it missing."""
    from mozo.vendors.sam3_deploy.click.decoder import MaskDecoder

    decoder = MaskDecoder()
    # Token 0 is a coin-flip mask (stability near zero); candidate 2 is decisive.
    masks = torch.zeros(1, 1, 4, 8, 8)
    masks[0, 0, 0] = torch.linspace(-0.04, 0.04, 64).reshape(8, 8)
    masks[0, 0, 3] = 5.0
    iou = torch.tensor([[[0.9, 0.1, 0.2, 0.8]]])

    chosen, score = decoder._stable(masks, iou)
    assert torch.equal(chosen[0, 0, 0], masks[0, 0, 3]), "an unstable token 0 falls back"
    assert score.item() == pytest.approx(0.8)


def test_a_stable_single_token_is_kept():
    from mozo.vendors.sam3_deploy.click.decoder import MaskDecoder

    decoder = MaskDecoder()
    masks = torch.full((1, 1, 4, 8, 8), 5.0)
    masks[0, 0, 3] = -5.0
    iou = torch.tensor([[[0.4, 0.1, 0.2, 0.9]]])
    chosen, score = decoder._stable(masks, iou)
    assert torch.equal(chosen[0, 0, 0], masks[0, 0, 0]), "a decisive token 0 is not second-guessed"
    assert score.item() == pytest.approx(0.4)


def test_several_prompt_sets_run_in_one_call():
    """The batch lands on the prompt axis: one image, several prompt sets. Putting it on the
    image axis instead makes the second set fail to broadcast against the one image it asks
    about, which is what ``predict``'s documented ``(B, N, 2)`` needs."""
    head = ClickHead().eval()
    click = [torch.zeros(1, 256, 288, 288), torch.zeros(1, 256, 144, 144),
             torch.zeros(1, 256, 72, 72)]
    points = torch.tensor([[[500.0, 400.0]], [[300.0, 200.0]]])
    labels = torch.tensor([[1], [1]], dtype=torch.int32)

    masks, iou = head(click, points, labels, None, True)
    assert masks.shape[0] == 2 and iou.shape[0] == 2


def test_the_video_machinery_is_left_behind():
    """Memory attention and mask-memory fusion have nothing to attend to on a still image."""
    for key in ("tracker.transformer.", "tracker.maskmem_backbone.", "tracker.obj_ptr_proj."):
        assert loader._skipped(f"{key}weight")
    for key in ("tracker.sam_prompt_encoder.", "tracker.sam_mask_decoder.", "tracker.no_mem_embed"):
        assert not loader._skipped(key), "the click path itself must survive the filter"


@pytest.mark.parametrize("meta,ours", [
    ("sam_mask_decoder.transformer.layers.0.mlp.lin1.weight",
     "mask_decoder.transformer.layers.0.mlp.layers.0.weight"),
    ("sam_mask_decoder.transformer.layers.1.self_attn.out_proj.bias",
     "mask_decoder.transformer.layers.1.self_attn.o_proj.bias"),
    ("sam_mask_decoder.transformer.layers.0.norm3.weight",
     "mask_decoder.transformer.layers.0.layer_norm3.weight"),
    ("sam_mask_decoder.transformer.norm_final_attn.weight",
     "mask_decoder.transformer.layer_norm_final_attn.weight"),
    ("sam_mask_decoder.iou_prediction_head.layers.1.weight",
     "mask_decoder.iou_prediction_head.layers.1.weight"),
    ("sam_mask_decoder.output_hypernetworks_mlps.2.layers.2.bias",
     "mask_decoder.output_hypernetworks_mlps.2.layers.2.bias"),
    ("sam_mask_decoder.output_upscaling.1.weight", "mask_decoder.upscale_layer_norm.weight"),
    ("sam_prompt_encoder.mask_downscaling.4.bias", "prompt_encoder.mask_embed.layer_norm2.bias"),
    ("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
     "prompt_encoder.shared_embedding.positional_embedding"),
])
def test_the_click_rename_table_only_renames(meta, ours):
    """The checkpoint follows ``facebookresearch/sam3``; this package follows ``transformers``.
    Every rule moves a name or reindexes a Sequential whose members we name -- none of them is
    the place a number changes. The parity gate proves that; this keeps the table readable.
    """
    assert loader.rename(meta, loader.CLICK_RULES) == ours


def test_a_box_is_two_corners_with_reserved_labels():
    """There is no box input. This is what a box actually becomes."""
    coords, marks = _segmenter()._prompt(
        None, None, np.array([0.0, 0.0, 400.0, 200.0]), (200, 400))
    assert coords.shape == (1, 2, 2)
    assert marks.tolist() == [[2, 3]]


def test_a_box_and_its_points_arrive_corners_first():
    """The encoder adds a different learned embedding per position, so the order is meaning."""
    coords, marks = _segmenter()._prompt(
        np.array([[100.0, 100.0]]), np.array([1]),
        np.array([0.0, 0.0, 400.0, 200.0]), (200, 400))
    assert marks.tolist() == [[2, 3, 1]], "corners first, then the clicks"
    assert coords.shape == (1, 3, 2)


def test_points_and_labels_have_to_agree_in_number():
    with pytest.raises(ValueError, match="2 points but 1 labels"):
        _segmenter()._prompt(
            np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1]), None, (200, 400))


@pytest.mark.parametrize("kwargs,message", [
    ({}, "a prompt is required"),
    ({"labels": np.array([1])}, "a prompt is required"),   # labels alone are not a prompt
    ({"points": np.zeros((1, 2))}, "points and labels go together"),
    ({"labels": np.array([1]), "boxes": np.zeros(4)}, "points and labels go together"),
])
def test_a_click_without_a_prompt_is_refused(kwargs, message):
    """Checked before ``self`` or the image is touched, so it needs neither weights nor pixels."""
    with pytest.raises(ValueError, match=message):
        Segmenter.segment(None, None, **kwargs)


# --- the checkpoint translation ----------------------------------------------------------------

def test_the_rename_table_is_applied_in_order_and_leaves_the_rest_alone():
    assert loader.rename("trunk.ln_pre.weight", loader.VISION_RULES) == "trunk.layer_norm.weight"
    assert loader.rename("nothing.matches.this", loader.VISION_RULES) == "nothing.matches.this"


@pytest.mark.parametrize("key", loader.UNUSED)
def test_weights_this_package_does_not_build_are_skipped(key):
    """Each of these is loaded by upstream and either discarded or unreachable here."""
    assert loader._skipped(f"{key}.weight")


def test_a_file_that_is_not_a_sam3_checkpoint_says_so():
    """An empty tower means the wrong file, not a missing layer -- so it raises rather than
    handing back an empty state dict that would fail later and further away."""
    with pytest.raises(KeyError, match="SAM 3 checkpoint"):
        loader.vision_state_dict({"something.else": torch.zeros(1)})


@pytest.mark.parametrize("zipfile_format", [True, False])
def test_a_checkpoint_loads_whichever_way_it_was_serialised(tmp_path, zipfile_format):
    """Reading is mapped for the memory it saves, and mapping refuses the pre-1.6 layout.

    A caller may hand ``Sam3Predictor`` a checkpoint of their own, so the old layout has to keep
    working -- and produce the same tensors, since only the cost of getting them differs.
    """
    payload = {"model": {"a": torch.randn(8), "b": torch.randn(4, 4)}}
    path = tmp_path / "checkpoint.pth"
    torch.save(payload, path, _use_new_zipfile_serialization=zipfile_format)

    got = loader.load_state_dict(path)

    assert sorted(got) == ["a", "b"], "the 'model' envelope is unwrapped either way"
    assert all(torch.equal(payload["model"][key], got[key]) for key in payload["model"])


# --- what the caller gets ----------------------------------------------------------------------

def _result(scores: list[float]) -> dict[str, torch.Tensor]:
    """A forward pass's shape, with presence pinned high so ``scores`` decides alone."""
    queries = len(scores)
    return {
        "logits": torch.logit(torch.tensor([scores])),
        "presence": torch.full((1, 1), 20.0),
        "boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]] * queries]),
        "masks": torch.zeros(1, queries, 8, 8),
    }


def test_only_instances_above_the_threshold_come_back():
    found = instances(_result([0.9, 0.6, 0.1]), (40, 60), threshold=0.5)[0]
    assert len(found["scores"]) == 2
    assert found["masks"].shape == (2, 40, 60)
    assert found["boxes"].shape == (2, 4)


def test_finding_nothing_is_an_answer_and_not_an_error():
    """Ask a picture of an office for "cow" and every query should fall below the threshold."""
    found = instances(_result([0.1, 0.1, 0.1]), (40, 60))[0]
    assert len(found["scores"]) == 0
    assert found["masks"].shape == (0, 40, 60)


def test_presence_gates_every_score():
    """Without the presence term an absent concept still returns the queries' best guesses."""
    absent = _result([0.9, 0.9, 0.9])
    absent["presence"] = torch.full((1, 1), -20.0)
    assert len(instances(absent, (40, 60))[0]["scores"]) == 0


def test_boxes_come_back_in_the_source_image_pixels():
    found = instances(_result([0.9]), (100, 200))[0]
    # (0.5, 0.5, 0.2, 0.2) normalised is the middle fifth, so x spans 80..120 of 200.
    assert torch.allclose(found["boxes"][0], torch.tensor([80.0, 40.0, 120.0, 60.0]))


# --- the caches ---------------------------------------------------------------------------------

class _Recorder:
    """Stands in for the encoders, counting how often the cache actually misses."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *_):
        self.calls += 1
        return {"called": self.calls}


def _segmenter():
    """A Segmenter with its caches and its device, built without touching 3.45 GB of weights.

    ``__new__`` rather than a constructor because ``Segmenter.__init__`` loads the checkpoint.
    Everything the weightless tests reach for is set here, so there is one of these rather than
    one per group of attributes.
    """
    from collections import OrderedDict
    from threading import Lock

    blank = Segmenter.__new__(Segmenter)
    blank._images, blank._clicks = OrderedDict(), OrderedDict()
    blank._prompts, blank._lock = OrderedDict(), Lock()
    blank.device = "cpu"
    return blank


def test_the_image_cache_holds_what_it_says_it_holds():
    segmenter, encoder = _segmenter(), _Recorder()
    for index in range(IMAGE_CACHE + 2):
        segmenter._remember(segmenter._images, bytes([index]), encoder, IMAGE_CACHE)
    assert len(segmenter._images) == IMAGE_CACHE
    assert encoder.calls == IMAGE_CACHE + 2


def test_a_second_prompt_on_a_held_image_does_not_re_encode():
    segmenter, encoder = _segmenter(), _Recorder()
    for _ in range(3):
        segmenter._remember(segmenter._images, b"same", encoder, IMAGE_CACHE)
    assert encoder.calls == 1, "the whole point of the image cache"


def test_the_least_recently_used_entry_is_the_one_evicted():
    segmenter, encoder = _segmenter(), _Recorder()
    for key in (b"a", b"b"):
        segmenter._remember(segmenter._images, key, encoder, 2)
    segmenter._remember(segmenter._images, b"a", encoder, 2)  # touch it
    segmenter._remember(segmenter._images, b"c", encoder, 2)  # evicts b, not a
    assert set(segmenter._images) == {b"a", b"c"}


def test_a_repeated_prompt_is_encoded_once_and_kept_apart_from_the_images():
    """An encoded phrase is 33 KB against an image's 223 MB, which is why it gets its own,
    larger cache rather than competing with images for a slot in theirs."""
    segmenter, encoder = _segmenter(), _Recorder()
    for _ in range(3):
        segmenter._remember(segmenter._prompts, "cow", encoder, PROMPT_CACHE)
    segmenter._remember(segmenter._images, b"pixels", encoder, IMAGE_CACHE)

    assert encoder.calls == 2, "one encode for the phrase, one for the image"
    assert set(segmenter._prompts) == {"cow"} and set(segmenter._images) == {b"pixels"}
    assert PROMPT_CACHE > IMAGE_CACHE


def test_the_two_heads_do_not_share_a_cache():
    """They cannot: their preprocessing differs, so the same photograph produces two different
    encodes. One cache keyed on pixels alone would serve the click head the concept head's
    features -- which is wrong by 3.5 standard deviations of the feature map, not by a rounding."""
    segmenter, encoder = _segmenter(), _Recorder()
    segmenter._remember(segmenter._images, b"same", encoder, IMAGE_CACHE)
    segmenter._remember(segmenter._clicks, b"same", encoder, CLICK_CACHE)
    assert encoder.calls == 2, "the same key in the two caches must not collide"
    assert len(segmenter._images) == len(segmenter._clicks) == 1


def test_the_click_cache_is_bounded_and_evicts_least_recently_used():
    segmenter, encoder = _segmenter(), _Recorder()
    for index in range(CLICK_CACHE + 2):
        segmenter._remember(segmenter._clicks, bytes([index]), encoder, CLICK_CACHE)
    assert len(segmenter._clicks) == CLICK_CACHE
    assert encoder.calls == CLICK_CACHE + 2


# --- the registry ------------------------------------------------------------------------------

def test_registry_agrees_with_the_adapter():
    """The variant list is written twice -- here and in the adapter -- so that answering "what
    exists" needs no torch import. This is what holds the two copies in step."""
    from mozo.adapters.sam3 import Sam3Predictor

    entry = get_model_info("sam3")
    assert entry["adapter_class"] == "Sam3Predictor"
    assert entry["module"] == "mozo.adapters.sam3"
    assert entry["task_type"] == "concept_segmentation"
    assert set(entry["variants"]) == set(Sam3Predictor.VARIANTS)


@pytest.mark.parametrize("empty", ["", "   ", ["car", ""], ["  "]])
def test_an_empty_prompt_is_refused_rather_than_guessed(empty):
    """SAM 3 will encode the empty string and return whatever is most salient, which is not what
    an empty prompt means. One empty concept among several is still an empty concept, so the
    whole call is refused rather than that prompt quietly dropped -- a caller who asked for three
    classes and got two would have no way to tell. The refusal happens before ``self`` or the
    image is touched, so it needs neither weights nor a real instance."""
    from mozo.adapters.sam3 import Sam3Predictor

    with pytest.raises(ValueError, match="concept to look for"):
        Sam3Predictor.predict(None, None, empty)


def test_asking_for_nothing_is_refused_too():
    from mozo.adapters.sam3 import Sam3Predictor

    with pytest.raises(ValueError, match="no text was given"):
        Sam3Predictor.predict(None, None, [])


class _FixedSegmenter:
    """Stands in for the vendor Segmenter, returning one instance per prompt asked for."""

    def __init__(self):
        self.prompts = []

    def predict(self, pixels, text, threshold=0.5):
        self.prompts.append(text)
        return {
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "scores": torch.tensor([0.9]),
            "masks": torch.zeros(1, 8, 8, dtype=torch.bool),
        }


def test_the_prompt_list_is_the_vocabulary():
    """With several concepts the prompts become the class list and ``class_ids`` index it --
    which is what that field is for, and what a single prompt leaves unused. Asserted on the
    result rather than on the source, so renaming a local cannot break it and passing the wrong
    vocabulary cannot pass it."""
    from mozo.adapters.sam3 import Sam3Predictor

    model = Sam3Predictor.__new__(Sam3Predictor)
    model._segmenter = _FixedSegmenter()
    found = model.predict(np.zeros((8, 8, 3), dtype=np.uint8), ["car", "person", "dog"])

    assert model._segmenter.prompts == ["car", "person", "dog"], "one decode per concept, in order"
    assert [int(d.class_id) for d in found] == [0, 1, 2]
    assert [d.class_name for d in found] == ["car", "person", "dog"]


def test_one_concept_still_comes_back_as_one_class():
    """The single-prompt path is the common one and must be unchanged by the list support."""
    from mozo.adapters.sam3 import Sam3Predictor

    model = Sam3Predictor.__new__(Sam3Predictor)
    model._segmenter = _FixedSegmenter()
    found = model.predict(np.zeros((8, 8, 3), dtype=np.uint8), "car")

    assert model._segmenter.prompts == ["car"]
    assert [int(d.class_id) for d in found] == [0]
    assert [d.class_name for d in found] == ["car"]


# --- the graph encoder -------------------------------------------------------------------------

class _StubRunner:
    """A CoreML runner's surface, without CoreML: named outputs in declaration order."""

    outputs = ("level0", "level1", "level2", "positions")

    def __call__(self, batch):
        sides = (288, 144, 72)
        return (*(np.zeros((1, 256, s, s), dtype=np.float32) for s in sides),
                np.full((1, 256, 72, 72), 7.0, dtype=np.float32))


def test_the_graph_encoder_answers_the_vision_encoders_question():
    """Its whole job is to be substitutable for ``VisionEncoder.forward``, keys and order alike."""
    from mozo.adapters.sam3 import GraphVision

    got = GraphVision(_StubRunner(), "cpu")(torch.zeros(1, 3, 1008, 1008))

    assert set(got) == {"concept", "positions"}
    assert [tuple(level.shape[-2:]) for level in got["concept"]] == [(288, 288), (144, 144), (72, 72)], \
        "finest first: the concept head reads levels[-1] as the grid it attends over"
    assert got["positions"].shape == (1, 256, 72, 72)
    assert all(isinstance(level, torch.Tensor) for level in got["concept"])


def test_the_graph_encoder_refuses_the_click_stack():
    """It carries the concept stack alone; answering with it would mask the wrong pixels."""
    from mozo.adapters.sam3 import GraphVision

    with pytest.raises(ValueError, match="concept stack only"):
        GraphVision(_StubRunner(), "cpu")(torch.zeros(1, 3, 1008, 1008), stacks=("click",))


def test_the_graphs_outputs_are_the_levels_the_neck_survives_the_scalp_with():
    """``LEVELS`` names the graph's outputs, and the neck decides how many there are."""
    from mozo.adapters.sam3 import LEVELS

    assert len(LEVELS) == len(SPEC.scale_factors) - SPEC.scalp


def test_a_supplied_vision_encoder_turns_the_click_path_off():
    """Refusing beats answering from the concept pyramid, which reads different pixels."""
    blank = _segmenter()
    blank.click = None
    with pytest.raises(RuntimeError, match="concept path only"):
        blank.encode_click(np.zeros((8, 8, 3), dtype=np.uint8))
