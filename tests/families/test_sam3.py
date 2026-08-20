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
from fastapi.testclient import TestClient

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


def test_the_click_head_is_built_at_sam3s_geometry_not_sam2s():
    """1008 over a 72x72 grid, not 1024 over 64x64 -- read back from the module the checkpoint
    has to load into. The mask input is four times the grid, which is where 288 comes from."""
    head = ClickHead()
    assert (head.image_size, head.grid) == (1008, 72)
    assert head.sam_prompt_encoder.image_embedding_size == (72, 72)
    assert head.sam_prompt_encoder.mask_input_size == (288, 288)
    assert head.sam_mask_decoder.num_mask_tokens == 4


def test_the_stability_fallback_is_on():
    """It carries no weights, so a strict load cannot catch it being wrong -- and it only changes
    an answer when the single-mask token is unstable, which is how it survived 23 of 24 parity
    prompts before one image caught it."""
    assert ClickHead().sam_mask_decoder.dynamic_multimask_via_stability is True


def test_the_video_machinery_is_left_behind():
    """Memory attention and mask-memory fusion have nothing to attend to on a still image."""
    for key in ("tracker.transformer.", "tracker.maskmem_backbone.", "tracker.obj_ptr_proj."):
        assert loader._skipped(f"{key}weight")
    for key in ("tracker.sam_prompt_encoder.", "tracker.sam_mask_decoder.", "tracker.no_mem_embed"):
        assert not loader._skipped(key), "the click path itself must survive the filter"


def test_the_transformer_mlp_is_renamed_not_restructured():
    assert loader.rename("sam_mask_decoder.transformer.layers.0.mlp.lin1.weight",
                         loader.TRACKER_RULES) == \
        "sam_mask_decoder.transformer.layers.0.mlp.layers.0.weight"
    assert loader.rename("sam_mask_decoder.transformer.layers.1.mlp.lin2.bias",
                         loader.TRACKER_RULES) == \
        "sam_mask_decoder.transformer.layers.1.mlp.layers.1.bias"


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
    assert set(entry["variants"]) == set(Sam3Predictor.VARIANTS)


def test_the_server_refuses_a_prompted_model_with_no_prompt():
    """The endpoint has to reach SAM 3's branch, take a ``text`` parameter, and reject a missing
    prompt as the caller's error -- before it decodes an image or loads 3.45 GB to find out.

    Asserting on the response rather than on the source: a task the registry declares but the
    endpoint has no branch for is a 501, and that is visible from the outside.
    """
    import inspect

    from mozo import server

    assert get_model_info("sam3")["task_type"] == "concept_segmentation"
    assert "text" in inspect.signature(server.predict).parameters

    client = TestClient(server.app)
    response = client.post(
        "/predict/sam3/sam3", files={"file": ("x.jpg", b"not really a jpeg", "image/jpeg")}
    )
    blank = client.post(
        "/predict/sam3/sam3?text=&text=%20",
        files={"file": ("x.jpg", b"not really a jpeg", "image/jpeg")},
    )
    assert blank.status_code == 400, "prompts that are all whitespace are no prompt at all"
    # The body is deliberately not a JPEG: reaching the prompt complaint rather than a decode
    # failure is what shows the check happens before any work is done.
    assert response.status_code == 400, response.text
    assert "text=" in response.json()["detail"], response.text


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
