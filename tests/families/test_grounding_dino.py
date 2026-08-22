"""Does Grounding DINO actually work.

Unlike the module tests, these load a real checkpoint and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_grounding_dino.py -q

Three promises are protected here. mozo must not change the answer: the vendored package run
directly, with none of mozo between it and the weights, has to agree with what a mozo user
receives. The *name* on a detection must be the phrase the caller passed, not a span decoded out
of the model's tokens -- which is where this package deliberately parts company with upstream. And
the prompt contract must be refused before anything expensive happens.

``tools/verify/grounding_dino.py`` is the bit-exact comparison against the authors' own
implementation; that needs a checkout and does not run here.
"""

from __future__ import annotations

import pytest

from conftest import as_pixelflow_reports, published, require_weights
from mozo.registry import MODEL_REGISTRY
from mozo.vendors.grounding_dino_deploy import SEPARATORS, VARIANTS, caption_for
from mozo.vendors.grounding_dino_deploy.text.tokenizer import Tokenizer

FAMILY = "grounding_dino"
#: Read off the vendor, not retyped. A third variant then gets its licence and artifact coverage
#: without anyone remembering to widen a list here.
ALL = VARIANTS

#: What the fixture photograph -- a desk scene -- reliably contains. Both variants find these;
#: the point is that the family keeps reading the photograph as this scene, not that the two
#: sizes agree on everything.
PROMPTS = ["person", "laptop", "cup"]


@pytest.fixture(scope="module")
def predictor():
    """The smallest variant, built once for the file."""
    require_weights(FAMILY, "tiny")
    from mozo.adapters.grounding_dino import GroundingDinoPredictor
    from mozo.weights import WeightsError

    try:
        return GroundingDinoPredictor("tiny", device="cpu")
    except WeightsError as error:
        pytest.skip(str(error))


# --- the registry and the adapter must agree ---


def test_registry_agrees_with_the_adapter():
    """The catalogue is written twice on purpose; a test is what holds the two in step."""
    from mozo.adapters.grounding_dino import GroundingDinoPredictor

    assert set(MODEL_REGISTRY[FAMILY]["variants"]) == set(GroundingDinoPredictor.VARIANTS)


def test_the_registry_calls_this_open_vocabulary_detection():
    """It shares a task, an endpoint and a response shape with OWLv2. That is the point of it."""
    assert MODEL_REGISTRY[FAMILY]["task_type"] == "open_vocabulary_detection"
    assert MODEL_REGISTRY[FAMILY]["task_type"] == MODEL_REGISTRY["owlv2"]["task_type"]


@pytest.mark.parametrize("variant", ALL)
def test_every_published_variant_ships_its_terms(variant):
    """A checkpoint mozo publishes travels with the licence it is published under."""
    from mozo.weights import companions

    if not published(FAMILY, variant):
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert "LICENSE" in companions(FAMILY, variant)
    assert "NOTICE" in companions(FAMILY, variant)


@pytest.mark.parametrize("variant", ALL)
def test_only_torch_is_published(variant):
    """The input size is not fixed, so there is no graph to export. If that ever changes, this
    test is the reminder that ``EXECUTES`` has to change with it."""
    from mozo.adapters.grounding_dino import GroundingDinoPredictor

    keys = published(FAMILY, variant)
    if not keys:
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert keys == ["torch-fp32"]
    assert GroundingDinoPredictor.EXECUTES == ("torch",)


# --- the prompt contract ---


def test_it_refuses_no_prompt(predictor, image):
    with pytest.raises(ValueError, match="no text was given"):
        predictor.predict(image, [])


def test_it_refuses_a_blank_prompt(predictor, image):
    with pytest.raises(ValueError, match="text was empty"):
        predictor.predict(image, ["person", "  "])


def test_it_refuses_a_prompt_carrying_a_separator(predictor, image):
    """``.`` and ``?`` split concepts, so a prompt containing one would be silently split into
    two and every detection reported against the wrong phrase."""
    with pytest.raises(ValueError, match="may not contain"):
        predictor.predict(image, ["a person. holding a mug"])


def test_it_refuses_more_prompts_than_the_token_budget(predictor, image):
    """Upstream truncates past 256 tokens without saying so, which drops prompts silently."""
    with pytest.raises(ValueError, match="token budget"):
        predictor.predict(image, [f"a photograph of object number {n}" for n in range(80)])


# --- what comes back ---


def test_it_finds_the_scene(predictor, image):
    found = predictor.predict(image, PROMPTS)
    assert len(found) > 0
    names = {row["class_name"] for row in found.to_dict()}
    assert names <= set(PROMPTS), f"{names} contains something nobody asked for"
    assert "person" in names and "laptop" in names


def test_the_name_is_the_callers_phrase_not_a_decoded_span(predictor, image):
    """The divergence from upstream, pinned.

    Upstream decodes the tokens above its text threshold back into a string, which for a
    multi-word prompt can return a fragment. mozo reports the prompt that matched, verbatim.
    """
    prompts = ["a yellow school bus", "a person holding a mug"]
    found = predictor.predict(image, prompts)
    for row in found.to_dict():
        assert row["class_name"] in prompts, f"{row['class_name']!r} is not one of {prompts}"


def test_class_id_indexes_the_prompt_list(predictor, image):
    found = predictor.predict(image, PROMPTS)
    for row in found.to_dict():
        assert PROMPTS[row["class_id"]] == row["class_name"]


def test_a_prompt_matching_nothing_returns_nothing(predictor, image):
    found = predictor.predict(image, ["dinosaur"])
    assert len(found) == 0


def test_detections_come_back_best_first(predictor, image):
    scores = [row["confidence"] for row in predictor.predict(image, PROMPTS).to_dict()]
    assert scores == sorted(scores, reverse=True)


def test_boxes_are_inside_the_source_image(predictor, image):
    height, width = image.shape[:2]
    for row in predictor.predict(image, PROMPTS).to_dict():
        x1, y1, x2, y2 = row["bbox"]
        assert 0 <= x1 < x2 <= width + 1, row["bbox"]
        assert 0 <= y1 < y2 <= height + 1, row["bbox"]


def test_a_higher_threshold_keeps_fewer(predictor, image):
    loose = predictor.predict(image, PROMPTS, threshold=0.2)
    tight = predictor.predict(image, PROMPTS, threshold=0.6)
    assert len(tight) <= len(loose)


# --- mozo must not change the vendor's answer ---


def test_mozo_agrees_with_the_vendor(predictor, image):
    """The adapter sorts, converts and rounds. It must do nothing else."""
    from mozo.vendors.grounding_dino_deploy import SPECS, Predictor
    from mozo.weights import resolve

    vendor = Predictor(resolve(FAMILY, "tiny", "torch-fp32"), SPECS["tiny"], device="cpu")
    raw = sorted(vendor(image, PROMPTS), key=lambda d: -d.score)
    got = predictor.predict(image, PROMPTS).to_dict()

    assert len(raw) == len(got)
    # Both sides through the one helper that owns PixelFlow's rounding, so a change to that
    # policy cannot leave a private copy here silently disagreeing with it.
    boxes, scores = as_pixelflow_reports(
        [d.box for d in raw], [d.score for d in raw], [d.prompt_index for d in raw]
    )
    assert [list(b) for b in boxes] == [row["bbox"] for row in got]
    assert list(scores) == [row["confidence"] for row in got]
    assert [PROMPTS[d.prompt_index] for d in raw] == [row["class_name"] for row in got]


# --- the pieces that can be checked without weights ---


def test_the_caption_is_built_the_way_upstream_expects():
    """Lowercased, joined by ' . ', trailing '.'. The separators are what the phrase map reads."""
    assert caption_for(["Person", " laptop "]) == "person . laptop ."


def test_the_tokenizer_matches_bert_base_uncased():
    """Ids the model was trained against, not merely a plausible split."""
    tokenizer = Tokenizer()
    ids, types, mask = tokenizer.encode("person . laptop . cup.")
    assert ids == [101, 2711, 1012, 12191, 1012, 2452, 1012, 102]
    assert types == [0] * len(ids)
    assert mask == [1] * len(ids)
    assert tokenizer.convert_tokens_to_ids(list(SEPARATORS)) == [101, 102, 1012, 1029]


def test_the_vocabulary_is_the_published_one():
    from mozo.vendors.grounding_dino_deploy.text.tokenizer import vocabulary

    vocab = vocabulary()
    assert len(vocab) == 30522
    assert vocab["[PAD]"] == 0 and vocab["[UNK]"] == 100


def test_phrases_are_isolated_from_each_other():
    """The sub-sentence mask, which is what keeps one prompt from conditioning another."""
    import torch

    from mozo.vendors.grounding_dino_deploy.network import phrase_masks

    tokenizer = Tokenizer()
    ids, _, _ = tokenizer.encode("person . laptop . cup.")
    separators = torch.tensor(tokenizer.convert_tokens_to_ids(list(SEPARATORS)))
    attention, positions, phrases = phrase_masks(torch.tensor([ids]), separators)

    assert phrases[0].shape[0] == 3, "three prompts should give three phrases"
    # Each phrase owns exactly one token here, and no two phrases own the same one.
    assert phrases[0].sum().item() == 3
    assert (phrases[0].sum(dim=0) <= 1).all()
    # 'person' (index 1) must not see 'laptop' (index 3).
    assert not bool(attention[0, 1, 3])
    # Positions restart inside each phrase rather than running across the caption.
    assert positions[0].max().item() < len(ids)


def test_the_resize_preserves_aspect_and_caps_the_long_side():
    """Unlike every other family here, nothing is letterboxed to a square.

    Every case below was read off upstream's own transform rather than derived, including the
    one that looks wrong: a 500x4000 image comes back **1336** wide, three pixels over the 1333
    cap. Upstream rounds the short side to an integer first and then scales the long side from
    it, so the cap binds the ratio rather than the result. Enforcing 1333 here would be a
    reasonable-looking correction that silently moves every box on a panoramic image.
    """
    from mozo.vendors.grounding_dino_deploy.image import resized_size

    assert resized_size(1281, 1920, 800, 1333) == (800, 1199)
    assert resized_size(1920, 1281, 800, 1333) == (1199, 800)
    assert resized_size(500, 4000, 800, 1333) == (167, 1336)
    assert resized_size(4000, 500, 800, 1333) == (1336, 167)
    # Already square and already the target: returned untouched by the early branch.
    assert resized_size(800, 800, 800, 1333) == (800, 800)
    assert resized_size(100, 100, 800, 1333) == (800, 800)
