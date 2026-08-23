"""Does SigLIP 2 actually work.

Unlike the module tests, these load a real checkpoint and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_siglip2.py -q

SigLIP 2 makes every promise CLIP makes -- a vector's shape and scale must not depend on how the
caller passed the input, and the two towers must load independently -- plus one CLIP cannot. Its
scores are per-pair probabilities rather than cosine similarities, so a phrase that describes
nothing in the image must land near zero *on its own*, without a competing phrase to normalise it
against. That is the reason this family exists beside CLIP and it is tested directly.

The text side is where the traps are, and three of them are checked here because they are silent:
the tokenizer lowercases (the published ``tokenizer_config.json`` does not say so and
``AutoTokenizer`` does not do it), every row is padded to a fixed 64, and pooling reads the last
slot -- which is coherent only because the padding is never optional.

``tools/verify/siglip2.py`` is the bit-exact comparison against ``transformers``; it needs the
reference installed and does not run here.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import as_pixelflow_classifications, published, require_weights
from mozo.registry import ENCODES, MODEL_REGISTRY, PROMPTED
from mozo.vendors.siglip2_deploy import CONTEXT, SPECS, VARIANTS
from mozo.vendors.siglip2_deploy.image import preprocess

FAMILY = "siglip2"
#: Read off the vendor, not retyped, so a variant added there is covered here the same day.
ALL = VARIANTS

#: The variant the suite runs on: symmetric towers, smallest checkpoint.
SMALLEST = "base-224"

#: What the fixture photograph -- people around a table with a laptop and mugs -- contains, and
#: what it plainly does not.
PRESENT = "a photo of a group of people around a table"
ABSENT = "a photo of an elephant in a swimming pool"


def build_or_skip():
    """A cold predictor on the smallest variant, or a skip if its weights are not here.

    Written once because two callers need it: the module fixture below, and the lazy-loading test,
    which cannot use that fixture precisely because it needs a predictor nothing has touched yet.
    """
    require_weights(FAMILY, SMALLEST)
    from mozo.adapters.siglip2 import Siglip2Predictor
    from mozo.weights import WeightsError

    try:
        return Siglip2Predictor(SMALLEST, device="cpu")
    except WeightsError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def predictor():
    """The smallest variant, built once for the file."""
    return build_or_skip()


# --- the registry and the adapter must agree ---


def test_registry_agrees_with_the_adapter():
    """The catalogue is written twice on purpose; a test is what holds the two in step."""
    from mozo.adapters.siglip2 import Siglip2Predictor

    assert set(MODEL_REGISTRY[FAMILY]["variants"]) == set(Siglip2Predictor.VARIANTS)


def test_the_registry_calls_this_zero_shot_classification():
    """The task type is what makes the server demand a prompt before loading anything."""
    assert MODEL_REGISTRY[FAMILY]["task_type"] == "zero_shot_classification"
    assert MODEL_REGISTRY[FAMILY]["task_type"] in PROMPTED


def test_it_is_a_family_that_embeds():
    """``/encode`` refuses by family, before the download; the catalogue has to say so."""
    assert ENCODES[FAMILY] == frozenset({"image", "text"})


@pytest.mark.parametrize("variant", ALL)
def test_every_published_variant_ships_its_terms(variant):
    """A checkpoint mozo publishes travels with the licence it is published under.

    Apache-2.0 covers the code and the weights alike here, and the NOTICE is where the Gemma
    vocabulary's provenance is written down -- the one part of this family whose terms are worth
    reading twice.
    """
    from mozo.weights import companions

    if not published(FAMILY, variant):
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert "LICENSE" in companions(FAMILY, variant)
    assert "NOTICE" in companions(FAMILY, variant)


@pytest.mark.parametrize("variant", ALL)
def test_only_torch_is_published(variant):
    """The vendor builds two torch towers and has no graph path; ``auto`` must not offer one.

    If a graph artifact is ever added, this is the reminder that ``EXECUTES`` changes with it.
    """
    from mozo.adapters.siglip2 import Siglip2Predictor

    keys = published(FAMILY, variant)
    if not keys:
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert keys == ["torch-fp32"]
    assert Siglip2Predictor.EXECUTES == ("torch",)


# --- the geometry is written down, and nothing about it is derivable ---


def test_the_variants_geometry_is_written_down_not_inferred():
    """Every spec must be internally consistent, because the strict load is what checks it."""
    for spec in SPECS.values():
        assert spec.vision_width % spec.vision_heads == 0, spec.variant
        assert spec.text_width % spec.text_heads == 0, spec.variant
        assert spec.patches == spec.grid**2, spec.variant


def test_head_dimension_is_not_a_constant():
    """CLIP fixes it at 64 and divides. Deriving heads that way here is wrong for ten variants."""
    assert {s.vision_width // s.vision_heads for s in SPECS.values()} == {64, 72, 96}


def test_the_mlp_is_not_four_times_the_width():
    """``so400m`` is 1152 -> 4304. A derived MLP width is wrong for seven of the fifteen."""
    odd = [s.variant for s in SPECS.values() if s.vision_mlp != 4 * s.vision_width]
    assert sorted(odd) == sorted(s.variant for s in SPECS.values() if s.vision_width == 1152)


def test_the_patch_grid_floors_rather_than_dividing_evenly():
    """``so400m-384`` is 384 over 14, which is 27 patches and six pixels the model never sees."""
    spec = SPECS["so400m-384"]
    assert spec.resolution % spec.patch != 0
    assert spec.grid == 27


def test_giant_towers_are_asymmetric():
    """The text head projects *up* into the width the image tower defines."""
    spec = SPECS["giant-384"]
    assert spec.vision_width == 1536 and spec.text_width == 1152
    assert spec.projection == 1536


# --- what the model refuses ---


def test_it_refuses_no_prompt(predictor, image):
    with pytest.raises(ValueError, match="no text was given"):
        predictor.predict(image, [])


def test_it_refuses_a_blank_prompt(predictor, image):
    with pytest.raises(ValueError, match="text was empty"):
        predictor.predict(image, ["a person", "   "])


def test_it_refuses_a_prompt_over_the_context(predictor, image):
    """Upstream truncates and keeps the end marker. mozo says so instead."""
    with pytest.raises(ValueError, match=f"context is {CONTEXT}"):
        predictor.predict(image, [" ".join(["warehouse"] * 200)])


# --- what it answers ---


def test_every_prompt_comes_back_scored(predictor, image):
    found = predictor.predict(image, [PRESENT, ABSENT])
    assert len(found) == 2


def test_scores_are_probabilities_not_similarities(predictor, image):
    """Sigmoid output: strictly inside the unit interval, and never negative like a cosine."""
    found = predictor.predict(image, [PRESENT, ABSENT])
    for row in found.to_dict():
        assert 0.0 <= row["confidence"] <= 1.0


def test_it_reads_the_photograph(predictor, image):
    found = predictor.predict(image, [PRESENT, ABSENT]).to_dict()
    assert found[0]["class_name"] == PRESENT


def test_an_absent_phrase_scores_near_zero_on_its_own(predictor, image):
    """The property CLIP cannot offer: one phrase, no competitor, and a meaningful answer.

    A cosine similarity has no absolute zero -- an unrelated phrase still lands somewhere in the
    band, which is why CLIP's number needs a complete set of classes before it means anything.
    SigLIP's learned bias puts an unrelated pair at the bottom of the range by itself.
    """
    alone = predictor.predict(image, [ABSENT]).to_dict()
    assert alone[0]["confidence"] < 0.01


def test_class_id_indexes_the_prompt_list(predictor, image):
    prompts = [ABSENT, PRESENT]
    for row in predictor.predict(image, prompts).to_dict():
        assert prompts[row["class_id"]] == row["class_name"]


def test_scores_come_back_best_first(predictor, image):
    scores = [row["confidence"] for row in predictor.predict(image, [ABSENT, PRESENT]).to_dict()]
    assert scores == sorted(scores, reverse=True)


def test_a_threshold_keeps_fewer(predictor, image):
    prompts = [PRESENT, ABSENT]
    assert len(predictor.predict(image, prompts, threshold=0.5)) <= len(
        predictor.predict(image, prompts))


def test_a_single_phrase_may_be_given_as_a_string(predictor, image):
    assert len(predictor.predict(image, PRESENT)) == 1


def test_adding_a_phrase_does_not_move_the_others(predictor, image):
    """A sigmoid scores each pair alone. This is the difference from a softmax, so it is pinned."""
    def score_of(phrase, prompts):
        rows = predictor.predict(image, prompts).to_dict()
        return next(row["confidence"] for row in rows if row["class_name"] == phrase)

    assert score_of(PRESENT, [PRESENT]) == score_of(PRESENT, [PRESENT, ABSENT])


# --- the embedding contracts ---


def test_vectors_are_two_dimensional_even_for_one_input(predictor, image):
    assert predictor.encode_image(image).ndim == 2
    assert predictor.encode_text("a person").ndim == 2


def test_vectors_leave_normalised(predictor, image):
    for vectors in (predictor.encode_image(image), predictor.encode_text(["a person", "a mug"])):
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_both_towers_reach_one_shared_space(predictor, image):
    """A dot product between the two is only meaningful if the widths match."""
    assert predictor.encode_image(image).shape[1] == predictor.encode_text("a person").shape[1]
    assert predictor.encode_image(image).shape[1] == SPECS[SMALLEST].projection


def test_the_towers_load_independently(image):
    """An ingest job must never allocate the text tower -- here, most of the checkpoint."""
    model = build_or_skip()

    assert model._encoder.loaded == ()
    model.encode_text("a person")
    assert model._encoder.loaded == ("text",)
    model.encode_image(image)
    assert set(model._encoder.loaded) == {"vision", "text"}


def test_mozo_agrees_with_the_vendor(predictor, image):
    """The adapter sorts and rounds. It must change nothing else."""
    prompts = [PRESENT, ABSENT]
    raw = predictor._encoder.classify(image, prompts)
    assert predictor.predict(image, prompts).to_dict() == as_pixelflow_classifications(
        raw.numpy(), prompts)


# --- the tokenizer, where the silent traps live ---


@pytest.fixture(scope="module")
def tokenizer():
    from mozo.vendors.siglip2_deploy.text.tokenizer import Tokenizer

    return Tokenizer()


def test_every_row_is_padded_to_the_context(tokenizer):
    """Not a convenience. The tower attends the padding and pools the last slot."""
    tokens = tokenizer(["a", "a much longer phrase about a warehouse"])
    assert tokens.shape == (2, CONTEXT)
    assert tokens[0, -1] == 0 and tokens[1, -1] == 0


def test_the_end_marker_follows_the_prompt(tokenizer):
    row = tokenizer(["a photo of a cat"])[0].tolist()
    assert row[row.index(1) + 1:] == [0] * (CONTEXT - row.index(1) - 1)


def test_it_lowercases(tokenizer):
    """The published config does not ask for this and the model was trained with it."""
    assert tokenizer.encode("A PHOTO OF A CAT") == tokenizer.encode("a photo of a cat")


def test_spaces_become_the_underline_rather_than_splitting_words(tokenizer):
    """Merging runs over the whole phrase, so word boundaries are not merge boundaries."""
    assert tokenizer.pieces[tokenizer.encode("a photo")[1]].startswith("▁")


def test_there_is_no_prefix_space(tokenizer):
    """Most SentencePiece tokenizers prepend one. This one does not, and the ids differ."""
    assert tokenizer.pieces[tokenizer.encode("a photo")[0]] == "a"


def test_byte_fallback_covers_what_the_vocabulary_lacks(tokenizer):
    """A character with no piece becomes its UTF-8 bytes, not a single unknown token."""
    tokens = tokenizer.encode("𐐷")
    assert [tokenizer.pieces[t] for t in tokens[:-1]] == ["<0xF0>", "<0x90>", "<0x90>", "<0xB7>"]


def test_the_byte_table_has_a_hole_in_it(tokenizer):
    """``<0x09>`` is simply absent, which is why the table is read and not computed."""
    assert "<0x00>" in tokenizer.ids and "<0x09>" not in tokenizer.ids


def test_reserved_names_written_out_become_that_token(tokenizer):
    """Upstream matches them before normalising, so this reproduces it rather than correcting it."""
    assert tokenizer.encode("<eos>")[0] == 1
    assert tokenizer.encode("<EOS>")[0] != 1


def test_longest_added_token_wins(tokenizer):
    """``<unused1>`` prefixes ``<unused10>``; matching the short one first would be wrong."""
    assert len(tokenizer.encode("<unused10>")) == 2


def test_case_folding_does_not_depend_on_the_interpreter(tokenizer):
    """Python's case tables belong to the interpreter; this one belongs to the package.

    U+2C2F gained a lowercase mapping after Unicode 13, which Python 3.10 ships. It is one of 95
    codepoints where ``str.lower()`` and the reference disagree, and getting it from the
    interpreter would make a phrase's vectors depend on which Python wrote them.
    """
    assert "\u2c2f".lower() == "\u2c2f", "this Python already folds it; pick another codepoint"
    assert tokenizer.encode("\u2c2f") == tokenizer.encode("\u2c5f")


def test_it_refuses_an_unpaired_surrogate(tokenizer):
    """Valid Python, not valid text. The reference refuses it too, less helpfully."""
    with pytest.raises(ValueError, match="unpaired surrogate"):
        tokenizer.encode("hello\ud800world")


def test_it_refuses_a_phrase_that_does_not_fit(tokenizer):
    with pytest.raises(ValueError, match=f"context is {CONTEXT}"):
        tokenizer([" ".join(["warehouse"] * 200)])


def test_it_refuses_nothing_and_blanks(tokenizer):
    with pytest.raises(ValueError, match="at least one"):
        tokenizer([])
    with pytest.raises(ValueError, match="blank"):
        tokenizer(["   "])


# --- preprocessing ---


def test_the_preprocess_squares_the_image_without_preserving_aspect(image):
    """Not CLIP's short-side resize and centre crop: the whole frame is squashed to a square."""
    assert preprocess(image, 224).shape == (3, 224, 224)


def test_the_preprocess_lands_in_roughly_minus_one_to_one(image):
    pixels = preprocess(image, 224)
    assert -1.0 <= float(pixels.min()) and float(pixels.max()) <= 1.0


def test_the_preprocess_requires_rgb_uint8(image):
    """The reference does not convert; mozo guarantees RGB upstream and this holds it to that."""
    with pytest.raises(ValueError, match="uint8"):
        preprocess(image.astype(np.float32), 224)
    with pytest.raises(ValueError, match="HxWx3"):
        preprocess(image[:, :, 0], 224)
