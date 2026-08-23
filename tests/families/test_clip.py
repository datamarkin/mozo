"""Does CLIP actually work.

Unlike the module tests, these load a real checkpoint and run real inference, so they are slow and
need the published artifacts. They are skipped rather than failed when those are absent.

Point them at a local weights tree with::

    MOZO_BASE_URL=file:///path/to/weights python -m pytest tests/families/test_clip.py -q

CLIP is the first family here that answers with neither a box nor a map, so the promises are
different from every other file in this directory. mozo must not change the vendor's answer, as
always. But it must also keep two contracts that only an embedding model has: a vector's shape and
scale must not depend on how the caller passed the input, and the two towers must load
independently -- an ingest job that only encodes images should never allocate the text tower.

``tools/verify/clip.py`` is the bit-exact comparison against the authors' own implementation; that
needs a checkout and does not run here.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import as_pixelflow_classifications, published, require_weights
from mozo.registry import ENCODES, MODEL_REGISTRY, PROMPTED
from mozo.vendors.clip_deploy import SPECS, VARIANTS

FAMILY = "clip"
#: Read off the vendor, not retyped. The five ResNet variants then get their licence and artifact
#: coverage the day the image tower for them lands, without anyone remembering to widen a list.
ALL = VARIANTS

#: What the fixture photograph -- a desk scene -- contains, and what it plainly does not. CLIP
#: scores every phrase, so the assertion is about the ordering rather than about a threshold.
PRESENT = "a photo of a laptop on a desk"
ABSENT = "a photo of an elephant in a swimming pool"


@pytest.fixture(scope="module")
def predictor():
    """The smallest variant, built once for the file."""
    require_weights(FAMILY, "base")
    from mozo.adapters.clip import ClipPredictor
    from mozo.weights import WeightsError

    try:
        return ClipPredictor("base", device="cpu")
    except WeightsError as error:
        pytest.skip(str(error))


# --- the registry and the adapter must agree ---


def test_registry_agrees_with_the_adapter():
    """The catalogue is written twice on purpose; a test is what holds the two in step."""
    from mozo.adapters.clip import ClipPredictor

    assert set(MODEL_REGISTRY[FAMILY]["variants"]) == set(ClipPredictor.VARIANTS)


def test_the_registry_calls_this_zero_shot_classification():
    """A task of its own. It shares the prompt contract with the detectors and nothing else --
    no boxes, no masks, and every phrase comes back scored rather than only the ones that hit."""
    assert MODEL_REGISTRY[FAMILY]["task_type"] == "zero_shot_classification"
    assert MODEL_REGISTRY[FAMILY]["task_type"] in PROMPTED


def test_it_is_the_family_that_embeds():
    """``ENCODES`` is what ``/encode`` refuses from before loading anything, so it has to name
    both towers -- a missing kind is a 400 on a call that would have worked."""
    assert ENCODES[FAMILY] == frozenset({"image", "text"})


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
    """Upstream ships TorchScript archives and mozo republishes plain fp32 tensors, so there is
    no graph artifact here. If one is ever added, this is the reminder that ``EXECUTES`` has to
    change with it."""
    from mozo.adapters.clip import ClipPredictor

    keys = published(FAMILY, variant)
    if not keys:
        pytest.skip(f"{FAMILY}/{variant} publishes nothing")
    assert keys == ["torch-fp32"]
    assert ClipPredictor.EXECUTES == ("torch",)


# --- the prompt contract ---


def test_it_refuses_no_prompt(predictor, image):
    with pytest.raises(ValueError, match="no text was given"):
        predictor.predict(image, [])


def test_it_refuses_a_blank_prompt(predictor, image):
    with pytest.raises(ValueError, match="text was empty"):
        predictor.predict(image, ["a laptop", "  "])


def test_it_refuses_a_prompt_over_the_context(predictor, image):
    """Upstream's ``tokenize`` can truncate to 76 tokens and overwrite the last with the end
    marker, silently scoring a phrase nobody wrote. This raises instead."""
    with pytest.raises(ValueError, match="token context"):
        predictor.predict(image, [" ".join(["warehouse"] * 100)])


# --- what comes back ---


def test_every_prompt_comes_back_scored(predictor, image):
    """Unlike detection, nothing is filtered out. A classifier that drops a class has not
    classified; the caller asked what each phrase scores and every phrase has an answer."""
    prompts = [PRESENT, ABSENT, "a photo of a bicycle"]
    found = predictor.predict(image, prompts)
    assert len(found) == len(prompts)
    assert {row["class_name"] for row in found.to_dict()} == set(prompts)


def test_it_reads_the_photograph(predictor, image):
    """The one assertion here that is about CLIP being CLIP rather than about mozo's plumbing."""
    found = predictor.predict(image, [ABSENT, PRESENT]).to_dict()
    assert found[0]["class_name"] == PRESENT


def test_class_id_indexes_the_prompt_list(predictor, image):
    """Rows come back reordered, so the id is the only thing tying one to what was asked."""
    prompts = [ABSENT, PRESENT]
    for row in predictor.predict(image, prompts).to_dict():
        assert prompts[row["class_id"]] == row["class_name"]


def test_scores_come_back_best_first(predictor, image):
    scores = [row["confidence"] for row in predictor.predict(image, [ABSENT, PRESENT]).to_dict()]
    assert scores == sorted(scores, reverse=True)


def test_a_threshold_keeps_fewer(predictor, image):
    prompts = [PRESENT, ABSENT]
    assert len(predictor.predict(image, prompts)) == 2
    assert len(predictor.predict(image, prompts, threshold=0.9)) == 0


def test_a_single_phrase_may_be_given_as_a_string(predictor, image):
    assert len(predictor.predict(image, PRESENT)) == 1


def test_adding_a_phrase_does_not_move_the_others(predictor, image):
    """The reason nothing is softmaxed. Each phrase is scored against the image independently,
    so a score means something on its own -- which is what makes a threshold calibratable at all.
    """
    alone = predictor.predict(image, [PRESENT]).to_dict()[0]["confidence"]
    crowded = {row["class_name"]: row["confidence"]
               for row in predictor.predict(image, [PRESENT, ABSENT, "a photo of a cat"]).to_dict()}
    assert crowded[PRESENT] == alone


# --- the embeddings ---


def test_vectors_are_two_dimensional_even_for_one_input(predictor, image):
    """A shape that depends on whether the caller passed a list is the same trap as a response
    shape that depends on a query parameter."""
    assert predictor.encode_image(image).shape == (1, SPECS["base"].embed_dim)
    assert predictor.encode_text("a laptop").shape == (1, SPECS["base"].embed_dim)
    assert predictor.encode_image([image, image]).shape == (2, SPECS["base"].embed_dim)


def test_vectors_leave_normalised(predictor, image):
    """So a dot product between any two of them is a cosine, with no convention to document.
    Two callers normalising differently is a class of bug that never raises."""
    for vectors in (predictor.encode_image(image), predictor.encode_text([PRESENT, ABSENT])):
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_both_towers_reach_one_shared_space(predictor, image):
    """The whole premise: an image vector and a phrase vector are comparable. If the two towers
    landed in different spaces every dot product would still compute, and mean nothing."""
    scored = predictor.predict(image, [PRESENT, ABSENT]).to_dict()
    by_hand = predictor.encode_image(image) @ predictor.encode_text([PRESENT, ABSENT]).T

    assert by_hand.shape == (1, 2)
    assert as_pixelflow_classifications(by_hand[0], [PRESENT, ABSENT]) == scored


def test_the_towers_load_independently(image):
    """Advertised on the adapter, so it is checked: encoding phrases must not build the image
    tower. It is the difference between a query service holding 63.4M parameters and 151.3M.
    """
    require_weights(FAMILY, "base")
    from mozo.adapters.clip import ClipPredictor
    from mozo.weights import WeightsError

    try:
        model = ClipPredictor("base", device="cpu")
    except WeightsError as error:
        pytest.skip(str(error))

    assert model._encoder.loaded == (), "a tower was built before anything was encoded"
    model.encode_text("a laptop")
    assert model._encoder.loaded == ("text",), "encoding a phrase built the image tower"
    model.encode_image(image)
    assert set(model._encoder.loaded) == {"vision", "text"}


# --- mozo must not change the vendor's answer ---


def test_mozo_agrees_with_the_vendor(predictor, image):
    """The adapter sorts, converts and rounds. It must do nothing else."""
    from mozo.vendors.clip_deploy import Encoder
    from mozo.weights import resolve

    prompts = [PRESENT, ABSENT, "a photo of a bicycle"]
    vendor = Encoder(resolve(FAMILY, "base", "torch-fp32"), SPECS["base"], device="cpu")
    raw = vendor.classify(image, prompts)
    got = predictor.predict(image, prompts).to_dict()

    # Both sides through the one helper that owns PixelFlow's rounding and ordering, so a change
    # to either policy cannot leave a private copy here silently disagreeing with it. Whole rows
    # rather than confidences: this then also pins the rank and the label-to-id mapping.
    assert got == as_pixelflow_classifications(raw.numpy(), prompts)


# --- the pieces that can be checked without weights ---


def test_the_tokenizer_matches_the_published_one():
    """Ids the model was trained against, not merely a plausible split. ``49406`` and ``49407``
    are the start and end markers, and everything after the end marker is zero padding."""
    from mozo.vendors.clip_deploy.text.tokenizer import CONTEXT_LENGTH, Tokenizer

    tokens = Tokenizer()(["a diagram"])
    assert tokens.shape == (1, CONTEXT_LENGTH)
    assert tokens[0, :4].tolist() == [49406, 320, 22697, 49407]
    assert not tokens[0, 4:].any()


def test_the_end_marker_is_the_highest_id_in_the_vocabulary():
    """Pooling finds it with ``argmax`` over the ids rather than by counting positions, which
    only works because it outranks every real token and zero padding is the lowest id there is."""
    from mozo.vendors.clip_deploy.text.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    tokens = tokenizer(["a photograph of a warehouse"])
    assert int(tokens.argmax()) == int((tokens != 0).sum()) - 1
    assert tokenizer.end_id > tokenizer.start_id


def test_digits_are_tokenized_one_at_a_time():
    """The split pattern is ``[\\p{N}]``, so "2024" is four tokens rather than one and a numeric
    prompt eats the 77-token context faster than its length suggests."""
    from mozo.vendors.clip_deploy.text.tokenizer import Tokenizer

    tokens = Tokenizer()(["2024"])[0]
    assert int((tokens != 0).sum()) == 6, "start, four digits, end"


def test_the_preprocess_resizes_the_short_side_and_crops_the_centre():
    """``Resize(224)`` scales the short side and keeps the aspect ratio; ``Resize((224, 224))``
    squashes the picture. The two differ by more than three units per channel on a real
    photograph, and neither raises."""
    from mozo.vendors.clip_deploy.image import preprocess

    tall = np.zeros((400, 200, 3), dtype=np.uint8)
    wide = np.zeros((200, 400, 3), dtype=np.uint8)
    assert preprocess(tall, 224).shape == (3, 224, 224)
    assert preprocess(wide, 224).shape == (3, 224, 224)
    assert preprocess(tall, 336).shape == (3, 336, 336)


def test_the_variants_geometry_is_written_down_not_inferred():
    """Upstream reads the shapes out of the state dict to decide what to build. Writing them
    down is what lets the strict load check them -- a spec that is inferred cannot be wrong."""
    assert SPECS["base"].embed_dim == 512 and SPECS["base"].resolution == 224
    assert SPECS["large"].embed_dim == 768 and SPECS["large"].patch == 14
    assert SPECS["large-336"].resolution == 336
    # Upstream's own rule, not a per-variant number: 64 channels per head.
    assert SPECS["large"].vision_heads == SPECS["large"].vision_width // 64
