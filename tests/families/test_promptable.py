"""The contract every promptable family shares, checked against every one of them.

Two families implement it today and two more are planned. Rather than pick one and test the
shared code under its name, the families are *discovered* from the registry — the same move
``tests/test_vendor_agreement.py`` makes one level down, and for the same stated reason: a
hand-maintained list is how the fourth family lands with a green suite and no coverage.

So MobileSAM and EfficientViT-SAM will arrive already covered here, by adding a registry entry.
What stays in a family's own file is what is genuinely family-specific — SAM 2's four variants,
EdgeTAM's trunk geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from mozo.adapters._promptable import bounds
from mozo.registry import MODEL_REGISTRY, get_model_info
from mozo.weights import WeightsError

#: Every family the registry says answers a click. Discovered, not listed.
FAMILIES = sorted(
    family for family, entry in MODEL_REGISTRY.items()
    if entry["task_type"] == "promptable_segmentation"
)


def test_more_than_one_family_was_discovered():
    """Discovery that silently found nothing would make every test below vacuously pass."""
    assert len(FAMILIES) >= 2, f"expected several promptable families, found {FAMILIES}"


@pytest.fixture(params=FAMILIES, scope="module")
def family(request):
    """One promptable family's name."""
    return request.param


@pytest.fixture(scope="module")
def model(family):
    """That family's default variant, loaded — or a skip. Published is not the same as present."""
    import importlib

    entry = get_model_info(family)
    predictor = getattr(importlib.import_module(entry["module"]), entry["adapter_class"])
    try:
        return predictor(device="cpu")
    except (WeightsError, FileNotFoundError) as error:
        pytest.skip(f"{family} weights unavailable: {error}")


@pytest.fixture
def click(image):
    """A click on the fixture photograph, in its own pixels."""
    height, width = image.shape[:2]
    return dict(points=np.array([[0.62 * width, 0.55 * height]]), labels=np.array([1]))


# --- the box a mask gets ------------------------------------------------------------------------

def test_a_mask_with_no_foreground_still_gets_a_row():
    """Dropping it would silently unalign masks, scores and boxes from what the model returned.
    A box around nothing is zeros, which is the only honest answer for it."""
    masks = np.zeros((2, 4, 4), bool)
    masks[0, 1:3, 2:4] = True
    got = bounds(masks)
    assert got.shape == (2, 4)
    assert list(got[0]) == [2, 1, 4, 3]
    assert list(got[1]) == [0, 0, 0, 0]


# --- what every promptable adapter hands back ----------------------------------------------------

def test_a_click_returns_its_candidates_ranked_best_first(model, image, click):
    """The model emits them in its own order -- the mask tokens specialise towards whole, part
    and subpart -- so the highest-scoring candidate is not the one in slot zero. Every other
    family in mozo hands back ranked detections, so these are ranked too."""
    scores = [row.confidence for row in model.predict(image, **click)]
    assert len(scores) == 3
    assert scores == sorted(scores, reverse=True)


def test_ranking_keeps_each_mask_with_its_own_score(model, image, click):
    """Sorting scores and masks apart would be invisible in the score column and wrong in every
    mask, so the two are checked against each other rather than each on its own."""
    found = model.predict(image, **click)
    raw = model._segmenter.predict(image, **click)

    for row in found:
        mask = np.asarray(row.masks[0] if isinstance(row.masks, list) else row.masks)
        candidate = int(np.argmin(np.abs(raw.scores[0] - row.confidence)))
        assert mask.sum() == raw.masks[0, candidate].sum()


def test_nothing_names_what_was_clicked_unless_the_caller_does(model, image, click):
    """A click does not say what it clicked. PixelFlow leaves ``class_name`` as None when no
    labels are given, and mozo does not fill it in -- a name comes from the weights or from the
    user, never from the library."""
    single = dict(click, multimask_output=False)
    assert model.predict(image, **single)[0].class_name is None
    assert model.predict(image, name="kettle", **single)[0].class_name == "kettle"


def test_a_batch_of_prompts_stays_separable(model, image):
    """``class_id`` is the index of the prompt that produced the row, which is the only thing
    keeping two prompts' candidates apart once they are one flat list."""
    height, width = image.shape[:2]
    found = model.predict(
        image,
        points=np.array([[[0.62 * width, 0.55 * height]], [[0.30 * width, 0.40 * height]]]),
        labels=np.array([[1], [1]]),
        multimask_output=False,
    )
    assert [row.class_id for row in found] == [0, 1]


def test_a_name_per_prompt_or_none_at_all(model, image, click):
    """Silently reusing one name across several prompts, or dropping the extras, would label
    detections with something the caller did not say about them."""
    with pytest.raises(ValueError, match="names for"):
        model.predict(image, name=["a", "b"], **click)


def test_a_box_prompt_needs_no_labels(model, image):
    """Its corners carry reserved labels the adapter writes; a caller passing labels for a box
    would be describing something the prompt does not have."""
    height, width = image.shape[:2]
    found = model.predict(
        image,
        boxes=np.array([0.55 * width, 0.42 * height, 0.72 * width, 0.78 * height]),
        multimask_output=False,
    )
    assert len(found) == 1
    assert found[0].class_name is None


# --- what every promptable family looks like over HTTP -------------------------------------------

def test_the_server_refuses_a_promptable_model_with_no_prompt(family):
    """Rejected before the image is decoded and the weights are loaded, so a forgotten prompt
    does not cost a multi-gigabyte load to be told about."""
    variant = get_model_info(family)["variants"][0]
    response = _post(family, variant, "")
    assert response.status_code == 400
    assert "point at something" in response.json()["detail"]


@pytest.mark.parametrize("query,detail", [
    ("?point=1,2", "give one ?label="),
    ("?point=1,2&label=1&point=3,4", "give one ?label="),
    ("?point=1&label=1", "point takes 2"),
    ("?box=1,2,3", "box takes 4"),
    ("?point=1,,2&label=1", "is not numbers"),
    ("?box=1,2,3,4,", "is not numbers"),
])
def test_the_server_rejects_a_malformed_prompt(family, query, detail):
    """Also rejected before any load. The last two are why ``_coordinates`` does not filter empty
    segments before parsing: skipping them would let ``1,,2`` through as a well-formed point."""
    variant = get_model_info(family)["variants"][0]
    response = _post(family, variant, query)
    assert response.status_code == 400
    assert detail in response.json()["detail"]


def _post(family: str, variant: str, query: str):
    """Post a deliberately unreadable body: every assertion here must fire before the decode."""
    from fastapi.testclient import TestClient

    from mozo.server import app

    return TestClient(app).post(
        f"/predict/{family}/{variant}{query}",
        files={"file": ("x.jpg", b"not really an image", "image/jpeg")},
    )
