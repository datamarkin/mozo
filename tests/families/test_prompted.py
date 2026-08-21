"""The contract every text-prompted family shares, checked against every one of them.

Two families ask the model a question in words: SAM 3 answers with masks, OWLv2 with boxes. What
they have in common is the *request* -- a phrase, or several -- and therefore the way a missing or
blank one has to be refused. ``mozo.registry`` names that set once, as ``PROMPTED``, and this file
discovers the families from it rather than listing them, the move
``tests/families/test_promptable.py`` makes one shelf over and for the same stated reason: a
hand-maintained list is how a third family lands with a green suite and no coverage.

Only the server contract is here. What comes back differs so much between the two that there is
nothing else honest to assert about both, and each family's own file holds the rest.

Nothing here loads weights. Every assertion below must fire before the image is decoded and before
the model is resolved, which is the property being tested as much as the status code is: a
forgotten prompt should not cost a multi-gigabyte download to be told about.
"""

from __future__ import annotations

import pytest

from mozo.registry import MODEL_REGISTRY, PROMPTED, get_model_info

#: Every family the registry says is asked a question in words. Discovered, not listed.
FAMILIES = sorted(
    family for family, entry in MODEL_REGISTRY.items() if entry["task_type"] in PROMPTED
)


def test_more_than_one_family_was_discovered():
    """Discovery that silently found nothing would make every test below vacuously pass."""
    assert len(FAMILIES) >= 2, f"expected several prompted families, found {FAMILIES}"


def test_every_prompted_task_has_a_family():
    """``PROMPTED`` names task types; the registry assigns them. A name in the set that no family
    claims is a branch of the endpoint nothing can reach."""
    assert PROMPTED <= {entry["task_type"] for entry in MODEL_REGISTRY.values()}


@pytest.fixture(params=FAMILIES)
def family(request):
    """One prompted family's name."""
    return request.param


def test_the_server_refuses_a_prompted_model_with_no_prompt(family):
    response = _post(family, "")
    assert response.status_code == 400
    assert "is prompted" in response.json()["detail"]


@pytest.mark.parametrize("query", ["?text=", "?text=%20", "?text=car&text=", "?text=%20&text=car"])
def test_the_server_refuses_a_blank_prompt(family, query):
    """Blank is not a narrower question, it is no question -- and both models will happily encode
    the empty string and return whatever they find most salient."""
    response = _post(family, query)
    assert response.status_code == 400
    assert "is prompted" in response.json()["detail"]


def test_a_prompt_may_contain_a_comma(family):
    """``?text=`` is repeated rather than comma-separated, unlike ``?labels=``, because a phrase
    is free text: ``"a person, holding a mug"`` is one concept and splitting it would be two.
    Checked by getting past the prompt guard and failing on the *image* instead."""
    response = _post(family, "?text=a%20person%2C%20holding%20a%20mug")
    assert response.status_code == 400
    assert "decode" in response.json()["detail"]


def _post(family: str, query: str):
    """Post a deliberately unreadable body: every assertion here must fire before the decode."""
    from fastapi.testclient import TestClient

    from mozo.server import app

    variant = get_model_info(family)["variants"][0]
    return TestClient(app).post(
        f"/predict/{family}/{variant}{query}",
        files={"file": ("x.jpg", b"not really an image", "image/jpeg")},
    )
