"""What ``MOZO_ENABLE`` narrows a server to.

The catalogue is what mozo publishes; a deployment is what one server offers. They come apart
because the weights are separate works with their own licences -- the README's licence section has
the split -- and an operator who serves predictions from the non-permissive ones takes on
obligations the rest do not carry. Deployment is automatic, so the choice has to survive a
``pip install``: an environment variable, not a field in a file inside the wheel.

Two things are worth stating about the shape, because both are choices that could have gone the
other way and neither is visible from the code alone.

**It is an allow-list.** A deny-list naming today's AGPL families serves whatever tomorrow's
upgrade adds, silently and without asking. An allow-list serves nothing it was not told to, and
the absence is the visible failure rather than the invisible one.

**An unknown name warns rather than refusing to start.** That is safe only because it is an
allow-list: a token that matches nothing can only subtract, so ``MOZO_ENABLE=siglip`` yields a
server missing SigLIP and never one serving something unsanctioned. A typo cannot produce the
exposure this exists to prevent, which is why it does not justify killing the process.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from mozo.server import app

    with TestClient(app) as running:
        yield running


def refuse(client, payload, family: str, variant: str) -> str:
    """Ask for a prediction that should be refused, and return the reason given."""
    response = client.post(f"/predict/{family}/{variant}",
                           files={"file": ("image.jpg", payload, "image/jpeg")})
    assert response.status_code == 404, f"expected a refusal, got {response.status_code}"
    return response.json()["detail"]


class TestSelection:
    """What a given ``MOZO_ENABLE`` resolves to, before any HTTP is involved."""

    def test_unset_offers_everything_mozo_publishes(self):
        from mozo.registry import MODEL_REGISTRY
        from mozo.server import _deployed

        offered = _deployed()
        assert set(offered) == set(MODEL_REGISTRY)
        for family, entry in MODEL_REGISTRY.items():
            assert list(offered[family]) == entry["variants"]

    def test_an_empty_value_reads_as_unset(self, deploy):
        """Consistent with ``MOZO_CACHE`` and the rest, which all strip and fall through on falsy.

        An exported-but-empty variable is what a shell leaves behind, and reading it as "offer
        nothing" would turn a stray ``export MOZO_ENABLE=`` into a server that answers 404 to
        everything.
        """
        from mozo.registry import MODEL_REGISTRY

        assert set(deploy("  ")) == set(MODEL_REGISTRY)

    def test_a_bare_family_offers_all_of_its_variants(self, deploy):
        from mozo.registry import MODEL_REGISTRY

        offered = deploy("clip")
        assert set(offered) == {"clip"}
        assert list(offered["clip"]) == MODEL_REGISTRY["clip"]["variants"]

    def test_a_qualified_name_offers_that_variant_alone(self, deploy):
        assert deploy("clip/base") == {"clip": ("base",)}

    def test_families_and_variants_mix_freely(self, deploy):
        offered = deploy("clip,siglip2/base-224")
        assert offered["siglip2"] == ("base-224",)
        assert len(offered["clip"]) == 4

    def test_naming_a_family_and_one_of_its_variants_is_a_union(self, deploy):
        """The wider of the two wins, rather than the later one silently overriding the earlier.

        Order-dependence would make ``clip,clip/base`` and ``clip/base,clip`` mean different
        things, which is not something a comma-separated list should express.
        """
        from mozo.registry import MODEL_REGISTRY

        both = deploy("clip,clip/base")
        assert list(both["clip"]) == MODEL_REGISTRY["clip"]["variants"]
        assert deploy("clip/base,clip") == both

    def test_whitespace_and_empty_items_are_ignored(self, deploy):
        assert deploy(" clip , , siglip2/base-224 ,") == deploy("clip,siglip2/base-224")

    def test_the_order_is_the_registry_s_not_the_variable_s(self, deploy):
        """Narrowing removes entries; it does not reshuffle the ones that remain.

        ``/models`` is what the browser page renders in order, and tying that to how an operator
        happened to type an environment variable would make the page's layout a deployment
        detail.
        """
        from mozo.registry import MODEL_REGISTRY

        backwards = deploy("siglip2,rfdetr,clip")
        assert list(backwards) == [f for f in MODEL_REGISTRY if f in backwards]

    def test_a_variant_is_offered_in_the_registry_s_order_too(self, deploy):
        assert deploy("clip/large,clip/base") == {"clip": ("base", "large")}


class TestUnknownNames:
    """A name that matches nothing is dropped, not fatal. See this module's docstring for why."""

    def test_an_unknown_family_is_ignored_and_named_in_the_log(self, deploy, caplog):
        with caplog.at_level(logging.WARNING, logger="mozo.server"):
            offered = deploy("clip,nosuchfamily")
        assert set(offered) == {"clip"}
        assert "nosuchfamily" in caplog.text

    def test_an_unknown_variant_takes_only_itself_out(self, deploy, caplog):
        """The family is not dropped with it: ``clip,clip/nope`` still offers clip whole."""
        with caplog.at_level(logging.WARNING, logger="mozo.server"):
            offered = deploy("clip,clip/nope")
        assert len(offered["clip"]) == 4
        assert "clip/nope" in caplog.text

    def test_a_lone_unknown_name_leaves_nothing_deployed(self, deploy):
        assert deploy("nosuchfamily") == {}

    def test_one_typo_is_one_complaint(self, deploy, caplog):
        """``MOZO_ENABLE=siglip`` is both an unusable name and an empty deployment, and reporting
        those as two warnings is how an operator concludes they have two problems. One line, which
        always ends with what was actually deployed."""
        with caplog.at_level(logging.WARNING, logger="mozo.server"):
            deploy("siglip")
        assert len(caplog.records) == 1
        assert "siglip" in caplog.text and "no models at all" in caplog.text

    def test_the_warning_is_not_repeated_once_per_request(self, monkeypatch, caplog, client):
        """The memo is what keeps it to one line, so this is a test of the caching as much as the
        logging: an operator should not have to scroll past the same typo on every request."""
        from conftest import _forget_deployment

        monkeypatch.setenv("MOZO_ENABLE", "clip,nosuchfamily")
        _forget_deployment()
        caplog.set_level(logging.WARNING, logger="mozo.server")
        for _ in range(3):
            client.get("/models")
        assert caplog.text.count("nosuchfamily") == 1


class TestCatalogue:
    """What ``/models`` says, which is also what the browser page renders."""

    def test_it_lists_only_what_is_deployed(self, deploy, client):
        deploy("clip,siglip2/base-224")
        body = client.get("/models").json()
        assert set(body) == {"clip", "siglip2"}
        assert body["siglip2"]["variants"] == ["base-224"]

    def test_the_rest_of_each_entry_is_untouched(self, deploy, client):
        """Only the variant list narrows. A deployment does not change what a family *is*, so the
        entry is compared against the same family's entry from an unnarrowed catalogue rather than
        against a second copy of the field list."""
        whole = client.get("/models").json()["siglip2"]
        deploy("siglip2/base-224")
        narrowed = client.get("/models").json()["siglip2"]
        assert narrowed["variants"] == ["base-224"] != whole["variants"]
        assert {k: v for k, v in narrowed.items() if k != "variants"} == \
               {k: v for k, v in whole.items() if k != "variants"}


class TestRefusals:
    """Asking for something this deployment does not offer."""

    def test_predict_refuses_an_undeployed_model(self, deploy, client, payload):
        deploy("clip")
        assert "not deployed" in refuse(client, payload, "yolov8", "nano")

    def test_encode_refuses_an_undeployed_model(self, deploy, client):
        deploy("siglip2")
        response = client.post("/encode/clip/base", params={"text": "a cat"})
        assert response.status_code == 404
        assert "not deployed" in response.json()["detail"]

    def test_the_refusal_costs_no_load(self, deploy, monkeypatch, client, payload):
        """This is the assertion that makes the feature worth anything: a model excluded on
        licence grounds must not be fetched or built in the course of refusing it. A 404 raised
        after the download would still be a 404, and would still have put the weights on disk.

        Asserted by making the loader itself fatal rather than by watching what stays resident.
        Residency cannot tell the two apart: an undeployed model whose weights were never
        downloaded fails to load and leaves the same empty manager behind as one that was
        correctly refused, so that check would pass whether the guard fired or not.
        """
        def explode(*args, **kwargs):
            raise AssertionError("a request reached the model loader after being refused")

        deploy("clip")
        monkeypatch.setattr(client.app.state.model_manager, "get_model", explode)
        refuse(client, payload, "yolov8", "nano")
        assert client.post("/encode/siglip2/base-224", params={"text": "a cat"}).status_code == 404

    def test_undeployed_and_nonexistent_read_differently(self, deploy, client, payload):
        """One is fixed by correcting the request, the other by changing the deployment. A single
        message covering both sends an operator hunting for a typo they did not make.
        """
        deploy("clip")
        assert "not deployed" in refuse(client, payload, "yolov8", "nano")
        assert "not a model mozo publishes" in refuse(client, payload, "nosuchfamily", "nano")

    def test_a_refusal_never_names_a_model_this_server_declined(self, deploy, client, payload):
        """The point of narrowing on licence grounds is that the server does not offer those
        models. Answering a typo with the registry's own "Available: [...]" would close the
        catalogue at ``/models`` and reopen it at every wrong turn -- which is what the registry's
        messages do, and why these are composed from the deployment instead.

        Echoing back the name the caller themselves typed is not the leak; enumerating the others is.
        """
        deploy("clip")
        declined = ("yolov11", "sam3", "depth_anything_v2")
        for detail in (refuse(client, payload, "yolov8", "nano"),
                       refuse(client, payload, "nosuchfamily", "nano"),
                       refuse(client, payload, "clip", "nosuchvariant")):
            assert not any(family in detail for family in declined), detail
        # /encode has a refusal of its own, which named every embedding family in the catalogue.
        assert "siglip2" not in client.post("/encode/rfdetr/nano").json()["detail"]

    def test_a_narrowed_family_still_refuses_its_other_variants(self, deploy, client, payload):
        """The variant-level case, which is the one Depth Anything needs: seven of its nine
        variants are Apache-2.0 and two are CC-BY-NC-4.0, so a family-level answer cannot
        express what an operator wants here."""
        deploy("depth_anything_v2/small")
        detail = refuse(client, payload, "depth_anything_v2", "large")
        assert "not deployed" in detail and "Served here: ['small']" in detail

    def test_what_is_deployed_is_let_through(self, deploy):
        """The narrowing must cost nothing to the models that remain: the guard has to stop at the
        boundary it was given, not one variant either side of it.

        Asked of the guard directly rather than through ``/predict``, because running the model
        would be a test of the model. What is in question is which names get past this function.
        """
        from mozo.server import _catalogue_entry

        deploy("depth_anything_v2/small,clip")
        assert _catalogue_entry("depth_anything_v2", "small")["task_type"] == "depth_estimation"
        assert _catalogue_entry("clip", "large-336")["adapter_class"] == "ClipPredictor"
