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
