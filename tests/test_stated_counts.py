"""Every number mozo states about itself, held against the manifest that decides it.

The package docstring said "Seventeen published variants across two families" for eight families
longer than it was true, and the README said 52 while the manifest said 54. Both are the first
thing a reader sees -- ``help(mozo)`` opens with one and PyPI opens with the other -- and both
were wrong because nothing checked them.

Prose cannot be generated: a module docstring has to be a literal for ``help()`` and doctest, and
a README is written for people. So the numbers stay written by hand and this file is what stops
them drifting, exactly as ``tests/families/test_*.py::test_registry_agrees_with_the_adapter``
stops the catalogue drifting from the adapters.

The manifest is the authority because it is generated from the weights tree
(``tools/generate_manifest.py``) rather than typed, so it cannot itself be the thing that is
stale.
"""

from __future__ import annotations

import re
from pathlib import Path

import mozo
from mozo.registry import MODEL_REGISTRY
from mozo.weights import manifest

ROOT = Path(__file__).resolve().parent.parent

#: Numbers written out in words in the package docstring. Only as far as mozo could plausibly
#: grow before someone rewrites the sentence anyway.
WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}


def published_variants() -> int:
    """How many variants the manifest publishes."""
    return len(manifest()["models"])


def published_families() -> int:
    """How many distinct families those variants belong to."""
    return len({model_id.split("/")[0] for model_id in manifest()["models"]})


def test_the_package_docstring_states_the_published_counts():
    """``help(mozo)`` opens with these two numbers, so they are the ones read most and checked
    least."""
    text = mozo.__doc__
    variants, families = published_variants(), published_families()

    assert WORDS.get(families), f"{families} families: add it to WORDS and rewrite the sentence"
    assert f"{variants} published variants" in text.replace(
        "Fifty-four", "54"
    ), f"the package docstring does not say {variants} published variants:\n{text[:200]}"
    assert f"{WORDS[families]} families" in text, (
        f"the package docstring does not say {WORDS[families]} families"
    )


def test_the_readme_headline_states_the_published_count():
    """The first line of the README is the first thing anyone reads about mozo."""
    readme = (ROOT / "README.md").read_text()
    variants = published_variants()

    headline = re.search(r"^### (\d+) computer vision models", readme, re.M)
    assert headline, "the README no longer opens with a model count"
    assert int(headline.group(1)) == variants, (
        f"README headline says {headline.group(1)}, manifest publishes {variants}"
    )


#: Where the README states the model count, as the words either side of the number. Tight on
#: purpose: matching a bare "the NN" would fire on "the 16 tokens" in the OWLv2 row and fail the
#: suite for an unrelated edit, which is worse than not checking -- a test nobody trusts gets
#: deleted. Reword the README and this list is what you update.
COUNT_CONTEXTS = (
    r"### (\d+) computer vision models",
    r"runs all (\d+)",
    r"# all (\d+), no torch import",
    r"Pick any of the (\d+)",
    r"a curated (\d+)",
    r"Of the (\d+) published variants",
)


def test_the_readme_repeats_that_count_consistently():
    """It appears five more times after the headline. One stale copy contradicts the page."""
    readme = (ROOT / "README.md").read_text()
    variants = published_variants()

    for pattern in COUNT_CONTEXTS:
        found = re.findall(pattern, readme)
        assert found, f"the README no longer contains {pattern!r} -- update COUNT_CONTEXTS"
        for number in found:
            assert int(number) == variants, (
                f"README says {number} in {pattern!r}, manifest publishes {variants}"
            )


def test_the_registry_and_the_manifest_agree_on_the_families():
    """A family in the catalogue that publishes nothing is fine -- it takes your own checkpoint.
    A family in the manifest that the catalogue does not know is not reachable at all."""
    catalogued = set(MODEL_REGISTRY)
    published = {model_id.split("/")[0] for model_id in manifest()["models"]}
    assert published <= catalogued, f"published but not in the registry: {published - catalogued}"


def test_every_published_variant_is_one_the_registry_names():
    """The manifest is generated from directory names; a typo there publishes bytes under a
    variant no adapter will ever ask for."""
    for model_id in manifest()["models"]:
        family, variant = model_id.split("/", 1)
        known = MODEL_REGISTRY[family]["variants"]
        assert not known or variant in known, f"{model_id} is published but {family} lists {known}"


def test_the_readme_states_how_many_families_ship_a_verification_gate():
    """`tools/verify/` is the claim the whole README rests on, so its arithmetic is checked."""
    gates = {
        path.stem for path in (ROOT / "tools" / "verify").glob("*.py")
        if not path.stem.startswith("_") and not path.stem.endswith("_reference")
    }
    families = published_families()
    readme = (ROOT / "README.md").read_text()

    assert gates <= set(MODEL_REGISTRY), f"gates for unknown families: {gates - set(MODEL_REGISTRY)}"
    assert WORDS.get(len(gates)) and WORDS.get(families)
    assert f"{WORDS[len(gates)].capitalize()} of the {WORDS[families]} families ship one" in readme, (
        f"{len(gates)} of {families} families have a gate; the README says otherwise"
    )
