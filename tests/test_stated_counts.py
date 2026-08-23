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

#: The irregular part of writing a number out. Everything from twenty up is composed from these.
_UNITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
          "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def spell(number: int) -> str:
    """Write *number* out in words, the way prose does.

    Derived rather than tabulated because the alternative -- a dict of the counts mozo happens to
    have today -- is the same hand-maintained number this file exists to stop drifting, one level
    up. It covers 0-99, which outlasts any catalogue anyone will read in one sitting.
    """
    if not 0 <= number < 100:
        raise ValueError(f"{number} is past where a docstring should be spelling numbers out")
    if number < 20:
        return _UNITS[number]
    tens, unit = divmod(number, 10)
    return _TENS[tens] + (f"-{_UNITS[unit]}" if unit else "")


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

    assert f"{spell(variants).capitalize()} published variants" in text, (
        f"the package docstring does not open with {spell(variants)} published variants:"
        f"\n{text[:200]}"
    )
    assert f"{spell(families)} families" in text, (
        f"the package docstring does not say {spell(families)} families"
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


def test_every_registered_variant_is_one_the_manifest_publishes():
    """The other direction, and the one that reaches a user.

    A variant the manifest does not carry is a name the catalogue offers and no request can
    satisfy: ``/models`` lists it, the test page puts it in the dropdown, and picking it fails on
    a download that was never going to find anything. SigLIP 2 shipped that way -- fifteen
    registered against three published -- and the symptom was a 500 rather than a red test.

    An empty ``variants`` list means the family accepts any name and publishes nothing of its own,
    which ``registry.py`` documents; those are skipped rather than failed.
    """
    models = manifest()["models"]
    for family, entry in MODEL_REGISTRY.items():
        missing = [v for v in entry["variants"] if f"{family}/{v}" not in models]
        assert not missing, f"{family} registers {missing} but the manifest publishes no weights"


def test_each_family_counts_its_own_variants_correctly():
    """Eight descriptions open by counting their variants, and ``/models`` serves that text.

    The count is written by hand next to the list it counts, so it is the one number in the
    catalogue that can contradict the entry carrying it.
    """
    for family, entry in MODEL_REGISTRY.items():
        stated = re.search(r"(\d+) variants?\b", entry["description"])
        if not stated or not entry["variants"]:
            continue
        assert int(stated.group(1)) == len(entry["variants"]), (
            f"{family} says {stated.group(1)} variants and lists {len(entry['variants'])}"
        )


def test_the_licence_breakdown_accounts_for_every_variant():
    """The README splits the published variants by licence. The parts must make the whole.

    Not derived from the manifest, which records a LICENSE artifact's hash but not its name, so
    the split stays hand-written. Summing it is what catches a stale one: the numbers were 31 and
    then 34 and then 36 as families landed, and each time only one of the five moved.
    """
    readme = (ROOT / "README.md").read_text()
    sentence = re.search(r"Of the \d+ published variants.*?SAM License", readme, re.S)
    assert sentence, "the README no longer states a licence breakdown -- update this test"
    parts = [int(n) for n in re.findall(r"\**(\d+) (?:are|carries)\b", sentence.group(0))]
    assert sum(parts) == published_variants(), (
        f"the licence breakdown sums to {sum(parts)}, manifest publishes {published_variants()}"
    )


def test_the_readme_states_how_many_families_ship_a_verification_gate():
    """`tools/verify/` is the claim the whole README rests on, so its arithmetic is checked."""
    gates = {
        path.stem for path in (ROOT / "tools" / "verify").glob("*.py")
        if not path.stem.startswith("_") and not path.stem.endswith("_reference")
    }
    families = published_families()
    readme = (ROOT / "README.md").read_text()

    assert gates <= set(MODEL_REGISTRY), f"gates for unknown families: {gates - set(MODEL_REGISTRY)}"
    assert f"{spell(len(gates)).capitalize()} of the {spell(families)} families ship one" in readme, (
        f"{len(gates)} of {families} families have a gate; the README says otherwise"
    )
