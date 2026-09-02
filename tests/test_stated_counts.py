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
    r"a curated (\d+)",
    r"Of the (\d+) published variants",
)


def test_the_readme_repeats_that_count_consistently():
    """It appears four more times after the headline. One stale copy contradicts the page."""
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


#: The sha256 of every LICENSE file mozo publishes that is not permissive, and what it is.
#:
#: The manifest records a licence as the hash of the file published beside the weights, never as a
#: name, so which licence a hash *is* has to be written down once. This is that once. Everything
#: downstream is derived: a new YOLO size, a second non-commercial depth variant or another
#: SAM-licensed checkpoint ships the same LICENSE bytes and is recognised without an edit here.
#:
#: Keyed on content rather than on family and variant names deliberately. A hand-written list of
#: names only notices a non-permissive *family* being added; mozo's actual exposure is per
#: variant, and Depth Anything already proves the two differ -- seven of its nine are Apache-2.0
#: and two are not.
NON_PERMISSIVE = {
    "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0": "AGPL-3.0",
    "41003d4a74749c0220e33dd415042164b5a1093ed401f36277234f772d22d3d0": "CC-BY-NC-4.0",
    "4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be": "Meta SAM License",
}


def non_permissive() -> dict[tuple[str, str], str]:
    """Every published variant whose weights are not permissively licensed, from the manifest."""
    found = {}
    for model_id, entry in manifest()["models"].items():
        licence = entry["revisions"][entry["latest"]]["artifacts"].get("LICENSE")
        assert licence, f"{model_id} publishes no LICENSE, so its terms cannot be checked"
        if licence["sha256"] in NON_PERMISSIVE:
            family, variant = model_id.split("/", 1)
            found[(family, variant)] = NON_PERMISSIVE[licence["sha256"]]
    return found


def test_the_readme_s_permissive_deployment_offers_what_it_claims(deploy):
    """The README shows a ``MOZO_ENABLE`` line for serving only the permissively licensed weights,
    and says how many that is. It is the one example whose being wrong is not a documentation bug.

    An operator pastes it to avoid taking on AGPL section 13 obligations, so a variant that slips
    through is served under a licence they were reading this section specifically to decline. And
    it is exactly the kind of list that rots: adding a sixth YOLO size would leave the line still
    looking right while quietly offering it.

    Three independent things are checked, because each can be wrong on its own -- the count the
    prose states, that nothing non-permissive is offered, and that nothing permissive was dropped
    by accident along the way.
    """
    readme = (ROOT / "README.md").read_text()
    example = re.search(r"```bash\nMOZO_ENABLE=(.*?)\n```", readme, re.S)
    assert example, "the README no longer shows a MOZO_ENABLE example -- update this test"
    stated = re.search(r"That is the (\d+) Apache-2\.0 and MIT variants", readme)
    assert stated, "the README no longer states what that line offers -- update this test"

    deployment = deploy(example.group(1).replace("\\\n", ""))
    offered = {(family, v) for family, variants in deployment.items() for v in variants}

    assert len(offered) == int(stated.group(1)), (
        f"the README's MOZO_ENABLE line offers {len(offered)} variants, and says {stated.group(1)}"
    )

    restricted = non_permissive()
    leaked = {model: restricted[model] for model in offered & set(restricted)}
    assert not leaked, f"the README's MOZO_ENABLE line offers non-permissive weights: {leaked}"

    everything = {(family, v) for family, entry in MODEL_REGISTRY.items()
                  for v in entry["variants"]}
    assert everything - offered == set(restricted), (
        f"it also leaves out {sorted(everything - offered - set(restricted))}, which are permissive"
    )


#: What mozo installs. The workflow runtime was absorbed from a separate project that carried
#: flask, flask-cors, requests and tqdm; none of them came, and the whole merge added nothing. That
#: was the constraint the work was done under, so it is stated here rather than left as intent --
#: it is the invariant most easily broken by an ordinary-looking commit.
REQUIRED = {
    "fastapi", "uvicorn", "opencv-python-headless", "torch", "Pillow", "numpy", "pixelflow",
    "click", "python-multipart", "regex", "ftfy", "torchvision",
}


def test_mozo_installs_exactly_what_it_says_it_does():
    # Read rather than parsed with tomllib, which is 3.11+ where mozo supports 3.9 -- and adding
    # tomli to read it would break the very thing this test exists to hold.
    block = re.search(r"^dependencies = \[(.*?)^\]", (ROOT / "pyproject.toml").read_text(),
                      re.S | re.M).group(1)
    named = {re.match(r'"([A-Za-z0-9_.-]+)', line.strip()).group(1)
             for line in block.splitlines()
             if line.strip().startswith('"')}
    assert named == REQUIRED, (
        f"dependencies changed: added {named - REQUIRED}, removed {REQUIRED - named}. "
        f"If that is deliberate, say so here.")
