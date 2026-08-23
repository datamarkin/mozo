#!/usr/bin/env python3
"""Check a built wheel and sdist before they are published.

Three things go wrong between ``python -m build`` and PyPI, and none of them is loud. The
version in the filename is not the version that was tagged. The readme does not render. A data
file the code reads at import time is simply absent, because the package-data rule in
``pyproject.toml`` never matched it. The first is caught by the tag guard in the publish
workflow, the second by ``twine check --strict``. This catches the third, and it catches it by
rule rather than by list.

The rule: every file under ``mozo/`` that is not source, not documentation and not junk is a
file the package needs at run time, so it must appear in *both* distributions. ``PROVENANCE.md``
is documentation by extension but a compliance file by purpose, so it is required alongside the
licence and the notice it belongs to. Vendor ``README.md`` files are the only Markdown left
behind, because nothing reads them and no licence turns on them.

Deriving that expectation from the tree instead of writing it down is the whole point. A
hand-maintained list of what to ship is what let five of fourteen families ship their code with
none of their terms; a rule read off the working tree cannot go stale the same way.

Run from the repository root::

    python tools/check_dist.py dist --expect-version 0.5.0

Exits non-zero, listing every missing path, if either distribution is short.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "mozo"

#: Documentation by extension, compliance by purpose. Required despite the ``.md`` rule below.
REQUIRED_MARKDOWN = {"PROVENANCE.md"}

#: Source and documentation are not run-time data. Everything else under ``mozo/`` is.
NOT_DATA_SUFFIXES = {".py", ".md"}

#: Never shipped, never checked for.
JUNK_NAMES = {".DS_Store"}
JUNK_DIRS = {"__pycache__"}


def required_paths() -> list[str]:
    """Every ``mozo/``-relative path the wheel and sdist must both carry.

    Read off the working tree, so a family added tomorrow is covered today.
    """
    found = []
    for path in sorted(PACKAGE_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.name in JUNK_NAMES or JUNK_DIRS & set(path.relative_to(ROOT).parts):
            continue
        if path.suffix in NOT_DATA_SUFFIXES and path.name not in REQUIRED_MARKDOWN:
            continue
        found.append(str(path.relative_to(ROOT)))
    return found


def wheel_contents(wheel: Path) -> set[str]:
    """Paths inside *wheel*, which are already package-relative (``mozo/...``)."""
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def sdist_contents(sdist: Path) -> set[str]:
    """Paths inside *sdist*, with the ``mozo-<version>/`` prefix every member carries removed."""
    with tarfile.open(sdist) as archive:
        return {
            name.split("/", 1)[1]
            for name in archive.getnames()
            if "/" in name
        }


def sole(paths: list[Path], what: str) -> Path:
    """The one *what* in ``dist/``, or an exit explaining what was found instead.

    Two wheels means a stale build is sitting next to the fresh one, and half of publishing it
    would be an accident rather than a decision.
    """
    if len(paths) != 1:
        names = ", ".join(sorted(p.name for p in paths)) or "nothing"
        sys.exit(f"expected exactly one {what}, found {len(paths)}: {names}")
    return paths[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", type=Path, help="the directory python -m build wrote to")
    parser.add_argument(
        "--expect-version",
        required=True,
        help="the version both distributions must be named for, without a leading v",
    )
    args = parser.parse_args()

    wheel = sole(sorted(args.dist.glob("*.whl")), "wheel")
    sdist = sole(sorted(args.dist.glob("*.tar.gz")), "sdist")

    version = args.expect_version
    problems = []
    if wheel.name != f"mozo-{version}-py3-none-any.whl":
        problems.append(f"wheel is named {wheel.name}, not mozo-{version}-py3-none-any.whl")
    if sdist.name != f"mozo-{version}.tar.gz":
        problems.append(f"sdist is named {sdist.name}, not mozo-{version}.tar.gz")

    required = required_paths()
    for label, contents in (("wheel", wheel_contents(wheel)), ("sdist", sdist_contents(sdist))):
        missing = [path for path in required if path not in contents]
        if missing:
            problems.append(
                f"{len(missing)} run-time file(s) missing from the {label}:\n    "
                + "\n    ".join(missing)
            )

    if problems:
        print("\n".join(problems), file=sys.stderr)
        if any("missing" in problem for problem in problems):
            print(
                "\nA missing data file usually means the package-data rule in pyproject.toml "
                "does not match it.",
                file=sys.stderr,
            )
        return 1

    print(f"mozo {version}: {len(required)} run-time files present in both the wheel and the sdist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
