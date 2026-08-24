"""The rule that makes everything else about this subpackage cheap.

``mozo.workflow`` may import ``mozo``. ``mozo`` may never import ``mozo.workflow`` -- with one
exception, the router ``mozo.server`` mounts.

One directed edge, at the outermost layer. It is what lets this directory be deleted, rewritten or
extracted without touching a vendor, and it is why the model zoo can never be broken by the
workflow runtime. Stated in prose it would rot; here it is checked.

This is the same rule the vendors already live under -- see
``tests/test_vendor_agreement.py::test_a_vendor_imports_nothing_of_mozo_s_but_itself`` -- applied
one level up, and checked the same way.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import mozo
from conftest import imported

PACKAGE = Path(mozo.__file__).parent

#: The modules allowed to name ``mozo.workflow``, and why. Both are entry points a person invokes
#: rather than library code something imports: the server mounts the router, and ``mozo run``
#: executes a workflow. Kept as a mapping so that adding an entry is a deliberate, reviewable act
#: with a reason attached, rather than a regex getting looser.
#:
#: ``cli.py`` imports it inside the command, so ``mozo version`` still costs nothing -- which
#: :class:`TestImportCost` holds rather than leaving to this comment.
MAY_IMPORT_THE_WORKFLOW_RUNTIME = {
    "server.py": "mounts the workflow router",
    "cli.py": "runs a workflow, inside the `run` command",
}


def _reaches(source: Path) -> set:
    """Every module *source* imports. Shared with the vendor rule -- see ``conftest.imported``."""
    return imported(source, PACKAGE, "mozo")


def _mozo_outside_the_runtime() -> list:
    """Every source file in mozo that is not part of the workflow runtime."""
    return [path for path in sorted(PACKAGE.rglob("*.py"))
            if "workflow" not in path.relative_to(PACKAGE).parts]


class TestTheImportRule:
    """Nothing in mozo reaches into the workflow runtime, except the one place that mounts it."""

    def test_mozo_does_not_import_the_workflow_runtime(self):
        offenders = []
        for source in _mozo_outside_the_runtime():
            if source.name in MAY_IMPORT_THE_WORKFLOW_RUNTIME:
                continue
            for name in _reaches(source):
                if name == "mozo.workflow" or name.startswith("mozo.workflow."):
                    offenders.append(f"{source.relative_to(PACKAGE)} -> {name}")

        assert not offenders, (
            "mozo must not depend on its workflow runtime:\n  " + "\n  ".join(offenders))

    def test_enough_of_mozo_was_scanned_for_that_to_mean_anything(self):
        """A discovery bug would make the test above pass over nothing and read green."""
        scanned = _mozo_outside_the_runtime()
        assert len(scanned) >= 20, f"expected most of mozo, scanned {len(scanned)} files"
        assert any(path.name == "server.py" for path in scanned)
        assert not any("workflow" in path.parts for path in scanned)

    def test_the_scan_can_see_an_import_when_there_is_one(self):
        """And a walker that returned nothing would pass the test above just as quietly.

        ``mozo/server.py`` reaches these three by relative import, so this also holds that
        ``from .registry import ...`` is resolved rather than skipped -- which is the only way
        ``from .workflow.api import router`` would ever be seen.
        """
        assert {"mozo.image", "mozo.manager", "mozo.registry"} <= _reaches(PACKAGE / "server.py")

    def test_there_is_a_workflow_runtime_to_exclude(self):
        """Excluding a directory that is not there would make the scan trivially clean."""
        assert len(sorted((PACKAGE / "workflow").rglob("*.py"))) >= 3


#: Run in a subprocess so the block cannot leak into the rest of the session. It imports mozo with
#: ``mozo.workflow`` made unimportable, which is what deleting the directory would do.
#:
#: ``mozo.server`` is deliberately *not* checked here, and that is a correction to what this file
#: first claimed. The server mounts the workflow router, so it genuinely depends on the runtime and
#: cannot import without it. The alternative -- wrapping the mount in ``try: ... except
#: ImportError`` -- would turn a typo in ``api.py`` into a workflow API that silently is not there,
#: which is a worse failure than an honest one.
#:
#: What must survive is the model zoo itself: ``get_model``, the registry, the weight resolver and
#: every adapter. That is the thing the workflow runtime must never be able to break, and it is
#: what this checks.
WITHOUT_THE_RUNTIME = """
import sys

class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name == "mozo.workflow" or name.startswith("mozo.workflow."):
            raise ImportError(f"{name} is deleted, for the purposes of this test")
        return None

sys.meta_path.insert(0, Blocked())

try:
    import mozo.workflow
except ImportError:
    pass
else:
    raise AssertionError("the block did not block -- this test proves nothing")

import importlib

import mozo
import mozo.manager
import mozo.registry
import mozo.weights
import mozo.image
import mozo.labels
import mozo.runtimes

assert mozo.get_model is not None
assert len(mozo.MODEL_REGISTRY) >= 14

# Every adapter, because an adapter is what turns a checkpoint into an answer, and one of them
# reaching for the workflow runtime would be the dependency this rule forbids.
for family, entry in mozo.MODEL_REGISTRY.items():
    importlib.import_module(entry["module"])

print("OK")
"""


class TestTheRuntimeReachesNothing:
    """What running a graph must never be a reason to do."""

    def test_the_workflow_runtime_imports_nothing_that_opens_a_socket(self):
        """Fetching weights is ``weights.py``'s job. Running a graph is not a reason to open one.

        Resolved by imports rather than by looking for ``urlopen`` in the text, for the reason
        ``conftest.imported`` was written: a substring search stays green against ``http.client``,
        ``aiohttp``, ``urllib3`` and anything else nobody thought to list.
        """
        networking = {"urllib", "urllib3", "http", "socket", "ssl", "ftplib",
                      "requests", "httpx", "aiohttp", "asyncio"}
        reaching = []
        for source in sorted((PACKAGE / "workflow").rglob("*.py")):
            for name in _reaches(source):
                if name.split(".")[0] in networking:
                    reaching.append(f"{source.relative_to(PACKAGE)} -> {name}")
        assert not reaching, f"the workflow runtime must not reach out: {reaching}"


#: What ``import mozo`` must not drag in. The workflow runtime needs ``pixelflow.Detections`` at
#: decoration time, so it pays for PixelFlow -- 179 ms against mozo's 2.5 ms. (It was 528 ms until
#: PixelFlow deferred ``pixelflow.tracker``, which was pulling SciPy for ByteTrack on every import.)
#: Most callers of ``get_model`` never build a workflow, and none of them should pay for one.
STAYS_CHEAP = """
import sys
import mozo
import mozo.cli

heavy = sorted(name for name in ("mozo.workflow", "pixelflow", "scipy", "torch")
               if name in sys.modules)
assert not heavy, f"importing mozo and its CLI pulled in {heavy}"

print("OK")
"""


class TestImportCost:
    """``import mozo`` stays cheap, whatever the workflow runtime costs."""

    def test_importing_mozo_does_not_import_the_workflow_runtime(self):
        """The rule in this file is structural; this is the same rule, priced.

        ``mozo.server`` does pay it -- it mounts the workflow router, so it needs the runtime, and
        four hundred modules once at server start is nothing. ``mozo`` itself must not: a caller
        who wants ``get_model`` should never load PixelFlow, and ``mozo/__init__.py`` reaching
        ``server`` or ``workflow`` would make that unavoidable for everyone.

        ``mozo.cli`` is checked too, because it is what ``mozo version`` loads. It reaches the
        runtime only inside ``mozo run``, and this is what says so.
        """
        assert _run(STAYS_CHEAP) == "OK"


class TestDeletingTheRuntime:
    """The model zoo works with the workflow runtime gone."""

    def test_the_model_zoo_still_works_without_it(self):
        pytest.importorskip("fastapi")
        assert _run(WITHOUT_THE_RUNTIME) == "OK"


def _run(script: str) -> str:
    """Run *script* in a fresh interpreter and return its last line.

    A subprocess because both scripts are about what a *clean* import does, and this session has
    already imported everything they are checking for.
    """
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=PACKAGE.parent)
    assert finished.returncode == 0, finished.stderr
    return finished.stdout.strip().splitlines()[-1]
