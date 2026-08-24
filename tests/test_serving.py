"""Rules about how mozo serves, which belong to mozo rather than to any subpackage.

Both scan ``mozo/`` as a whole. They lived under ``tests/workflow/`` when the HTTP layer was
written, which put the guarantees about the model zoo inside the directory ``I2`` promises the zoo
can live without -- delete ``mozo/workflow/`` and its tests together and the rule that
``mozo/server.py`` has no blocking handler and names no company endpoint would go with them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mozo

PACKAGE = Path(mozo.__file__).parent

#: The modules that answer HTTP requests. One list, read by the rule below and by the guard that
#: proves the rule looked at something -- a guard reading off its own copy could stay green while
#: the rule it guards had quietly stopped scanning half of them.
SERVING = (PACKAGE / "server.py", PACKAGE / "workflow" / "api.py")


def _decorated(kind) -> list:
    """Every decorated definition of *kind* in the modules that serve HTTP."""
    return [f"{source.name}:{node.lineno} {node.name}" for source in SERVING
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, kind) and node.decorator_list]


class TestNothingBlocksTheEventLoop:

    def test_no_handler_that_runs_a_model_or_a_graph_is_async(self):
        """A handler defined ``async def`` runs on the event loop.

        Inference and a workflow are both seconds of blocking work, so a second request would wait
        behind the first with nothing to say why. ``def`` hands it to the threadpool instead. The
        implementation this replaces got exactly this wrong.
        """
        offenders = _decorated(ast.AsyncFunctionDef)
        assert not offenders, "these would block the event loop:\n  " + "\n  ".join(offenders)

    def test_the_scan_looked_at_handlers_that_exist(self):
        """A parse that found nothing would make the rule above vacuous."""
        assert len(_decorated(ast.FunctionDef)) >= 8


class TestNothingPhonesHome:

    def test_nothing_in_mozo_names_a_datamarkin_host(self):
        """The one host mozo may never call.

        mozo does reach the network in exactly one place -- ``weights.py`` fetches a checkpoint on
        first use, from the base URL the manifest names -- and that is the whole of it. What must
        not appear is a company endpoint: no account, no telemetry, nothing that turns running a
        model locally into a request someone else can see.
        """
        offenders = [source.relative_to(PACKAGE) for source in sorted(PACKAGE.rglob("*.py"))
                     if "datamarkin.com" in source.read_text()]
        assert not offenders, f"these name a datamarkin host: {offenders}"

    def test_the_scan_read_the_package(self):
        """Scanning an empty tree would pass just as quietly."""
        assert len(list(PACKAGE.rglob("*.py"))) >= 50
