"""What travels, and the boundary that keeps it one answer.

:mod:`mozo.workflow.wire` was four helpers at the bottom of :mod:`mozo.workflow.api`. Splitting it
is only worth anything if the split holds, so the direction is asserted here rather than left to
reviewers to notice.

Two things are checked, and only one of them lives here. That ``wire`` reaches downward and never
sideways is this file's. That exactly one place turns an array into encoded bytes is repo-wide, so
it is in ``tests/test_image_contract.py`` beside the decode rule it mirrors -- the first version of
that check was a substring search of ``api.py``, and it passed green while ``mozo/server.py`` was
importing a function that no longer existed.

What the values come out as is checked through the endpoint, in ``tests/workflow/test_api.py``,
because that is where a caller sees them. This file is about shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mozo
from conftest import imported
from mozo.workflow import PortType, get
from mozo.workflow.wire import as_json, serialise

PACKAGE = Path(mozo.__file__).parent
WIRE = PACKAGE / "workflow" / "wire.py"

#: What ``wire`` may reach for, and why those two and nothing else.
#:
#: :mod:`mozo.workflow.node` because port types are what it dispatches on, and the two encoders
#: because it owns the JSON envelope and delegates every byte format. A subset rather than an exact
#: set: dropping a dependency is the direction this rule wants, and only gaining one is a breach.
MAY_IMPORT = {"mozo.depth", "mozo.image", "mozo.workflow.node"}


class TestTheBoundary:
    """``wire`` sits below every transport and beside none of them."""

    def test_it_reaches_for_nothing_but_port_types_and_the_two_encoders(self):
        """Resolved with :func:`conftest.imported`, which the vendor and package isolation rules
        already share -- a second copy of "what is an import" is how two rules come to disagree."""
        reached = imported(WIRE, PACKAGE, "mozo")
        inside = {name for name in reached if name.startswith("mozo")}
        assert inside <= MAY_IMPORT, f"wire.py reaches for {sorted(inside - MAY_IMPORT)}"
        assert not any(name.startswith(("fastapi", "starlette")) for name in reached), (
            "wire.py must not know about HTTP -- a second transport has to be able to call it")

    def test_serialising_takes_a_node_not_a_workflow(self):
        """It used to take the whole graph to reach one spec, which made it look like a question
        about running rather than a question about ports."""
        assert serialise(get("save_image"), None) is None


class TestPortTypesItRefuses:
    """A type nothing knows how to send says so, rather than guessing from the array."""

    def test_an_unknown_port_type_is_a_refusal(self):
        class Invented:
            value = "invented"

        with pytest.raises(TypeError, match="invented"):
            as_json(Invented, np.zeros((2, 2)))

    def test_a_batch_keeps_the_port_type_of_the_wire_it_is_on(self):
        """A list is one wire carrying many, not a different kind of value."""
        frames = [np.zeros((2, 2, 3), dtype=np.uint8)] * 3
        sent = as_json(PortType.IMAGE, frames)
        assert len(sent) == 3 and all(item.startswith("data:image/png") for item in sent)

    def test_the_media_type_travels_with_the_bytes(self):
        """So a second encoding is a different value passed in, not a second copy of the builder.
        Nothing asks for a JPEG yet; this is the seam that means nothing will have to fork to."""
        from mozo.workflow.wire import _data_uri

        assert _data_uri(b"x", "image/jpeg").startswith("data:image/jpeg;base64,")
