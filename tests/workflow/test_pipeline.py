"""Running a workflow as a staged pipeline.

The rule these tests hold is that ``workers`` changes only how long a run takes. Same items in,
same results out, in the same order, with failures reaching the caller the same way -- so every
test here compares the pipelined answer against the serial one rather than against a constant.

The join tests are the reason this file exists. A value travelling between stages carries both the
sequence number of its item and the name of the port it fills; a sequence number alone would tell a
two-input node that both its values belong to item 5 without saying which is the left and which is
the right. That bug is invisible whenever branches finish in declaration order, so the graphs here
use ``dawdle`` to make sure they do not.
"""

from __future__ import annotations

import pytest

import workflow_nodes  # noqa: F401  -- importing it is what registers make/brighten/dawdle
from conftest import document
from mozo.workflow import Workflow
from mozo.workflow.node import Image, NodeSpec
from mozo.workflow.pipeline import DEPTH, _width


def spec_for(*, ordered: bool = False, exclusive: bool = False) -> NodeSpec:
    """A real spec, built the way every node's is, so a renamed field breaks this too."""
    def sample(image: Image) -> Image:
        """A node that does nothing, to hang scheduling flags on."""
        return image

    return NodeSpec.from_function(sample, "Test", ordered=ordered, exclusive=exclusive)


def chain() -> Workflow:
    """``make -> brighten -> widen``. One value, one reader, no join."""
    return Workflow.from_dict(document(
        {"a": ("make", {}), "b": ("brighten", {"by": 3}), "c": ("widen", {"times": 2})},
        [("a", "image", "b", "image"), ("b", "image", "c", "image")]))


def diamond(ms: int = 4) -> Workflow:
    """``make`` into two branches that race, joined by ``combine``.

    ``combine`` concatenates left and right, and the two branches make visibly different images --
    ``brighten`` adds three and keeps the width, ``widen`` doubles the width and keeps the value --
    so a swapped join is readable in the answer rather than merely suspected.
    """
    return Workflow.from_dict(document(
        {"a": ("make", {}), "slow": ("dawdle", {"ms": ms}), "l": ("brighten", {"by": 3}),
         "r": ("widen", {"times": 2}), "j": ("combine", {})},
        [("a", "image", "slow", "image"), ("slow", "image", "l", "image"),
         ("a", "image", "r", "image"),
         ("l", "image", "j", "left"), ("r", "image", "j", "right")]))


class TestSameAnswer:
    """Workers change the schedule, not the result."""

    def test_a_chain_gives_what_serial_gives(self):
        workflow, widths = chain(), [2, 3, 4, 5, 6, 7]
        serial = [(item, r["c"].tolist())
                  for item, r in workflow.run_many(widths, over="width", workers=1)]
        piped = [(item, r["c"].tolist())
                 for item, r in workflow.run_many(widths, over="width", workers=4)]
        assert piped == serial

    def test_every_node_is_reported_not_only_the_last(self):
        workflow = chain()
        for _, results in workflow.run_many([2, 3], over="width", workers=4):
            assert sorted(results) == ["a", "b", "c"]

    def test_a_diamond_gives_what_serial_gives(self):
        workflow, widths = diamond(ms=0), [2, 3, 4, 5]
        serial = [r["j"].tolist() for _, r in workflow.run_many(widths, over="width", workers=1)]
        piped = [r["j"].tolist() for _, r in workflow.run_many(widths, over="width", workers=4)]
        assert piped == serial

    def test_a_generator_source_is_consumed_lazily(self):
        workflow = chain()
        seen = []

        def source():
            for width in (2, 3, 4):
                seen.append(width)
                yield width

        out = list(workflow.run_many(source(), over="width", workers=3))
        assert [item for item, _ in out] == [2, 3, 4]
        assert seen == [2, 3, 4]


class TestAdmission:
    """How many items may be alive at once, which bounded queues alone do not settle."""

    def test_a_straggler_does_not_let_the_rest_pile_up(self):
        """The failure this guards: backpressure only reaches back from a queue that is full, and
        the last stage never fills one -- it hands its answers to the caller. So one slow item let
        every item behind it run to completion and wait, finished, for its turn, and peak memory
        tracked the length of the source rather than the depth of the queues.
        """
        started = workflow_nodes.STARTED
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("hesitate", {"slow": 3}), "c": ("brighten", {})},
            [("a", "image", "b", "image"), ("b", "image", "c", "image")]))

        cap = len(workflow.order) * DEPTH * 2
        for count in (40, 160):
            started.clear()
            handed = alive = 0
            for _ in workflow.run_many(list(range(2, 2 + count)), over="width", workers=2):
                handed += 1
                alive = max(alive, len(started) - handed)
            assert alive <= cap, f"{count} items: {alive} alive, cap is {cap}"


class TestOrder:
    """``run_many`` promises arrival order, and wide stages do not finish in it."""

    def test_results_arrive_in_the_order_the_items_did(self):
        workflow = diamond(ms=4)
        widths = list(range(2, 60))
        got = [item for item, _ in workflow.run_many(widths, over="width", workers=4)]
        assert got == widths


class TestJoin:
    """A two-input node is handed the right value on the right port, for the right item."""

    def test_a_racing_join_pairs_the_right_values_on_the_right_ports(self):
        """left is brighten -- same width, value + 3. right is widen -- twice the width, value.

        So a swapped port or a mispaired item is readable in the answer rather than suspected.
        """
        workflow = diamond(ms=4)
        for width, results in workflow.run_many(list(range(2, 80)), over="width", workers=4):
            joined = results["j"]
            assert joined.shape[1] == width * 3, f"width {width} joined the wrong shapes"
            assert set(joined[0, :width, 0].tolist()) == {width + 3}, f"width {width} swapped"
            assert set(joined[0, width:, 0].tolist()) == {width}


class TestFailure:
    """One bad item does not end the run, and reaches the caller the way it always did."""

    def bad(self) -> Workflow:
        return Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("explode", {})}, [("a", "image", "b", "image")]))

    def test_on_error_is_called_per_item_as_serial_does(self):
        workflow = self.bad()
        for workers in (1, 4):
            seen = []
            out = list(workflow.run_many([2, 3, 4], over="width", workers=workers,
                                         on_error=lambda item, event: seen.append(item)))
            assert out == []
            assert sorted(seen) == [2, 3, 4]

    def test_without_on_error_it_raises_as_serial_does(self):
        workflow = self.bad()
        for workers in (1, 4):
            with pytest.raises(RuntimeError, match="as promised"):
                list(workflow.run_many([2, 3], over="width", workers=workers))

    def test_a_good_item_still_arrives_when_a_neighbour_fails(self):
        """``explode`` fires on every item, so failure is mixed in by the source instead."""
        workflow = chain()
        seen = []
        out = list(workflow.run_many([2, None, 4], over="width", workers=4,
                                     on_error=lambda item, event: seen.append(item)))
        assert [item for item, _ in out] == [2, 4]
        assert seen == [None]


class TestWidth:
    """How many items may be inside a node, and who decides."""

    def test_an_ordinary_node_takes_the_worker_count(self):
        assert _width(spec_for(), 4, model_workers=8) == 4

    def test_an_exclusive_node_runs_one_at_a_time_by_default(self):
        assert _width(spec_for(exclusive=True), 8) == 1

    def test_an_exclusive_node_takes_the_width_it_is_given(self):
        assert _width(spec_for(exclusive=True), 8, model_workers=3) == 3
        assert _width(spec_for(exclusive=True), 1, model_workers=4) == 4

    def test_an_ordered_node_is_pinned_to_one_whatever_is_asked(self):
        assert _width(spec_for(ordered=True), 8) == 1
        assert _width(spec_for(ordered=True), 8, model_workers=4) == 1

    def test_ordered_implies_exclusive(self):
        assert spec_for(ordered=True).exclusive, "ordering a node without excluding it means little"

    def test_every_shipped_model_node_says_it_is_exclusive(self):
        """The guard that matters: a family added without the flag would silently widen.

        Checked against the nodes, not against a list of categories or a module path -- a node
        declared anywhere else is exactly the case those would miss.
        """
        from mozo.workflow import registry

        models = [name for name in registry.names()
                  if registry.get(name).run.__module__.endswith(".model")]
        assert models, "no model nodes found; this guard would pass vacuously"
        for name in models:
            assert registry.get(name).exclusive, f"{name} would run several inferences at once"


class TestSaturation:
    """What the run reports about itself, so a caller can see where the limit was."""

    def test_stats_name_every_node_with_its_width_and_queue(self):
        workflow, seen = chain(), {}
        list(workflow.run_many([2, 3, 4, 5], over="width", workers=2, stats=seen))
        assert set(seen) == {"elapsed_s", "stages"}
        assert set(seen["stages"]) == {"a", "b", "c"}
        for node in ("a", "b", "c"):
            assert seen["stages"][node]["width"] == 2
            assert 0.0 <= seen["stages"][node]["saturation"] <= 1.0
            assert seen["stages"][node]["queue_peak"] <= seen["stages"][node]["queue_size"]

    def test_stats_are_absent_unless_asked_for(self):
        workflow = chain()
        list(workflow.run_many([2, 3], over="width", workers=2))     # no stats= and no crash

    def test_asking_a_serial_run_for_stats_is_refused_rather_than_ignored(self):
        workflow = chain()
        with pytest.raises(ValueError, match="workers > 1"):
            list(workflow.run_many([2, 3], over="width", workers=1, stats={}))

    def test_a_busy_node_reports_more_saturation_than_an_idle_one(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "slow": ("dawdle", {"ms": 8}), "b": ("brighten", {})},
            [("a", "image", "slow", "image"), ("slow", "image", "b", "image")]))
        seen: dict = {}
        list(workflow.run_many(list(range(2, 22)), over="width", workers=2, stats=seen))
        assert seen["stages"]["slow"]["saturation"] > seen["stages"]["b"]["saturation"]
