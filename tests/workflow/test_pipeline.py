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

import itertools
import threading
import time

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


class TestASourceIsTheWholeGraph:
    """A workflow whose only node is its source: read a file, do nothing else.

    Not a contrived shape. It is what the editor shows the moment an input node is dropped on an
    empty canvas, and it is what a person runs to check that their file opens before wiring
    anything to it.
    """

    def test_it_finishes_rather_than_waiting_for_a_stage_that_does_not_exist(self):
        """It used to hang. Every other item is complete once the stages after the source report,
        and this one has none, so the run waited on a count that could never arrive -- forever,
        holding the file open, with no error and nothing to look at."""
        made = Workflow.from_dict(document({"a": ("emit", {"count": 4})}))
        items = list(made.process())
        assert len(items) == 4
        assert [int(item[0, 0, 0]) for item, _ in items] == [0, 1, 2, 3]

    def test_the_source_is_still_reported_as_a_result(self):
        """The node is the whole graph, so dropping its output would leave nothing at all."""
        made = Workflow.from_dict(document({"a": ("emit", {"count": 2})}))
        for item, results in made.process():
            assert list(results) == ["a"]
            assert results["a"] is item

    def test_serial_and_pipelined_agree(self):
        made = Workflow.from_dict(document({"a": ("emit", {"count": 3})}))
        serial = [item.tolist() for item, _ in made.process(workers=1)]
        assert [item.tolist() for item, _ in made.process(workers=4)] == serial


class TestTheDefault:
    """``run_many`` pipelines unless told otherwise, and what that costs."""

    def test_it_pipelines_without_being_asked(self):
        seen: dict = {}
        workflow = chain()
        list(workflow.run_many([2, 3, 4], over="width", stats=seen))
        assert seen["stages"]["a"]["width"] == 2, "the default should be two workers"

    def test_the_source_is_read_ahead_but_only_by_a_bounded_amount(self):
        """Not zero any more -- bounded, which is what keeps a million items affordable.

        Serial read nothing ahead; a pipeline must read ahead or its stages would have nothing to
        overlap. What matters is that the amount cannot grow with the source, so an endless one
        costs what is in flight and nothing that accumulates.
        """
        reached = []

        def counted():
            for width in range(2, 500):
                reached.append(width)
                yield width

        workflow, workers = chain(), 2
        answers = workflow.run_many(counted(), over="width", workers=workers)
        next(answers)
        # Every stage can hold a full queue *and* have one item per worker in hand -- the second
        # term is easy to forget and is half the total. Measured max on this graph is exactly this.
        ceiling = len(workflow.order) * (DEPTH * workers + workers) + 1
        assert len(reached) <= ceiling, f"read {len(reached)} ahead, bound is {ceiling}"


class TestOrderedNodes:
    """A node that says its calls are a sequence is given one, whatever the worker count."""

    def sink(self, kind: str) -> Workflow:
        return Workflow.from_dict(document(
            {"a": ("make", {}), "j": ("dawdle", {"ms": 5}), "w": (kind, {})},
            [("a", "image", "j", "image"), ("j", "image", "w", "image")]))

    def test_an_ordered_sink_sees_every_item_in_order(self):
        SEQUENCE = workflow_nodes.SEQUENCE
        SEQUENCE.clear()
        widths = list(range(2, 42))
        list(self.sink("append_ordered").run_many(widths, over="width", workers=4))
        assert SEQUENCE == widths

    def test_an_unordered_sink_does_not(self):
        """The falsification: without the flag the same graph scrambles, so the flag is doing it."""
        RECORD = workflow_nodes.RECORD
        RECORD.clear()
        widths = list(range(2, 42))
        list(self.sink("measure").run_many(widths, over="width", workers=4))
        seen = [width for _, width in RECORD if _ == "measure"]
        assert seen and seen != sorted(seen), "expected disorder without ordered=True"

    def test_a_failed_item_still_takes_its_turn(self):
        """Otherwise the counter waits on a number never coming and the node stops for good."""
        SEQUENCE = workflow_nodes.SEQUENCE
        SEQUENCE.clear()
        workflow = self.sink("append_ordered")
        items = [w if w % 4 else None for w in range(2, 30)]
        failures: list = []
        list(workflow.run_many(items, over="width", workers=4,
                               on_error=lambda item, event: failures.append(item)))
        assert failures, "the None items should have failed"
        assert SEQUENCE == [w for w in items if w is not None], "order survived the failures"


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


class TestFailureStopsTheWholeItem:
    """A failure ends the item, not merely the branches downstream of it.

    The serial engine returns at the first failure, so nothing after it in topological order runs
    -- including nodes on branches that never depended on it. A pipeline has no such moment: the
    failure travels along wires, and a branch beside the failure has no wire from it. Left alone,
    a node with a side effect runs for an item the caller is told failed, and *which* nodes run
    depends on ``workers``.
    """

    #: The one item that fails. The others keep the run alive around it, which is what makes the
    #: question askable: a run that has ended stops its stages, and then no sink runs anywhere.
    BAD = 4
    ITEMS = [2, 3, BAD, 5, 6, 7]

    def graph(self) -> Workflow:
        """``make`` into two branches: one fails on a single item, the other reaches a sink.

        The sink sits behind a steady pause so that the failure has landed while the other branch
        is still inside a node -- the window the fix closes. ``sink`` is declared after ``boom``,
        so the serial engine's topological order reaches the failure first and never gets to it.
        """
        return Workflow.from_dict(document(
            {"a": ("make", {}), "boom": ("explode_on", {"on": self.BAD}),
             "slow": ("linger", {"ms": 40}), "sink": ("measure", {})},
            [("a", "image", "boom", "image"), ("a", "image", "slow", "image"),
             ("slow", "image", "sink", "image")]))

    def test_a_branch_beside_the_failure_does_not_run(self):
        workflow = self.graph()
        survivors = [item for item in self.ITEMS if item != self.BAD]
        for workers in (1, 2, 4):
            workflow_nodes.RECORD.clear()
            failures: list = []
            out = list(workflow.run_many(self.ITEMS, over="width", workers=workers,
                                         on_error=lambda item, event: failures.append(item)))
            reached = sorted(width for what, width in workflow_nodes.RECORD if what == "measure")
            assert failures == [self.BAD], f"only {self.BAD} should fail at workers={workers}"
            assert [item for item, _ in out] == survivors
            assert reached == survivors, (
                f"at workers={workers} the sink ran for {sorted(set(reached) - set(survivors))}, "
                f"an item the caller was told had failed")


class TestLettingGo:
    """A run the caller walked away from leaves nothing behind.

    ``run_many`` is documented for sources that never end, which makes stopping part way the
    ordinary case rather than the exceptional one. Every thread a run starts is a daemon, so a
    leak is invisible in a script that exits and unbounded in a process that does not.
    """

    def settled(self, deadline: float = 5.0) -> list:
        """The pipeline's own threads still alive, once they have had time to notice."""
        limit = time.monotonic() + deadline
        names = lambda: [thread.name for thread in threading.enumerate()
                         if thread.name.startswith(("pipeline-feed", "stage-"))]
        while time.monotonic() < limit and names():
            time.sleep(0.05)
        return names()

    def test_stopping_part_way_leaves_no_threads_behind(self):
        run = chain().run_many(itertools.count(2), over="width", workers=2)
        next(run), next(run)
        run.close()
        assert self.settled() == []

    def test_a_failure_that_raises_leaves_no_threads_behind(self):
        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("explode", {})}, [("a", "image", "b", "image")]))
        with pytest.raises(RuntimeError, match="as promised"):
            list(workflow.run_many(itertools.count(2), over="width", workers=2))
        assert self.settled() == []

    def test_an_on_error_that_raises_leaves_no_threads_behind(self):
        def angry(item, event):
            raise RuntimeError("the caller's budget ran out")

        workflow = Workflow.from_dict(document(
            {"a": ("make", {}), "b": ("explode", {})}, [("a", "image", "b", "image")]))
        with pytest.raises(RuntimeError, match="budget"):
            list(workflow.run_many(itertools.count(2), over="width", workers=2, on_error=angry))
        assert self.settled() == []


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
