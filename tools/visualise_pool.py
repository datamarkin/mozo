#!/usr/bin/env python3
"""Watch a pool of K workers fill a GPU, or fail to.

    python tools/visualise_pool.py
    python tools/visualise_pool.py --seed 7 --out pool.mp4

Three random workflows, each run twice -- **top row K=4, bottom row K=1**, same workflow, same
seed, same clock. The only difference down a column is how many items are in flight, which is the
whole question: does one item at a time leave the hardware idle?

**This simulates, it does not execute.** A GPU node here can cost 1.5 s, and a hundred thousand
items through one of those at K=1 is forty-one hours -- there is no version of this that runs for
real inside a fifteen second video. So it is a discrete-event simulation of the design in
``engine-plan.md``: K workers, each walking one item through the whole graph, one node at a time.
Branches do not run in parallel *within* an item, because in that design they do not -- one worker
runs its own item's branches in order. Parallelism comes from the other items.

**How a node occupies its device, and why the two devices differ.** A CPU is many cores, so
several CPU nodes genuinely run at once and each takes its share. **A GPU is one device, so it is
exclusive**: one GPU node at a time, whatever its occupancy. Modelling the GPU the way the CPU is
modelled would let three 30% nodes run concurrently and each still finish in its own time -- free
parallelism, which no GPU performs, and it would flatter the busy row for a reason that does not
exist.

So a GPU node's 25-90% is *occupancy*, not a share to be divided: it says how much of the device
that model actually fills while it owns it, and the rest is headroom nobody gets. Which is the
point -- a small model leaves most of the GPU idle even when the GPU is busy, and the only way to
win back the time is to stop the device from going idle *between* items. That is what K is for.

The numbers under each panel are real for the simulated workload: items finished in the window, and
what that is per simulated second.
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter   # noqa: E402
from matplotlib.patches import FancyArrowPatch, Circle   # noqa: E402

# --- what the video is ---------------------------------------------------------------------------

SECONDS = 15
FPS = 30
FRAMES = SECONDS * FPS
COLUMNS = 3
ITEMS = 100_000
POOLS = (4, 1)                 # top row, bottom row
WINDOW_ITEMS = 45              # how many K=4 items the visible window should cover

#: Milliseconds a node costs, and how much of its device it occupies while it runs.
COST = {"cpu": (10, 300), "gpu": (30, 400)}
#: A CPU node's share is divided among cores; a GPU node's is how well it fills a device it owns
#: outright. Drawn from what real stages cost -- decode, letterbox, NMS, mask assembly and drawing
#: run 15-45% of a multi-core CPU, and a small detector fills only a quarter of a GPU.
DEMAND = {"cpu": (0.15, 0.45), "gpu": (0.25, 0.90)}
#: Devices that run one thing at a time. A CPU is not one of them.
EXCLUSIVE = {"gpu"}
#: The hop between two nodes. Small, but it is not nothing, and at K=1 nothing fills it.
IDLE_MS = 1.0

#: How much one execution of a node varies from the next, as a coefficient of variation. A CPU
#: stage is content-dependent -- a frame holding thirty objects costs more to suppress and more to
#: draw than one holding two -- so it varies a great deal. A GPU kernel is far steadier.
#:
#: **Zero here is not neutral, it is its own artefact.** With every execution costing exactly the
#: same, the workers start together and never drift: they arrive at the same node at the same
#: moment forever, a standing wave that leaves gaps no amount of K can fill. Real variance
#: diffuses them apart and those gaps close, which is why the deterministic version understates
#: what a pool is worth.
JITTER = {"cpu": 0.45, "gpu": 0.12}
#: Items to let through before the video starts, so the burst from starting K workers at once has
#: washed out and what is on screen is the steady state.
WARMUP_ITEMS = 6

INK = "#e8e8ec"
DIM = "#6b6b78"
GRID = "#26262e"
BACK = "#131318"
CPU_COLOUR = "#4fa8ff"
GPU_COLOUR = "#ff7a45"
FLOW_COLOUR = "#ffd166"


# --- the workload --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """One step of a workflow: what it costs, and what it holds while it runs."""

    name: str
    kind: str          # "cpu" | "gpu"
    cost: float        # milliseconds
    demand: float      # fraction of its device, 0..1
    layer: int
    slot: int          # position within the layer, for drawing


@dataclass
class Flow:
    """A random workflow: nodes in the order one worker walks them, plus how to draw it."""

    nodes: list
    edges: list                      # (from index, to index) into nodes
    label: str = ""
    layers: dict = field(default_factory=dict)

    @property
    def serial_ms(self) -> float:
        """What one item costs with nothing in its way -- every node plus every hop."""
        return sum(node.cost for node in self.nodes) + IDLE_MS * (len(self.nodes) - 1)


def random_flow(rng: random.Random) -> Flow:
    """A chain three to six deep, sometimes splitting in two and rejoining.

    Kept deliberately modest: one branch is enough to show that a worker runs its own branches in
    order, and more than two makes a 640px panel unreadable without teaching anyone anything.
    """
    depth = rng.randint(3, 6)
    widths = [1]
    for _ in range(depth - 2):
        # A split only from a single node, and never twice running, so the shapes stay legible.
        widths.append(2 if widths[-1] == 1 and rng.random() < 0.45 else 1)
    widths.append(1)

    nodes, layers = [], {}
    for layer, width in enumerate(widths):
        layers[layer] = []
        for slot in range(width):
            # At least one GPU node, and the first layer is CPU so something feeds the device.
            kind = "cpu" if layer == 0 else ("gpu" if rng.random() < 0.45 else "cpu")
            nodes.append(Node(
                name=f"{layer}.{slot}",
                kind=kind,
                cost=rng.uniform(*COST[kind]),
                demand=rng.uniform(*DEMAND[kind]),
                layer=layer,
                slot=slot,
            ))
            layers[layer].append(len(nodes) - 1)

    if not any(node.kind == "gpu" for node in nodes):
        victim = rng.randrange(1, len(nodes))
        nodes[victim] = Node(nodes[victim].name, "gpu", rng.uniform(*COST["gpu"]),
                             rng.uniform(*DEMAND["gpu"]), nodes[victim].layer, nodes[victim].slot)

    edges = []
    for layer in range(len(widths) - 1):
        for source in layers[layer]:
            for target in layers[layer + 1]:
                edges.append((source, target))

    gpus = sum(1 for node in nodes if node.kind == "gpu")
    return Flow(nodes=nodes, edges=edges, layers=layers,
                label=f"{len(nodes)} nodes · {gpus} on GPU")


# --- the simulation ------------------------------------------------------------------------------

@dataclass
class Trace:
    """What happened, sampled finely enough to draw."""

    changes: list          # (t, cpu used, gpu used) at every change
    inside: list           # (t, [count per node]) at every change
    finished: list         # t of each item completion
    horizon: float


def simulate(flow: Flow, pool: int, items: int, horizon: float, seed: int = 0) -> Trace:
    """Run *items* through *flow* with *pool* workers, stopping at *horizon* milliseconds.

    A worker holds one item and walks every node in order, waiting at each for the device it
    needs. The GPU is exclusive -- one node at a time -- while the CPU is shared by share. Waiting
    is first-come-first-served: a worker that cannot fit does not get skipped by a smaller one
    behind it. That is head-of-line blocking, it is what a queue in front of a device actually
    does, and pretending otherwise would flatter the busy row.

    Every execution of a node is priced separately, around that node's own cost -- see
    :data:`JITTER` for why a fixed price is the wrong answer rather than the neutral one.
    """
    rng = random.Random(seed)
    order = list(range(len(flow.nodes)))
    used = {"cpu": 0.0, "gpu": 0.0}
    queued = {"cpu": deque(), "gpu": deque()}
    inside = [0] * len(flow.nodes)

    changes = [(0.0, [0.0, 0.0])]
    counts = [(0.0, list(inside))]
    finished = []

    events: list = []                 # (time, tie, worker, "arrive" | "release", node index)
    tie = 0
    handed = min(pool, items)

    for worker in range(handed):
        heapq.heappush(events, (0.0, tie, worker, "arrive", 0))
        tie += 1

    def note(now: float) -> None:
        changes.append((now, [used["cpu"], used["gpu"]]))
        counts.append((now, list(inside)))

    def spent(node: Node) -> float:
        """What *this* execution of *node* costs. Lognormal, so the tail is slow runs, not fast."""
        spread = JITTER[node.kind]
        if spread <= 0.0:
            return node.cost
        sigma = math.sqrt(math.log(1.0 + spread * spread))
        return node.cost * math.exp(rng.gauss(-0.5 * sigma * sigma, sigma))

    def admit(now: float, worker: int, index: int) -> bool:
        """Start *worker* on node *index* if the device it needs will have it."""
        node = flow.nodes[order[index]]
        if node.kind in EXCLUSIVE:
            if used[node.kind] > 0.0:
                return False            # the device is busy; occupancy does not divide
        elif used[node.kind] + node.demand > 1.0 + 1e-9:
            return False
        used[node.kind] += node.demand
        inside[order[index]] += 1
        nonlocal tie
        heapq.heappush(events, (now + spent(node), tie, worker, "release", index))
        tie += 1
        note(now)
        return True

    while events:
        now, _, worker, what, index = heapq.heappop(events)
        if now > horizon:
            break

        if what == "arrive":
            node = flow.nodes[order[index]]
            if not admit(now, worker, index):
                queued[node.kind].append((worker, index))
            continue

        # A release: free the share, move the worker on, then see who was waiting.
        node = flow.nodes[order[index]]
        used[node.kind] -= node.demand
        inside[order[index]] -= 1

        following = index + 1
        if following < len(order):
            heapq.heappush(events, (now + IDLE_MS, tie, worker, "arrive", following))
            tie += 1
        else:
            finished.append(now)
            if handed < items:
                handed += 1
                heapq.heappush(events, (now, tie, worker, "arrive", 0))
                tie += 1

        for kind in ("cpu", "gpu"):
            while queued[kind]:
                waiting_worker, waiting_index = queued[kind][0]
                if not admit(now, waiting_worker, waiting_index):
                    break            # head of line blocks; that is the point
                queued[kind].popleft()
        note(now)

    return Trace(changes=changes, inside=counts, finished=finished, horizon=horizon)


def sample(log: list, when: list) -> list:
    """Step-hold a ``(time, values)`` log onto the frame times, in one pass.

    Both logs carry a list as their second field so this works on either without asking which.
    """
    out, cursor, held = [], 0, log[0][1]
    for moment in when:
        while cursor < len(log) and log[cursor][0] <= moment:
            held = log[cursor][1]
            cursor += 1
        out.append(held)
    return out


# --- the drawing ---------------------------------------------------------------------------------

def positions(flow: Flow, aspect: float) -> list:
    """Where each node sits, laid out left to right by layer.

    ``x`` runs to *aspect* rather than to 1 so the stage's data coordinates are square, which is
    the only way a :class:`Circle` on it comes out round rather than as an ellipse.
    """
    depth = max(node.layer for node in flow.nodes) + 1
    spots = []
    for node in flow.nodes:
        width = len(flow.layers[node.layer])
        x = aspect * (0.08 + 0.84 * (node.layer / max(1, depth - 1)))
        y = 0.5 if width == 1 else (0.5 + (0.26 if node.slot == 0 else -0.26))
        spots.append((x, y))
    return spots


class Panel:
    """One workflow at one pool size: a utilisation strip above, the graph below."""

    def __init__(self, figure, box, flow: Flow, trace: Trace, pool: int, when: list):
        self.flow, self.pool = flow, pool
        page = figure.get_size_inches()
        aspect = (box[2] * page[0]) / (box[3] * 0.58 * page[1])
        self.spots = positions(flow, aspect)

        self.chart = figure.add_axes([box[0], box[1] + box[3] * 0.62, box[2], box[3] * 0.30])
        self.stage = figure.add_axes([box[0], box[1], box[2], box[3] * 0.58])
        for axes in (self.chart, self.stage):
            axes.set_facecolor(BACK)
            for spine in axes.spines.values():
                spine.set_visible(False)
            axes.set_xticks([])
            axes.set_yticks([])

        self.util = sample(trace.changes, when)
        self.busy = sample(trace.inside, when)
        self.finished = trace.finished
        self.when = when
        self.opened = when[0]          # the window starts after the warm-up, not at zero

        # --- the utilisation strip
        self.chart.set_xlim(0, len(when))
        self.chart.set_ylim(0, 1.34)
        for level in (0.5, 1.0):
            self.chart.axhline(level, color=GRID, lw=0.7, zorder=0)
        self.cpu_line, = self.chart.plot([], [], color=CPU_COLOUR, lw=1.6)
        self.gpu_line, = self.chart.plot([], [], color=GPU_COLOUR, lw=1.6)
        self.chart.text(-0.012, 0.745, "100%", color=DIM, fontsize=7, ha="right", va="center",
                        transform=self.chart.transAxes, family="monospace")
        self.chart.text(-0.012, 0.0, "0", color=DIM, fontsize=7, ha="right", va="center",
                        transform=self.chart.transAxes, family="monospace")
        self.readout = self.chart.text(0.0, 0.875, "", color=INK, fontsize=9, ha="left",
                                       transform=self.chart.transAxes, family="monospace")
        self.title = self.chart.text(0.012, 1.22, "", color=INK, fontsize=10.5,
                                     transform=self.chart.transAxes, weight="bold")
        self.title.set_text(f"K = {pool}" + ("   ·   one item at a time" if pool == 1 else
                                             f"   ·   {pool} items in flight"))

        # --- the graph
        self.stage.set_xlim(0, aspect)
        self.stage.set_ylim(0, 1)
        for source, target in flow.edges:
            start, end = self.spots[source], self.spots[target]
            self.stage.add_patch(FancyArrowPatch(
                start, end, arrowstyle="-", color=GRID, lw=1.1, zorder=1,
                shrinkA=13, shrinkB=13))

        self.blobs, self.haloes = [], []
        for index, node in enumerate(flow.nodes):
            colour = GPU_COLOUR if node.kind == "gpu" else CPU_COLOUR
            radius = 0.030 + 0.045 * node.demand
            halo = Circle(self.spots[index], radius * 1.9, color=colour, alpha=0.0, zorder=2)
            blob = Circle(self.spots[index], radius, facecolor=BACK, edgecolor=colour,
                          lw=1.6, zorder=3)
            self.stage.add_patch(halo)
            self.stage.add_patch(blob)
            self.haloes.append(halo)
            self.blobs.append(blob)

        self.counter = self.stage.text(0.5, 0.045, "", color=DIM, fontsize=8.5, ha="center",
                                       transform=self.stage.transAxes, family="monospace")

    def draw(self, frame: int) -> None:
        start = max(0, frame - 90)
        history = range(start, frame + 1)
        self.cpu_line.set_data(list(history), [self.util[i][0] for i in history])
        self.gpu_line.set_data(list(history), [self.util[i][1] for i in history])
        self.chart.set_xlim(max(0, frame - 90), max(90, frame) + 2)

        cpu, gpu = (max(0.0, value) for value in self.util[frame])
        self.readout.set_text(f"GPU {gpu*100:3.0f}%      CPU {cpu*100:3.0f}%")

        for index, blob in enumerate(self.blobs):
            live = self.busy[frame][index]
            self.haloes[index].set_alpha(0.30 if live else 0.0)
            blob.set_facecolor(
                (GPU_COLOUR if self.flow.nodes[index].kind == "gpu" else CPU_COLOUR)
                if live else BACK)

        until = self.when[frame]
        done = sum(1 for moment in self.finished if self.opened < moment <= until)
        elapsed = (until - self.opened) / 1000.0
        rate = done / elapsed if elapsed > 0 else 0.0
        self.counter.set_text(f"{done:>4} items    {rate:5.1f}/s")


def render(seed: int, out: Path, columns: int, items: int) -> None:
    rng = random.Random(seed)
    flows = [random_flow(rng) for _ in range(columns)]

    # One window per column, opening after a warm-up so the burst from starting K workers together
    # is off screen, sized so the busy row shows enough items to read, and shared with the quiet
    # row underneath so the comparison is honest.
    plans, spans = [], []
    for flow in flows:
        opens = flow.serial_ms * WARMUP_ITEMS
        probe = simulate(flow, POOLS[0], items, seed=seed,
                         horizon=opens + flow.serial_ms * WINDOW_ITEMS)
        settled = [moment for moment in probe.finished if moment > opens]
        closes = (settled[WINDOW_ITEMS - 1] if len(settled) >= WINDOW_ITEMS
                  else max(probe.horizon, opens + flow.serial_ms * 2))
        spans.append((opens, closes))

    for flow, (opens, closes) in zip(flows, spans):
        plans.append([simulate(flow, pool, items, horizon=closes * 1.02, seed=seed)
                      for pool in POOLS])

    figure = plt.figure(figsize=(19.2, 10.8), dpi=100)
    figure.patch.set_facecolor(BACK)

    figure.text(0.5, 0.955, "One worker, or four?", color=INK, fontsize=27,
                ha="center", weight="bold")
    figure.text(0.5, 0.918,
                "The same workflow, the same hardware. Only the number of items in flight changes.",
                color=DIM, fontsize=13, ha="center")

    panels = []
    for column, (flow, (opens, closes)) in enumerate(zip(flows, spans)):
        when = [opens + (closes - opens) * frame / (FRAMES - 1) for frame in range(FRAMES)]
        left = 0.035 + column * 0.323
        for row, pool in enumerate(POOLS):
            bottom = 0.505 - row * 0.435
            panels.append(Panel(figure, (left, bottom, 0.285, 0.345),
                                flow, plans[column][row], pool, when))
        counts = [sum(1 for moment in plan.finished if opens < moment <= closes)
                  for plan in plans[column]]
        gain = counts[0] / counts[1] if counts[1] else float("inf")
        figure.text(left + 0.1425, 0.882, flow.label, color=DIM, fontsize=10, ha="center")
        figure.text(left + 0.1425, 0.462, f"{gain:.1f}x the items, same hardware",
                    color=FLOW_COLOUR, fontsize=11.5, ha="center", weight="bold")

    figure.text(0.5, 0.022,
                "simulated · the GPU runs one node at a time · CPU nodes share cores · "
                "a small model fills only a quarter of a GPU even while it owns it",
                color=DIM, fontsize=10, ha="center")
    figure.text(0.972, 0.022, f"seed {seed}", color=DIM, fontsize=10, ha="right")

    writer = (FFMpegWriter(fps=FPS, bitrate=6000, codec="libx264")
              if out.suffix == ".mp4" else PillowWriter(fps=FPS))
    with writer.saving(figure, str(out), dpi=100):
        for frame in range(FRAMES):
            for panel in panels:
                panel.draw(frame)
            writer.grab_frame(facecolor=BACK)
    plt.close(figure)

    print(f"\n{out}  ({SECONDS}s, {FRAMES} frames, seed {seed})\n")
    for column, flow in enumerate(flows):
        opens, closes = spans[column]
        counts = [sum(1 for moment in plan.finished if opens < moment <= closes)
                  for plan in plans[column]]
        gain = counts[0] / counts[1] if counts[1] else float("inf")
        print(f"  column {column + 1}: {flow.label:<22} window {(closes - opens)/1000:6.1f}s   "
              f"K=4 {counts[0]:>4} items   K=1 {counts[1]:>4} items   {gain:.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=random.randrange(10_000),
                        help="the workflows are random; this makes one of them repeatable")
    parser.add_argument("--out", type=Path, default=Path("pool.mp4"),
                        help=".mp4 needs ffmpeg; .gif does not")
    parser.add_argument("--columns", type=int, default=COLUMNS)
    parser.add_argument("--items", type=int, default=ITEMS)
    arguments = parser.parse_args()
    render(arguments.seed, arguments.out, arguments.columns, arguments.items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
