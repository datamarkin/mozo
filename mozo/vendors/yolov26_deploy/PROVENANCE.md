# Provenance

This package is deployment-only YOLO26 inference. Like its `yolov8_deploy`, `yolov11_deploy` and
`yolov12_deploy` siblings, and unlike mozo's other vendors, it is **not an extraction of an
upstream repository**: it does not contain, import or depend on `ultralytics`, and it reproduces no
model definition. It reads the checkpoint file directly and rebuilds the network from what that
file records.

| | |
|---|---|
| Relationship to `ultralytics` | none — no import, no dependency, no copied source |
| Written | independently, against the checkpoint format |
| Validated against | activations captured from the reference implementation |
| Harvested into mozo | 2026-08-19 |
| Verified with | `torch` 2.11.0, Python 3.10, on CPU |
| Checkpoint source | `ultralytics/assets` release `v8.4.0`, digests read from GitHub — see `tools/fetch/_ultralytics.py` |
| Corresponding source | https://github.com/ultralytics/ultralytics |
| Checkpoint writer | `ultralytics` 8.3.222, recorded in the checkpoints (2025-12-15 – 2026-01-07) |
| Reference | `ultralytics` 8.4.0, matching the assets release — `tools/verify/yolov26_reference.py` |
| Upstream commit | not pinned; the release the checkpoints came from is |

It is a separate vendor from its siblings rather than a shared substrate. That is a deliberate
call: a vendor is meant to be readable and replaceable on its own, and its numbers must stay
reproducible against its own upstream, which a shared substrate would put at the mercy of another
family's refactor. The adapters and the publishing tools are *not* duplicated, for the same reason
inverted: they contain no model maths, so nothing there can move a number.

## What makes this family different

**It is NMS-free.** The head is trained to fire once per object, so the network returns a ranked
detection list — `(1, max_det, 6)` rows of `x1, y1, x2, y2, score, class` — rather than a raw
per-anchor head. The box decode, the anchor grid and a two-stage top-k all live on the network in
`network.py`, because they need the anchor grid.

Three consequences, all of which the rest of mozo absorbs without a special case:

- `image.py` has a letterbox and its inverse and **no third function**. There is no `suppress`
  here, and nothing to overlap.
- `model.py`'s `detect` takes `(image, forward, imgsz, conf)` and no more. That is a strict subset
  of what the siblings take, and exactly what `mozo.adapters._yolo` already passes, so the shared
  adapter serves this family unchanged.
- `tests/test_vendor_agreement.py` holds invariants across the vendors. The letterbox ones apply
  here; the two suppression ones do not, and both report the family they skipped rather than
  dropping it in silence. Which families suppress is decided once, in `suppresses()`, so the two
  tests cannot disagree about it.

Only the `one2one_*` branches are evaluated. The `cv2`/`cv3` pair is the training-time
one-to-many assignment head and `dfl` is an identity, because this model regresses box distances
with a single bin. Both are still built, so the strict weight load stays complete: 0.15 M of
2.56 M parameters on nano, 6%. That check is worth the 6% — on the sibling YOLO12 it is what
surfaced an undocumented parameter that would otherwise have loaded silently and never been read.

## Why it is not a derivative

A `.pt` file is a ZIP archive holding one pickle plus the raw bytes of every tensor storage, and
that pickle records the whole module tree: each layer's class name, its wiring, and every leaf
module's hyperparameters — channel counts, kernel sizes, strides, paddings, batch-norm epsilons
(this family trains with 1e-3, an order above torch's default, and it is read rather than assumed),
attention head counts, per-level strides, the image size, the detection budget and the class names.

`reader.py` walks that pickle with a restricted unpickler. Every class the file names resolves to
an inert placeholder that keeps its name and attributes; nothing from the framework that wrote the
file is imported or executed. `build.py` then constructs the matching `torch.nn` modules from those
recorded numbers. There is no YAML parser, no width or depth scaling, no `make_divisible`, no
padding rule and no head-width formula, because every number those would produce is already in the
file.

The one thing a checkpoint does not record is the *dataflow* of composite blocks — which child
feeds which — and that is what `flow.py` supplies. Names like `cv1`, `cv2`, `m`, `nc`,
`one2one_cv2` and `nl` appear because they are read out of the file; they are interface facts about
a format, not copied expression.

One recorded thing is deliberately **not** read: the head's `anchors`, `strides` and `shape`
attributes. They are a cache from the last batch the model saw in training and are wrong for any
other input size, so the grid is rebuilt from the feature maps in hand.

## Licensing

The code here is **Apache-2.0**, like the rest of mozo.

Model weights are not. Checkpoints published by Ultralytics — including every variant mozo
publishes for this family — are **AGPL-3.0**, or covered by a commercial licence from Ultralytics.
mozo redistributes them unmodified, with the AGPL text and a NOTICE naming the exact upstream
release, in the same directory as the weights. An ONNX export contains the weights and is covered
by the same terms. Serving predictions from them over a network places AGPL-3.0 section 13
obligations on whoever runs the service; that is the operator's responsibility.

The two are separate works travelling together, which is what the GPL's aggregation clause permits.
Running an AGPL checkpoint through this code does not place this code under AGPL, and this code
does not change the checkpoint's terms.

## Measured parity

Reproducible, not remembered. `python tools/verify/yolov26_reference.py --variant <v>` runs both
implementations and prints this table; `tools/verify/yolov26_reference.json` is what it last wrote.
Every number is a maximum absolute difference, on `tests/fixtures/images/example.jpg` (1920×1281),
CPU, fused, `ultralytics` 8.4.0 against `torch` 2.11.0.

**These numbers carry tolerances, and that is structural.** This package is not an extraction of
upstream source: it is an independent implementation built from what the checkpoint records. Two
implementations of the same arithmetic in a different operator order do not agree bit for bit, so
what is measured is a maximum absolute difference against a stated bound rather than equality.

| Check | Tolerance | nano | small | medium | large | xlarge |
|---|---|---|---|---|---|---|
| Preprocessed input | 1e-6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Worst of all 23 layer outputs | 2e-3 | 1.08e-04 | 1.95e-04 | 1.89e-04 | 8.35e-04 | 9.58e-04 |
| Head boxes, px, above `conf` | 1e-2 | 7.78e-04 | 5.49e-04 | 2.75e-04 | 4.43e-04 | 1.53e-03 |
| Head boxes, px, all 8,400 anchors | — | 2.61e-03 | 2.81e-03 | 4.27e-03 | 1.24e-02 | 2.69e-03 |
| Head scores | 1e-3 | 8.49e-07 | 5.66e-06 | 3.99e-06 | 1.66e-05 | 1.37e-06 |
| Detections, boxes (px) | 1e-2 | 7.78e-04 | 5.49e-04 | 2.75e-04 | 4.43e-04 | 1.53e-03 |
| Detections, scores | 1e-3 | 8.49e-07 | 5.66e-06 | 3.99e-06 | 1.66e-05 | 1.37e-06 |
| Detections, class ids | exact | equal | equal | equal | equal | equal |
| Detections, count at `conf=0.001` | exact | equal | equal | equal | equal | equal |

155 comparisons across the five published variants, all within tolerance.

**Why the head is measured twice.** Over the whole anchor grid, `large` reads 1.24e-02 px — above
the 1e-2 the other rows are held to. That worst anchor is 6540, and its score is 0.000000: no
caller can receive it, and neither implementation reads it, because both take a top-k before
anything else. Over the 99 anchors that clear `conf=0.001` the same image reads 2.44e-04 px, and
above 0.01 it reads 6.10e-05. So the whole grid is reported and the anchors a caller can reach are
gated. Gating on the grid would hold the family to the float noise of its own dead anchors, which
can fail with nothing wrong; gating quietly on the survivors alone would hide that the grid moves.

**The padding case the old table could not see is now covered.** The two photographs this table was
previously measured on both letterboxed to whole-pixel padding — 80.0 and 140.0 — so they agreed
under either convention, and PROVENANCE recorded that as the fourth harvest in a row whose parity
suite could not see the half-pixel disagreement. `example.jpg` letterboxes to a spare of 106.5,
floor 106 and ceil 107, so the convention is load-bearing in every number above.

**The gate has been made to fail, and it localises.** `--falsify all` perturbs one constant at a
time and reports which stages move:

| perturbation | moves | leaves alone |
|---|---|---|
| anchor offset 0.5 → 0.0 | the boxes | all 23 layers, every score |
| all three strides collapsed to one | the boxes | all 23 layers, every score |
| top-k budget halved | the detection count and its rows | all 23 layers, the head |
| a constant added to every fused bias | 17 layers, then the head and detections | the preprocessing |
| sigmoid before the top-k instead of after | nothing | everything — the control |

The last row is the one that makes the other four mean something: a harness that fails at
everything has not been shown to localise, so one perturbation that provably cannot change a
number is run alongside the four that can.

**What this still does not cover.** The ONNX graph. `tools/verify/yolov26.py` compares it against
the torch path within one hundredth of a pixel, and that is a two-path comparison like every other
everyday check; the reference script runs torch only.

## What mozo changed on harvesting

The package arrived shaped as a standalone distribution. Nothing about the network, the checkpoint
reader or the block arithmetic was touched except where noted.

- **Removed:** `pyproject.toml`, `cli.py`, `__main__.py` and the test suite. A vendor is not
  separately installable and has no command line.
- **Moved:** `export.py` to `tools/export/yolov26.py`. Exporting is something you do once when
  publishing, not something a deployment package does.
- **Flattened and split** along the siblings' seams: `ckpt.py` → `reader.py`; `blocks.py` divided
  into `build.py` (the leaf table and `Block`) and `flow.py` (the dataflow table); `model.py` →
  `network.py`; `detector.py` divided into `image.py` and `model.py`.
- **Added `device` and the `detect()` seam.** The harvest ran on the CPU only and had no way to
  plug a graph runtime into the forward pass.
- **RGB instead of BGR:** `image.py` no longer decodes anything. It arrived reading files with
  `cv2.imread` and flipping to RGB while building the batch; mozo decodes in one place
  (`mozo.image.load_image`, RGB), so the decode was removed and the flip with it. Verified rather
  than assumed: the tensor handed to the network is bit-identical, `max|delta| = 0.0`.
- **Corrected the padding it reports.** The harvest returned the unrounded half and argued for it:
  *"rounding it would shift every box by up to half a pixel back in the original image whenever the
  total padding is odd."* That has it backwards. The content is written at `math.floor(spare_y)`,
  so the floor **is** the offset between the two coordinate systems; the unrounded value describes
  a placement that never happened. The cost is `0.5 / gain` source pixels — 1.5 px on mozo's
  fixture photograph, on every box, always in the same direction.
- **Pruned the retained activations.** `forward` held every layer's output to the end of the pass;
  of 23 layers only 7 are ever read back, which is 44.6 MB held where 8.2 MB is needed on nano.
- **Kept `compare_detection_sets`**, moved to `model.py`. It pairs detection sets by class and
  geometry rather than by position, and it matters more here than for the siblings because the
  ranking happens inside the graph — two executors of the same top-k are free to break ties
  differently. Comparing the full 300-row output by position reads 0.54 px where content pairing
  reads 0.001.

`tools/verify/yolov26.py` re-runs the vendor-against-mozo comparison over any images you give it.
