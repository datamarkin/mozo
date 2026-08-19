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
| Upstream repository | _(record it here — this package arrived as a standalone tree)_ |
| Upstream commit | _(record it here)_ |

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

Against activations captured from the reference implementation for `yolo26n.pt`, on `bus.jpg`
(1080×810) and `zidane.jpg` (720×1280). Every number is a maximum absolute difference.

| Check | Tolerance | bus.jpg | zidane.jpg |
|---|---|---|---|
| Preprocessed input | 1e-6 | 0.0 | 0.0 |
| All 23 recorded layer outputs | 2e-3 | 1.82e-04 | 9.14e-05 |
| Head output, boxes (px) | 1e-2 | 8.54e-04 | 9.46e-04 |
| Head output, scores | 1e-3 | 6.6e-07 | 1.3e-06 |
| Detections, boxes (px) | 1e-2 | 3.66e-04 | 6.71e-04 |
| Detections, scores | 1e-3 | 6.6e-07 | 1.3e-06 |
| Exported ONNX, boxes (px) | 1e-2 | 3.36e-04 | 1.71e-03 |
| Exported ONNX, scores | 1e-3 | 7.7e-07 | 1.2e-06 |
| Detection count at `conf=0.001` | exact | 177 / 177 | 125 / 125 |

Fusion is on for those numbers.

**What this table cannot show.** Both reference photographs letterbox to whole-pixel padding —
80.0 and 140.0 — so they agree under either padding convention, and the half-pixel disagreement
described below is invisible to every number in it. That is the fourth harvest in a row whose
parity suite could not see it. `tests/test_vendor_agreement.py` and the family's recorded-detection
fixture cover what the table cannot.

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
