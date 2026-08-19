# Provenance

This package is deployment-only YOLO12 inference. Like its `yolov8_deploy` and `yolov11_deploy`
siblings, and unlike mozo's other vendors, it is **not an extraction of an upstream repository**:
it does not contain, import or depend on `ultralytics`, and it reproduces no model definition. It
reads the checkpoint file directly and rebuilds the network from what that file records.

| | |
|---|---|
| Relationship to `ultralytics` | none — no import, no dependency, no copied source |
| Written | independently, against the checkpoint format |
| Validated against | per-layer activations captured from the original implementation |
| Harvested into mozo | 2026-08-19 |
| Verified with | `torch` 2.11.0, Python 3.10, on CPU |
| Upstream repository | _(record it here — this package arrived as a standalone tree)_ |
| Upstream commit | _(record it here)_ |

It is a separate vendor from its siblings rather than a shared substrate with a third block table.
That is a deliberate call: a vendor is meant to be readable and replaceable on its own, and its
numbers must stay reproducible against its own upstream, which a shared substrate would put at the
mercy of another family's refactor. The cost is real and is recorded here rather than left to be
discovered — a fix to the checkpoint reader, the builder or the box arithmetic has to land three
times. The adapters and the publishing tools are *not* duplicated for the same reason: they contain
no model maths, so nothing there can move a number.

## Why it is not a derivative

A `.pt` file is a ZIP archive holding one pickle plus the raw bytes of every tensor storage, and
that pickle records the whole module tree: each layer's class name, its wiring, and every leaf
module's hyperparameters — channel counts, kernel sizes, strides, paddings, batch-norm epsilons,
attention head counts, area splits, DFL bin counts, per-level strides and the class names.

`reader.py` walks that pickle with a restricted unpickler. Every class the file names resolves to
an inert placeholder that keeps its name and attributes; nothing from the framework that wrote the
file is imported or executed. `build.py` then constructs the matching `torch.nn` modules from those
recorded numbers. There is no YAML parser, no width or depth scaling, no `make_divisible`, no
padding rule and no head-width formula, because every number those would produce is already in the
file.

The one thing a checkpoint does not record is the *dataflow* of composite blocks — which child
feeds which — and that is what `flow.py` supplies. Names like `cv1`, `cv2`, `m`, `nc`, `reg_max`,
`area` and `num_heads` appear because they are read out of the file; they are interface facts about
a format, not copied expression.

## The one thing that had to be inferred

`A2C2f` in the `large` and `xlarge` variants records a per-channel `gamma` that the smaller
variants do not have. What a recorded *tensor* is for is not something the file states, so this is
the only place in the package where a reading had to be chosen rather than read — and it is
recorded here because inference deserves evidence.

The builder refuses a checkpoint that records a tensor the model has nowhere to put, which is how
this surfaced at all: `yolo12l.pt` failed to load with `model.6: records unusable ['gamma']`
rather than quietly ignoring it. Three readings were then tested on the fixture photograph:

| reading | `large` | `xlarge` |
|---|---|---|
| ignore `gamma` | 300 detections at confidence 1.000, naming ovens, toilets and teddy bears | 287 detections |
| `gamma` as a plain output scale | 0 detections | 0 detections |
| **`gamma` scaling a residual branch** | **14 detections, top 0.956, mean 0.80** | **14 detections, top 0.954** |

`medium` is the largest variant that records no `gamma` and therefore needs no interpretation. It
finds 14 detections on the same photograph at a top confidence of 0.946 and a mean of 0.804. The
residual reading is the only one that makes the larger variants behave like it.

The shapes agree independently: `gamma` is recorded only on blocks whose input and output channel
counts are equal — the only place a residual can be formed at all — and it is shaped exactly like
those channels (512 for `large`, 768 for `xlarge`).

## Licensing

The code here is **Apache-2.0**, like the rest of mozo.

Model weights are not. Checkpoints published by Ultralytics — including every variant mozo
publishes for this family — are **AGPL-3.0**, or covered by a commercial licence from Ultralytics.
mozo redistributes them unmodified, with the AGPL text and a NOTICE naming the exact upstream
release, in the same directory as the weights. An ONNX or CoreML export contains the weights and is
covered by the same terms. Serving predictions from them over a network places AGPL-3.0 section 13
obligations on whoever runs the service; that is the operator's responsibility.

The two are separate works travelling together, which is what the GPL's aggregation clause permits.
Running an AGPL checkpoint through this code does not place this code under AGPL, and this code
does not change the checkpoint's terms.

## Measured parity

Against per-layer activations captured from the original implementation, on `yolo12n.pt` at
640×640, compared at `conf=0.001` and `iou=0.7`. Every number is a maximum absolute difference.

| Check | bus.jpg | zidane.jpg | Tolerance |
|---|---|---|---|
| Preprocessed input tensor | 0 (bit-exact) | 0 (bit-exact) | 1e-6 |
| Worst of all 21 top-level layers | 1.70e-05 | 2.15e-05 | 2e-3 |
| Head output, boxes | 1.28e-03 px | 1.80e-03 px | 1e-2 px |
| Head output, class scores | 9.54e-07 | 8.05e-07 | 1e-3 |
| Detections, count | 190 / 190 | 187 / 187 | exact |
| Detections, boxes | 1.40e-03 px | 7.32e-04 px | 1e-2 px |
| Detections, scores | 4.66e-07 | 8.05e-07 | 1e-3 |
| Exported ONNX, detections | 190 / 190 | 187 / 187 | exact |
| Exported ONNX, boxes / scores | 1.83e-03 px / 9.5e-07 | 2.20e-03 px / 2.7e-06 | 1e-2 px / 1e-3 |

Class ids match one to one in every case.

**Two things this table cannot show.** It was measured on `yolo12n.pt` alone, which records no
`gamma` — so it could not have caught the reading described above, on the two variants that do.
And both reference photographs letterbox to whole-pixel padding, 80.0 and 140.0, so they agree
under either padding convention and the half-pixel disagreement below is invisible to every number
in it. `tests/test_vendor_agreement.py` and the family's recorded-detection fixture cover what
the table cannot.

## What mozo changed on harvesting

The package arrived shaped as a standalone distribution. Nothing about the network, the checkpoint
reader or the block arithmetic was touched except where noted; what changed is everything that
assumed it was a library rather than a vendor, plus three corrections.

- **Removed:** `pyproject.toml`, `cli.py`, `__main__.py` and the test suite. A vendor is not
  separately installable and has no command line.
- **Moved:** `export.py` to `tools/export/yolov12.py`. `__init__.py` re-exported it and it imports
  `onnx` at module scope, so importing the package at all required a library mozo does not depend
  on. Exporting is something you do once when publishing.
- **Flattened:** `yolov12_deploy/yolov12/*.py` up to `yolov12_deploy/`, and split along the
  siblings' seams: `checkpoint.py` → `reader.py`, `dataflow.py` → `flow.py`, and `detector.py`
  divided into `image.py` (letterbox, suppression, coordinate mapping) and `model.py` (the
  detector).
- **Added `device` and the `detect()` seam.** The harvest ran on the CPU only and had no way to
  plug a different runtime into the forward pass. `detect()` is letterbox → *forward* → suppress →
  map back, so torch, ONNX and CoreML share one pre- and post-processing path.
- **RGB instead of BGR:** `image.py` no longer decodes anything. It arrived reading files with
  `cv2.imread` and flipping to RGB while building the batch; mozo decodes in one place
  (`mozo.image.load_image`, RGB), so the decode was removed and the flip with it. Verified rather
  than assumed: the tensor handed to the network is bit-identical, `max|delta| = 0.0`.
- **Corrected the padding it reports.** The harvest placed the resized image at `floor(spare)` but
  returned `spare` unrounded, and its docstring argued for that — "the half-padding itself stays
  fractional, because that is the true offset of the image inside the canvas". It is not: the
  content is written at `floor`, so that is where it is. The difference is `0.5 / gain` source
  pixels, 1.5 px on mozo's fixture photograph, on every box, always in the same direction. Both
  siblings had the same disagreement and it is now measured for all three.
- **Widened the class-separating band in suppression** from `max + 1` to `max - min + 1`. Boxes
  leave the letterbox with negative coordinates whenever a detection runs off the padded edge, and
  a band narrower than the full span lets two classes occupy the same shifted range and suppress
  each other. Both siblings already computed it this way.
- **Reconciled the suppression threshold.** `predict` defaulted to `iou=0.45` where both siblings
  use `0.7` — and where this package's own parity table above was measured at `0.7`.
- **Registered block-owned parameters**, which is what made the `gamma` reading above possible to
  find rather than possible to miss.

`tools/verify/yolov12.py` re-runs the vendor-against-mozo comparison over any images you give it.
