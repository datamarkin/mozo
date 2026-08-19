# Provenance

This package is deployment-only YOLO11 inference. Like its `yolov8_deploy` sibling and unlike
mozo's other vendors it is **not an extraction of an upstream repository**: it does not contain,
import or depend on `ultralytics`, and it reproduces no model definition. It reads the checkpoint
file directly and rebuilds the network from what that file records.

| | |
|---|---|
| Relationship to `ultralytics` | none — no import, no dependency, no copied source |
| Written | independently, against the checkpoint format |
| Validated against | reference tensors captured from the original implementation |
| Harvested into mozo | 2026-08-19 |
| Verified with | `torch` 2.11.0, Python 3.10, on CPU |
| Upstream repository | _(record it here — this package arrived as a standalone tree)_ |
| Upstream commit | _(record it here)_ |

It is a separate vendor from `yolov8_deploy` rather than a shared substrate with a second block
table, even though roughly four fifths of the two is the same design written twice. That is a
deliberate call: a vendor is meant to be readable and replaceable on its own, and the families
diverge at the head as the line moves on. The cost is real and is recorded here so it is a known
cost rather than an accident — a fix to the checkpoint reader, the builder or the box arithmetic
has to land in both.

## Why it is not a derivative

A `.pt` file is a ZIP archive holding one pickle plus the raw bytes of every tensor storage, and
that pickle records the whole module tree: each layer's class name, its wiring, and every leaf
module's hyperparameters — channel counts, kernel sizes, strides, paddings, batch-norm epsilons,
groups, attention head counts and softmax scales, DFL bin counts, per-level strides and the class
names.

`reader.py` walks that pickle with a restricted unpickler. Every class the file names resolves to
an inert placeholder that keeps its name and attributes; nothing from the framework that wrote the
file is imported or executed, and the builtins it will resolve are whitelisted. `build.py` then
constructs the matching `torch.nn` modules from those recorded numbers. There is no YAML parser, no
width or depth scaling, no `make_divisible`, no padding rule and no head-width formula, because
every number those would produce is already in the file.

The one thing a checkpoint does not record is the *dataflow* of composite blocks — which child
feeds which — and that is what `flow.py` supplies, written from an understanding of what each block
computes. Names like `cv1`, `cv2`, `m`, `nc`, `reg_max`, `num_heads` and `key_dim` appear because
they are read out of the file; they are interface facts about a format, not copied expression.

## Licensing

The code here is **Apache-2.0**, like the rest of mozo.

Model weights are not. Checkpoints published by Ultralytics — including every variant mozo
publishes for this family — are **AGPL-3.0**, or covered by a commercial licence from Ultralytics.
mozo redistributes them unmodified, with the AGPL text and a NOTICE naming the exact upstream
release, in the same directory as the weights. An ONNX export contains the weights and is covered
by the same terms. Serving predictions from them over a network places AGPL-3.0 section 13
obligations on whoever runs the service; that is the operator's responsibility.

The two are separate works travelling together, which is what the GPL's aggregation clause
permits. Running an AGPL checkpoint through this code does not place this code under AGPL, and
this code does not change the checkpoint's terms.

## Measured parity

Against per-layer reference tensors captured from the original implementation, with `yolo11n.pt`
on `bus.jpg` (1080×810) and `zidane.jpg` (720×1280), fused, at `conf=0.001`, `iou=0.7`,
`max_det=300`. Every figure is a maximum absolute difference.

| Check | Tolerance | bus.jpg | zidane.jpg |
|---|---|---|---|
| Preprocessed input | 1e-6 | 0.0 | 0.0 |
| Worst of all 23 layers | 2e-3 | 4.18e-05 | 9.20e-05 |
| Head output, boxes (px) | 1e-2 | 2.14e-03 | 6.59e-03 |
| Head output, class scores | 1e-3 | 5.25e-06 | 1.85e-06 |
| Detections, count | exact | 249 = 249 | 191 = 191 |
| Detections, boxes (px) | 1e-2 | 1.10e-03 | 1.16e-03 |
| Detections, scores | 1e-3 | 6.56e-07 | 9.91e-07 |
| Detections, classes | exact | identical | identical |
| ONNX detections, boxes (px) | 1e-2 | 5.49e-04 | 2.08e-03 |
| ONNX detections, scores | 1e-3 | 7.15e-07 | 1.13e-06 |

Unfused inference produces the same 249 and 191 detections, with boxes within 1.53e-03 px of the
reference; its raw head boxes differ by up to 1.53e-02 px, slightly more than the fused tolerance,
because the reference itself was captured from a fused model.

The suite that produced this table did not travel with the package — it needed a checkpoint, a set
of photographs and a directory of reference tensors, all named through the environment. The numbers
are recorded here because deleting the tests would otherwise have deleted the only evidence.

**One thing this table cannot show.** Both reference photographs letterbox to whole-pixel padding —
80.0 and 140.0 — so they agree under either padding convention, and the half-pixel disagreement
described below is invisible to every number above. `tests/test_letterbox_geometry.py` covers what
the table cannot.

## What mozo changed on harvesting

The package arrived shaped as a standalone distribution. Nothing about the network, the checkpoint
reader or the block arithmetic was touched; what changed is everything that assumed it was a
library rather than a vendor, plus one correction.

- **Removed:** `pyproject.toml`, `__main__.py`, the command line and the test suite. A vendor is
  not separately installable and has no command line.
- **Moved:** `export.py` to `tools/export/yolov11.py`. `__init__.py` named it, and it imports
  `onnx` at module scope, so importing the package at all required a library mozo does not depend
  on. Exporting is something you do once when publishing, not something a deployment package does.
- **Flattened:** `yolov11_deploy/yolov11/*.py` up to `yolov11_deploy/`, and split along the same
  seams as the sibling vendor: `checkpoint.py` → `reader.py`, `dataflow.py` → `flow.py`,
  `model.py` merged into `build.py`, and `detector.py` divided into `image.py` (letterbox,
  suppression, coordinate mapping) and `model.py` (the detector).
- **Added the `detect()` seam:** letterbox → *forward* → suppress → map back, with the forward pass
  passed in. A torch module and an ONNX session then differ only in that middle step and share one
  pre- and post-processing path, which is the property mozo publishes two artifacts on.
- **RGB instead of BGR:** `image.py` no longer decodes anything. It arrived reading files with
  `cv2.imread` (BGR) and flipping to RGB while building the batch; mozo decodes in one place
  (`mozo.image.load_image`, RGB), so the decode was removed and the flip with it. The network sees
  a bit-identical tensor — the change cancels — and this was verified rather than assumed: on the
  fixture photograph the harvested pre-processing and the current one produce `max|delta| = 0.0`.
  `cv2.resize` and `cv2.copyMakeBorder` stay, because letterbox geometry is model maths.
- **Corrected the padding it reports.** The harvest placed the resized image at `floor(spare)` on
  the canvas but returned `spare` unrounded, and subtracted the unrounded value when mapping boxes
  back. For any source whose scaled side is odd that is half a canvas pixel of disagreement between
  where the image was put and where the arithmetic thinks it was, or `0.5 / gain` source pixels —
  1.5 px on mozo's fixture photograph, on every box, always in the same direction. `image.py` now
  reports the floored placement, which is the one that actually happened, and
  `tests/test_letterbox_geometry.py` holds both YOLO vendors to it by counting the border rows the
  letterbox really wrote. The sibling vendor already floored and did not change.

`tools/verify/yolov11.py` re-runs the vendor-against-mozo comparison over any images you give it.
