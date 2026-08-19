# Provenance

This package is deployment-only YOLOv8 inference. Unlike mozo's other vendors it is **not an
extraction of an upstream repository**: it does not contain, import or depend on `ultralytics`,
and it reproduces no model definition. It reads the checkpoint file directly and rebuilds the
network from what that file records.

| | |
|---|---|
| Relationship to `ultralytics` | none — no import, no dependency, no copied source |
| Written | independently, against the checkpoint format |
| Validated against | reference tensors captured from the original implementation |
| Harvested into mozo | 2026-08-19 |
| Verified with | `torch` 2.11.0, Python 3.10, on CPU |
| Upstream repository | _(record it here — this package arrived as a standalone tree)_ |
| Upstream commit | _(record it here)_ |

## Why it is not a derivative

A `.pt` file is a ZIP archive holding one pickle plus the raw bytes of every tensor storage, and
that pickle records the whole module tree: each layer's class name, its wiring, and every leaf
module's hyperparameters — channel counts, kernel sizes, strides, paddings, batch-norm epsilons,
split widths, DFL bin counts, per-level strides and the class names.

`reader.py` walks that pickle with a restricted unpickler. Every class the file names resolves to
an inert placeholder that keeps its name and attributes; nothing from the framework that wrote the
file is imported or executed. `build.py` then constructs the matching `torch.nn` modules from those
recorded numbers. There is no YAML parser, no width or depth scaling, no `make_divisible`, no
padding rule and no head-width formula, because every number those would produce is already in the
file.

The one thing a checkpoint does not record is the *dataflow* of composite blocks — which child
feeds which — and that is what `flow.py` supplies, written from an understanding of what each
block computes. Names like `cv1`, `cv2`, `m`, `nc` and `reg_max` appear because they are read out
of the file; they are interface facts about a format, not copied expression.

## Licensing

The code here is **Apache-2.0**, like the rest of mozo.

Model weights are not. Checkpoints published by Ultralytics — including every variant mozo
publishes for this family — are **AGPL-3.0**, or covered by a commercial licence from Ultralytics.
mozo redistributes them unmodified, with the AGPL text and a NOTICE naming the exact upstream
release, in the same directory as the weights. An ONNX or CoreML export contains the weights and
is covered by the same terms. Serving predictions from them over a network places AGPL-3.0
section 13 obligations on whoever runs the service; that is the operator's responsibility.

The two are separate works travelling together, which is what the GPL's aggregation clause
permits. Running an AGPL checkpoint through this code does not place this code under AGPL, and
this code does not change the checkpoint's terms.

## Measured parity

Against per-layer reference tensors captured from the original implementation for `yolov8n.pt` at
640×640, on `bus.jpg` (1080×810) and `zidane.jpg` (720×1280). Detections compared at `conf=0.001`,
`iou=0.7` in original image coordinates, paired one to one by class.

| Check | bus.jpg | zidane.jpg | Tolerance |
|---|---|---|---|
| Preprocessed input, max abs diff | 0.0 | 0.0 | 1e-6 |
| Worst of all 22 layer outputs | 1.26e-05 | 1.57e-05 | 2e-3 |
| Head boxes, px | 1.56e-03 | 1.46e-03 | 1e-2 |
| Head class scores | 1.55e-06 | 2.24e-06 | 1e-3 |
| Detections found (reference count) | 233/233 | 277/277 | exact |
| Detection boxes, px | 1.31e-03 | 1.83e-03 | 1e-2 |
| Detection scores | 9.54e-07 | 6.41e-07 | 1e-3 |
| ONNX detections found (reference count) | 233/233 | 277/277 | exact |
| ONNX detection boxes, px | 2.59e-03 | 2.14e-03 | 1e-2 |
| ONNX detection scores | 8.05e-07 | 9.54e-07 | 1e-3 |

Every detection matched its reference by class, so class ids agree exactly. Conv+BatchNorm fusion
is on by default; with `fuse_norm=False` the head agrees with the fused model to 7.0e-03 px on
boxes and 6.4e-06 on scores, and the strict weight-load check always runs against the unfused
parameters.

The suite that produced this table is extraction scaffolding and did not travel with the package —
it needed a checkpoint, a set of photographs and a directory of reference `.npz` files, all named
through the environment. The numbers are recorded here because deleting the tests would otherwise
have deleted the only evidence.

## What mozo changed on harvesting

The package arrived shaped as a standalone distribution. Nothing about the network, the checkpoint
reader or the arithmetic was touched; what changed is everything that assumed it was a library
rather than a vendor.

- **Removed:** `pyproject.toml`, `cli.py`, `__main__.py`, and the test suite. A vendor is not
  separately installable and has no command line.
- **Moved:** `export.py` to `tools/export/yolov8.py`. `__init__.py` re-exported it, and it imports
  `onnx` at module scope, so importing the package at all required a library mozo does not depend
  on — three of the four YOLO vendors harvested at the same time failed to import on a clean
  install for exactly this reason. Exporting is something you do once when publishing, not
  something a deployment package does.
- **Flattened:** `yolov8_deploy/yolov8/*.py` up to `yolov8_deploy/`, matching the other vendors.
- **RGB instead of BGR:** `image.py` no longer decodes anything. It arrived reading files with
  `cv2.imread` (BGR) and flipping to RGB inside `letterbox`; mozo decodes in one place
  (`mozo.image.load_image`, RGB), so the decode was removed and the flip with it. The network sees
  a bit-identical tensor — the change cancels — and this was verified rather than assumed: on the
  fixture photograph the harvested package and the current one return the same 8 detections with
  `max|delta| = 0.0` on both boxes and scores.
  `cv2.resize` and `cv2.copyMakeBorder` stay, because letterbox geometry is model maths.

`tools/verify/yolov8.py` re-runs the vendor-against-mozo comparison over any images you give it.
