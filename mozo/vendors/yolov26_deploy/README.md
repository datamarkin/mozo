# yolov26_deploy

Deployment-only YOLO26 detection. Apache-2.0 code that does not import, depend on, or contain any
code from the `ultralytics` package — it reads the checkpoint file directly.

```python
from mozo.vendors.yolov26_deploy import Detector
from mozo.image import load_image

detector = Detector("yolo26n.pt")                      # imgsz=640, device="cpu", fuse_norm=True
found = detector.predict(load_image("photo.jpg"), conf=0.25)   # no iou: see below

found.boxes        # (n, 4) float32 x1, y1, x2, y2 in the source image's pixels
found.scores       # (n,) float32
found.class_ids    # (n,) int64
found.names        # (n,) the class name of each detection
detector.names     # {class id: name} as recorded in the checkpoint
```

Most users reach this through `mozo.adapters.yolov26`, which resolves the weights, picks the
runtime and returns a PixelFlow result. This package is the layer under that.

## How it works

The checkpoint is treated as the specification. A `.pt` file is a ZIP archive holding one pickle
and the raw bytes of every tensor, and that pickle records the whole module tree: each layer's
class name, each layer's wiring (`f`), and every leaf module's hyperparameters — channel counts,
kernel sizes, strides, paddings, groups, the batch-norm epsilon, attention head counts and softmax
scales, split widths, the number of DFL bins, the per-level strides and the class names.

So this package reads all of that and builds the matching `torch.nn` modules directly. There is no
YAML parser, no width or depth scaling, no `make_divisible`, no padding rule and no head-width
formula, because every number those would produce is already written in the file. The only
hand-written part is the *dataflow* of the composite blocks — which child feeds which — which is
the one thing the file does not record.

Two consequences worth knowing:

- Any width or depth (n/s/m/l/x) and any fine-tuned class count loads with no special-casing.
- The archive is read with `zipfile` + a restricted `pickle.Unpickler` + `numpy`. Classes named in
  the pickle are never imported; each resolves to an inert placeholder that keeps its name and
  attributes.

## Images and the missing step

`predict` takes an `HxWx3` RGB `uint8` array — what `mozo.image.load_image` returns. Nothing here
decodes a file. mozo decodes in exactly one place so that one piece of code decides channel order,
and a numpy array carries nothing that would let this package check that decision afterwards.

`detect` is the seam a non-torch runtime plugs into: it letterboxes, calls the *forward* you give
it, keeps what clears the threshold and maps it back to source pixels. Both runtimes share that,
so they cannot drift apart.

It is one step shorter than the siblings', and that is the architecture rather than an omission.
**This family is NMS-free.** The head fires once per object, the network returns a ranked detection
list, and no box ever suppresses another — so `detect` takes no `iou` and no `max_det`, and
`image.py` has no `suppress` function to call. The detection budget is fixed by the graph.

## Supported

- Detection checkpoints in the YOLO26 format, which means **an end-to-end head**. A classic head
  is refused by name: it would need the non-maximum suppression this package does not implement.
- The module classes found in such a checkpoint, and only those: `Conv`, `DWConv`, `Bottleneck`,
  `C3k2`, `C3k`, `SPPF`, `C2PSA`, `PSABlock`, `Attention`, `Concat`, `Detect`, plus the `torch.nn`
  leaves `Conv2d`, `BatchNorm2d`, `SiLU`, `Identity`, `MaxPool2d`, `Upsample`, `Sequential` and
  `ModuleList`.
- CPU, CUDA or MPS, letterboxed square inference at any positive multiple of the model's coarsest
  stride.
- Checkpoints whose tensors are stored in half precision (released ones are): every tensor is
  loaded into the float32 module that needs it, and its shape is checked against that module first.

## Runtimes

mozo publishes `torch-fp32` and `onnx-fp32`, and **no CoreML** — for two independent reasons.

The converter refuses outright: the in-graph top-k's gather indices lose their integer dtype
through `expand`, and `gather_along_axis` will not take fp32 indices. Casting them to int32 is a
real fix and it does convert. What happens next is the same Metal compiler abort YOLO11 hits —
`MPSGraphExecutable.mm: failed assertion 'MLIR pass manager failed'` — from the same `C2PSA`
attention block. Off the GPU it runs accurately, 0.00006 px on CPU and the Neural Engine, at
22.6 ms against 13.1 ms for torch on MPS.

So the fix is recorded and not applied: it unlocks an artifact slower than the one already
published. `tools/export/yolov26.py` has the detail.

## Licensing

The **code** here is Apache-2.0 (see `LICENSE`).

**Model weights are not covered by it.** Ultralytics-trained checkpoints, including every variant
mozo publishes, are AGPL-3.0 unless you hold an Ultralytics commercial licence. Anything exported from
them — an ONNX graph — contains the weights and carries the same terms. mozo publishes the
licence text and a NOTICE beside every checkpoint. If you serve predictions from these weights over
a network, AGPL-3.0 section 13 places obligations on you.

See `PROVENANCE.md` for why this package is not a derivative of `ultralytics`, the measured parity
against the original implementation, and what mozo changed when it was harvested. See `NOTICE` for
attribution.
