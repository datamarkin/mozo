# yolov12_deploy

Deployment-only YOLO12 detection. Apache-2.0 code that does not import, depend on, or contain any
code from the `ultralytics` package — it reads the checkpoint file directly.

```python
from mozo.vendors.yolov12_deploy import Detector
from mozo.image import load_image

detector = Detector("yolo12n.pt")                      # imgsz=640, device="cpu", fuse_norm=True
found = detector.predict(load_image("photo.jpg"), conf=0.25, iou=0.7, max_det=300)

found.boxes        # (n, 4) float32 x1, y1, x2, y2 in the source image's pixels
found.scores       # (n,) float32
found.class_ids    # (n,) int64
found.names        # (n,) the class name of each detection
detector.names     # {class id: name} as recorded in the checkpoint
```

Most users reach this through `mozo.adapters.yolov12`, which resolves the weights, picks the
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

## Images

`predict` takes an `HxWx3` RGB `uint8` array — what `mozo.image.load_image` returns. Nothing here
decodes a file. mozo decodes in exactly one place so that one piece of code decides channel order,
and a numpy array carries nothing that would let this package check that decision afterwards.

`detect` is the seam a non-torch runtime plugs into: it letterboxes, calls the *forward* you give
it, suppresses and maps back to source pixels. Every runtime shares that pre- and post-processing,
so they cannot drift apart.

## Supported

- Detection checkpoints in the YOLO12 format (`task: detect`). Anything else is refused by name,
  including an end-to-end head, which needs no NMS and is not what this decodes.
- The module classes found in such a checkpoint, and only those: `Conv`, `DWConv`, `Bottleneck`,
  `C3k2`, `C3k`, `A2C2f`, `ABlock`, `AAttn`, `Concat`, `DFL`, `Detect`, plus the `torch.nn` leaves
  `Conv2d`, `BatchNorm2d`, `SiLU`, `Identity`, `Upsample`, `Sequential` and `ModuleList`.
- CPU, CUDA or MPS, letterboxed square inference at any positive multiple of the model's coarsest
  stride.
- Checkpoints whose tensors are stored in half precision (released ones are): every tensor is
  loaded into the float32 module that needs it, and its shape is checked against that module first.

## Runtimes

mozo publishes `torch-fp32`, `onnx-fp32` and `coreml-fp32` for this family. CoreML is by a wide
margin the fastest on Apple silicon — measured on one laptop, nano runs 7.0 ms against 14.3 ms on
torch MPS and 113.7 ms on torch CPU, at a worst box error of 0.0017 px.

That is worth stating because the sibling family cannot do it: YOLO11's `C2PSA` block makes Apple's
Metal graph compiler abort the process. This family's area-attention blocks (`A2C2f`, `ABlock`,
`AAttn`) convert and run cleanly. Checked on every compute-unit setting rather than assumed.

## Licensing

The **code** here is Apache-2.0 (see `LICENSE`).

**Model weights are not covered by it.** Ultralytics-trained checkpoints, including every variant
mozo publishes, are AGPL-3.0 unless you hold an Ultralytics commercial licence. Anything exported
from them — an ONNX graph or a CoreML package — contains the weights and carries the same terms. mozo publishes the
licence text and a NOTICE beside every checkpoint. If you serve predictions from these weights over
a network, AGPL-3.0 section 13 places obligations on you.

See `PROVENANCE.md` for why this package is not a derivative of `ultralytics`, the measured parity
against the original implementation, and what mozo changed when it was harvested. See `NOTICE` for
attribution.
