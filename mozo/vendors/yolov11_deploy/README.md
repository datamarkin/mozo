# yolov11_deploy

Deployment-only YOLO11 detection and instance segmentation. Apache-2.0 code that does not import,
depend on, or contain any code from the `ultralytics` package — it reads the checkpoint file
directly.

```python
from mozo.vendors.yolov11_deploy import Detector
from mozo.image import load_image

detector = Detector("yolo11n.pt")                      # imgsz=640, device="cpu", fuse_norm=True
found = detector.predict(load_image("photo.jpg"), conf=0.25, iou=0.7, max_det=300)

found.boxes        # (n, 4) float32 x1, y1, x2, y2 in the source image's pixels
found.scores       # (n,) float32
found.class_ids    # (n,) int64
found.names        # (n,) the class name of each detection
found.masks        # (n, h, w) bool from a Segment checkpoint, None from a Detect one
detector.names     # {class id: name} as recorded in the checkpoint
```

A segmentation checkpoint — `yolo11n-seg.pt` and its four siblings — goes through the same
`Detector` and the same call. It has a `Segment` head instead of a `Detect` one, which adds 32
mask coefficients per anchor and a prototype stack; `mask.py` combines them into one boolean mask
per surviving detection, at the source image's resolution. `masks` is `None` rather than empty
from a detection checkpoint, so the two are distinguishable without measuring a length.

Most users reach this through `mozo.adapters.yolov11`, which resolves the weights, picks the
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
it, suppresses and maps back to source pixels. Both runtimes share that pre- and post-processing,
so they cannot drift apart.

## Supported

- Detection checkpoints in the YOLO11 format (`task: detect`). Anything else is refused by name,
  including an end-to-end head, which needs no NMS and is not what this decodes.
- The module classes found in such a checkpoint, and only those: `Conv`, `DWConv`, `Bottleneck`,
  `C3k2`, `C3k`, `SPPF`, `C2PSA`, `PSABlock`, `Attention`, `Concat`, `DFL`, `Detect`, plus the
  `torch.nn` leaves `Conv2d`, `BatchNorm2d`, `SiLU`, `Identity`, `MaxPool2d`, `Upsample`,
  `Sequential` and `ModuleList`.
- CPU, CUDA or MPS, letterboxed square inference at any positive multiple of the model's coarsest
  stride.
- Checkpoints whose tensors are stored in half precision (released ones are): every tensor is
  loaded into the float32 module that needs it, and its shape is checked against that module first.

## Runtimes

mozo publishes `torch-fp32` and `onnx-fp32` for this family, and **no CoreML**. That is measured,
not assumed: the `C2PSA` attention block at layer 10 makes Apple's Metal graph compiler fail —
`MPSGraphExecutable.mm: failed assertion 'MLIR pass manager failed'` — which aborts the process
rather than raising. Layers 0–9 convert and run; the block converts and runs in isolation; the
assembled graph does not. CoreML restricted to CPU and the Neural Engine does work, accurately, at
23.5 ms against 10.4 ms for torch on MPS — so there would be nothing to gain even if it were safe.
See `tools/export/yolov11.py` for the numbers and `mozo/runtimes.py` for how the preference is
expressed.

## Licensing

The **code** here is Apache-2.0 (see `LICENSE`).

**Model weights are not covered by it.** Ultralytics-trained checkpoints, including every variant
mozo publishes, are AGPL-3.0 unless you hold an Ultralytics commercial licence. Anything exported
from them — an ONNX graph — contains the weights and carries the same terms. mozo publishes the
licence text and a NOTICE beside every checkpoint. If you serve predictions from these weights over
a network, AGPL-3.0 section 13 places obligations on you.

See `PROVENANCE.md` for why this package is not a derivative of `ultralytics`, the measured parity
against the original implementation, and what mozo changed when it was harvested. See `NOTICE` for
attribution.
