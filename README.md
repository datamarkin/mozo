# Mozo

Computer vision model server with automatic memory management.

Mozo provides HTTP and Python access to 43 published model variants across 9 model families —
object detection (RF-DETR, YOLOv8/11/12/26), promptable segmentation (SAM 2, EdgeTAM), concept
segmentation (SAM 3) and monocular depth (Depth Anything V2). Models load on-demand and clean up
automatically.

Every family is a **vendored, deployment-only extraction** of its upstream project, verified
bit-identical to it, so no family requires its upstream package to be installed. `pip install
mozo` and a model runs; there is no per-family build step and no Docker image.

## Quick Start

```bash
pip install mozo
mozo start
```

Server starts on `http://localhost:8000` with all models available via REST API.

### Examples

Object detection:
```bash
curl -X POST "http://localhost:8000/predict/rfdetr/medium" \
  -F "file=@image.jpg"
```

Depth estimation:
```bash
curl -X POST "http://localhost:8000/predict/depth_anything_v2/small" \
  -F "file=@image.jpg" -D headers.txt --output depth.png

# depth.png is a 16-bit PNG; headers.txt carries what it means:
#   X-Depth-Unit: metres | none      X-Depth-Min / X-Depth-Max: the endpoints
#   depth = min + png / 65535 * (max - min)
```

List available models:
```bash
curl http://localhost:8000/models
```

## Features

- **37 Published Variants** - RF-DETR (8), Depth Anything V2 (9) and four YOLO generations (5 each), weights hosted and hash-verified
- **Vendored Architectures** - no upstream package needed, each verified bit-identical to it
- **Multiple Runtimes** - the same model as torch, ONNX or CoreML, chosen automatically per device
- **Lazy Loading** - Models load on first use and are reused across requests
- **PixelFlow Integration** - Detection models return a unified format for filtering and annotation
- **Thread-Safe** - Concurrent requests share one loaded model, built once
- **Production Ready** - Multiple workers, configurable timeouts, health checks

## Installation

```bash
# Basic installation
pip install mozo

# Optional runtimes (install as needed)
pip install 'mozo[onnx]'     # onnxruntime
pip install 'mozo[coreml]'   # coremltools, macOS only — the fastest RF-DETR path on Apple silicon
```

No family needs an extra. The extras add *ways to execute* the same model, not models: without
them `runtime="auto"` simply does not select those artifacts.

## Available Models

### RF-DETR (8 variants)
Real-time transformer detection and instance segmentation by Roboflow. Apache 2.0.

- Detection: `nano`, `small`, `medium`, `large`
- Segmentation: `seg-nano`, `seg-small`, `seg-medium`, `seg-large`

Output: PixelFlow `Detections` — boxes, masks, class names, confidence scores

Every variant publishes two runtimes, and they are verified to return the same
detections:

```python
model = manager.get_model('rfdetr', 'small')                        # torch
model = manager.get_model('rfdetr', 'small', runtime='onnx-fp32')   # ONNX Runtime
```

Class names ship with the weights, so they are the vocabulary the checkpoint was
trained on rather than an assumption. A checkpoint of your own that carries no
names returns `class_id` with `class_name` unset — pass `labels=[...]` to name
them. Mozo never guesses a name.

### YOLOv8, YOLO11, YOLO12 and YOLO26 (5 variants each)
Real-time detection by Ultralytics. **The weights are AGPL-3.0**, unlike the rest of
mozo — see the licensing note below.

- All four families: `nano`, `small`, `medium`, `large`, `xlarge`

YOLO26 is NMS-free: its head fires once per object and the network returns a ranked
detection list, so there is no overlap threshold to tune. The others suppress in the
usual way. Either way `predict` takes a confidence threshold and returns the same
PixelFlow result.

Output: PixelFlow `Detections` — boxes, class names, confidence scores

```python
model = manager.get_model('yolov8', 'nano')                         # torch
model = manager.get_model('yolov11', 'nano', runtime='onnx-fp32')   # ONNX Runtime
```

YOLOv8 and YOLO12 also publish a CoreML artifact, which is by far the fastest way to run
them on Apple silicon. YOLO11 and YOLO26 do not: the `C2PSA` block they share makes
Apple's Metal graph compiler abort the process, and the configuration that avoids that is
slower than torch on MPS. `runtime="auto"` handles this by itself — it only ever chooses
among what a variant actually publishes, so nothing in mozo carries a per-family
exception.

Class names come from the checkpoint, so a fine-tuned model publishes its own
vocabulary. Mozo never guesses a name.

> **Licensing.** These weights are AGPL-3.0, or covered by a commercial licence from
> Ultralytics. Mozo's own code stays Apache-2.0 — they are separate works travelling
> together — but anything you export from them inherits their terms, and **serving
> predictions from them over a network places AGPL-3.0 section 13 obligations on you**.
> The full licence and a NOTICE naming the exact upstream release are published beside
> every checkpoint. Complying is the operator's responsibility.

### SAM 2 (4 variants) and EdgeTAM (1 variant)
Promptable segmentation: point at something and get back the thing you pointed at.
`sam2/{tiny,small,base_plus,large}` and `edgetam/edgetam`.

Every prompt is a set of points. A click is a point with a label — `1` to include, `0` to
exclude. A box is spelled as its two corners carrying reserved labels, because neither model has
a separate box input; the adapter writes those for you. Points and a box can be combined.

```python
from mozo import ModelManager
model = ModelManager().get_model("edgetam", "edgetam")

found = model.predict(image, points=[[820, 640]], labels=[1])   # three candidates, best first
found = model.predict(image, boxes=[40, 60, 300, 480])
found = model.predict(image, boxes=[40, 60, 300, 480], points=[[900, 700]], labels=[0])
```

`multimask_output=True` (the default) returns three candidate masks with the model's own
predicted IoU as the score, ranked. That is the right setting for a single click, which is
genuinely ambiguous about whether you meant the handle, the door or the car — take the first
row, or show all three. With a box the prompt is usually unambiguous and `multimask_output=False`
is tighter.

**Detections come back with `class_name=None`.** A click does not say what it clicked, and mozo
will not invent a name for it — a name comes from the weights or from the user. Pass `name="cat"`
if you know what you pointed at. `class_id` is the index of the prompt that produced the row, so
a batch of prompts stays separable.

The image encoder is the cost and it depends only on the image, so it is cached on pixel content
and a second prompt on the same photograph pays only for the decoder. On CPU at 2 MP: EdgeTAM
encodes in 272 ms and decodes in 33 ms; SAM 2 tiny encodes in 439 ms.

EdgeTAM is SAM 2 distilled for phones — a 9.1M-parameter image path against SAM 2 tiny's 31.4M —
and its masks agree with SAM 2 tiny's at 0.94 IoU on box prompts. Both are verified bit-identical
to their upstream implementations; see each vendor's `PROVENANCE.md`.

Unlike SAM 3, both families' published weights are Apache-2.0, the same as their code.

Only the torch runtime is served so far. SAM 2 also publishes ONNX and CoreML artifacts, which
this adapter refuses rather than quietly answering with torch: a promptable model exports as
several graphs and needs a runner that keeps the encode and the decode apart.

### SAM 3 (1 variant)
Promptable segmentation. Meta ships a single model rather than a size ladder, so there is
one variant: `sam3/sam3`.

Two ways to prompt it, off one checkpoint:

- **Name a concept** — `predict(image, "taxi")` returns every instance, with a mask, a
  box and a score. The phrase you searched for is the class name every detection carries,
  so there is no fixed vocabulary and nothing for mozo to guess. Pass a list —
  `predict(image, ["car", "person", "dog"])` — and you get one result carrying several
  classes, with `class_ids` indexing the prompts. Instances found by different prompts may
  overlap: ask for `"car"` and `"vehicle"` and the same car comes back under both names.
- **Point at one thing** — `Segmenter.segment(image, points, labels)` takes clicks
  (`1` include, `0` exclude), a box, or a previous mask to refine, and returns three
  candidate masks with predicted IoU. Reached through
  `mozo.vendors.sam3_deploy` rather than the adapter for now.

Both are verified bit-identical to Meta's implementation, stage by stage through the
model — see `mozo/vendors/sam3_deploy/PROVENANCE.md`.

Prompts are up to 32 tokens. Inference is a fixed 1008x1008 square — SAM 3 squashes rather
than letterboxing, so aspect ratio is not preserved.

The model wants a GPU. On Apple silicon MPS the image encoder is about 1.2 s and on CPU
about 5 s; the encode is cached, so further prompts on the same image cost only their own
decode. Concepts do not batch — the head takes one prompt at a time — so N concepts cost
one encode plus N decodes: on MPS, three concepts is about 2.1 s cold and 0.35 s each
afterwards. Encoded prompts are cached too (33 KB each), so the same three words on the
next image skip the text tower entirely.

> **Licensing — read this before deploying.** These weights are **not** open source. They
> carry Meta's **SAM License**, which no other family here does. It **restricts what they
> may be used for** — military, nuclear, espionage and weapons uses are prohibited — and
> those restrictions **flow through to whoever you serve predictions to**. It binds on use
> rather than on signing, and it must travel with the weights if you pass them on. Mozo's
> own SAM 3 code is Apache-2.0 and derived from `transformers`, not from
> `facebookresearch/sam3`; the code and the weights are separate works travelling together.
> The full licence and a NOTICE naming the exact upstream release are published beside the
> checkpoint. Complying is the operator's responsibility.

Output: PixelFlow `Detections` with masks, boxes and scores

### Depth Anything V2 (9 variants)
Monocular depth estimation, in two groups that are not interchangeable.

**Relative depth** — output is inverse depth on an arbitrary per-image scale: larger
means nearer, and that is all it means. Two images cannot be compared to each other,
and no value is a distance.

- `small` - Fastest, lowest memory (Apache-2.0)
- `base` - Balanced performance (**CC-BY-NC-4.0**, non-commercial)
- `large` - Best accuracy (**CC-BY-NC-4.0**, non-commercial)

**Metric depth** — output is in metres, from fine-tunes on Hypersim (indoor, 0–20 m)
and Virtual KITTI 2 (outdoor, 0–80 m). All Apache-2.0 per their model cards.

- `indoor-small`, `indoor-base`, `indoor-large`
- `outdoor-small`, `outdoor-base`, `outdoor-large`

`predictor.unit` is `"metres"` for the metric variants and `None` for the relative
ones — mozo never guesses a unit.

Output: `HxW` float32 array at the input's resolution

## Server

```bash
# Start with defaults (0.0.0.0:8000, auto-reload enabled)
mozo start

# Custom port
mozo start --port 8080

# Production mode with multiple workers
mozo start --workers 4

# Check version
mozo version
```

## API Reference

### Run Prediction
```http
POST /predict/{family}/{variant}
Content-Type: multipart/form-data
```

Parameters:
- `family` - Model family (`rfdetr`, `yolov8`, `yolov11`, `yolov12`, `yolov26`, `sam2`,
  `edgetam`, `sam3` or `depth_anything_v2`)
- `variant` - Model variant (e.g., `nano`, `indoor-small`)
- `file` - Image file
- `threshold` - Confidence threshold (detection models only)
- `labels` - Comma-separated class labels overriding the model defaults (detection models only)
- `text` - The concept to look for (prompted models only, required by them). **Repeat the
  parameter** to ask for several in one request: `?text=car&text=person`. It is deliberately
  not comma-separated the way `labels` is — a prompt is free text, so `?text=a person, holding
  a mug` stays one concept rather than becoming two wrong ones.

- `point`, `label` - A click, as `x,y` in the image's own pixels, and `1` to include it or `0`
  to exclude it (promptable models only). **Repeat both** to give several, in the same order:
  `?point=820,640&label=1&point=900,700&label=0`. `label` is required with `point` and has no
  default — guessing between include and exclude returns a confident mask of the wrong thing.
- `box` - A box, as `x1,y1,x2,y2` in the image's own pixels (promptable models only). May be
  combined with points.
- `name` - What to call what you pointed at (promptable models only). Omitted, detections come
  back with `class_name: null` — the model does not know what it segmented and mozo will not
  invent a name for it.
- `multimask` - Return three candidate masks ranked by predicted IoU rather than one
  (promptable models only, default `true`).

```bash
# One concept
curl -X POST "http://localhost:8000/predict/sam3/sam3?text=taxi" -F "file=@street.jpg"

# Several: one result carrying several classes, sharing one image encode
curl -X POST "http://localhost:8000/predict/sam3/sam3?text=car&text=person&text=dog" \
  -F "file=@street.jpg"

# Click one thing: three candidate masks, best first
curl -X POST "http://localhost:8000/predict/edgetam/edgetam?point=820,640&label=1" \
  -F "file=@street.jpg"

# Refine with a second click that excludes what you got too much of
curl -X POST "http://localhost:8000/predict/sam2/tiny?point=820,640&label=1&point=900,700&label=0" \
  -F "file=@street.jpg"

# A box, one mask, named by you
curl -X POST "http://localhost:8000/predict/sam2/tiny?box=40,60,300,480&multimask=false&name=cat" \
  -F "file=@street.jpg"
```

### Health Check
```http
GET /
```

Returns server status and loaded models.

### List Models
```http
GET /models
```

Returns all available model families and variants.

### List Loaded Models
```http
GET /models/loaded
```

Returns the models currently in memory.

## How It Works

**Lazy Loading**
Models load on first request, not at server startup. This keeps startup time instant regardless of available models.

**Smart Caching**
Loaded models stay in memory and are reused across requests. First request is slower (model download + load), subsequent requests are fast.

**Thread Safety**
A model is built once however many requests arrive for it at the same moment, and a request for a model already in memory never waits behind an unrelated load.

Example flow:
```bash
# Server starts instantly (no models loaded)
mozo start

# First request loads model
curl -X POST "http://localhost:8000/predict/rfdetr/medium" -F "file=@test.jpg"
# Output: [mozo] loading rfdetr/medium

# Subsequent requests reuse loaded model
curl -X POST "http://localhost:8000/predict/rfdetr/medium" -F "file=@test2.jpg"
# Served from memory, nothing logged
```

## Python SDK

For direct integration in Python applications:

```python
from mozo import ModelManager

manager = ModelManager()
model = manager.get_model('rfdetr', 'medium')

# A path, encoded bytes, or an RGB array. Decode with mozo.image.load_image rather than
# cv2.imread, which returns BGR and would silently give a slightly wrong answer.
detections = model.predict('image.jpg')

# Filter results
high_confidence = detections.filter_by_confidence(0.8)

# A separate manager is a separate lifetime — drop it and its models go with it
scratch = ModelManager()
```

### Custom Weights

Fine-tuned checkpoints load through the same API, on architectures Mozo supports:

```python
model = manager.get_model(
    'rfdetr', 'my-training',
    checkpoint_path='runs/best.pth',
    model_size='small', project_type='detection',
    labels=['hardhat', 'vest'],
)
```

### PixelFlow Integration

Detection models return PixelFlow Detections objects - a unified format across all ML frameworks:

```python
# Works the same for either family, or a checkpoint of your own
detections = model.predict(image)

# Filter and annotate
import pixelflow as pf
filtered = detections.filter_by_confidence(0.8).filter_by_class_id([0, 2])
annotated = pf.annotate.box(image, filtered)
annotated = pf.annotate.label(annotated, filtered)

# Export
json_output = filtered.to_json()
```

Learn more: [PixelFlow](https://github.com/datamarkin/pixelflow)

## Configuration

### Environment Variables

```bash
# Enable MPS fallback for macOS (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Configure HuggingFace cache location
export HF_HOME=~/.cache/huggingface
```

### Memory Management

Models load on first use and stay for the life of the process. Memory is yours to manage: load
what you need, and use a separate `ModelManager` when you want a separate lifetime.

## Extending Mozo

Add new models in 3 steps:

1. Create adapter in `mozo/adapters/your_model.py`
2. Register in `mozo/registry.py`
3. Use via HTTP or Python API

## Architecture

```
HTTP Request → FastAPI Server → ModelManager → Adapter → Vendor
                                      ↓
                                Thread-safe cache
```

Components:
- **Server** - FastAPI REST API
- **Manager** - Thread-safe cache of loaded models
- **Registry** - Catalog of families, answerable without importing torch
- **Adapters** - One per family, translating between mozo and a vendor
- **Weights** - Manifest lookup, download, hash verification
- **Runtimes** - Device detection and artifact selection (torch / ONNX / CoreML)
- **Image** - The one decode boundary: RGB, uint8, HxWx3

## Development

```bash
# Install in development mode
pip install -e .

# Start server with auto-reload
mozo start
```

## Documentation

- [Repository](https://github.com/datamarkin/mozo)
- [Issues](https://github.com/datamarkin/mozo/issues)

## License

MIT License
