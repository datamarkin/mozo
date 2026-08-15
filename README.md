# Mozo

Universal computer vision model server with automatic memory management and multi-framework support.

Mozo provides HTTP access to 48 pre-configured model variants across 10 model families from RF-DETR, HuggingFace Transformers, PaddleOCR, EasyOCR, Detectron2, and other frameworks. Models load on-demand and clean up automatically.

> **Note:** the Detectron2 family (12 variants) is currently unavailable while it is
> reimplemented on exported artifacts. Loading it raises `NotImplementedError`.
> 36 variants across the other 9 families are usable today.

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
curl -X POST "http://localhost:8000/predict/depth_anything/small" \
  -F "file=@image.jpg" --output depth.png
```

Vision-language Q&A:
```bash
curl -X POST "http://localhost:8000/predict/qwen2.5_vl/7b-instruct?prompt=What%20is%20in%20this%20image" \
  -F "file=@image.jpg"
```

List available models:
```bash
curl http://localhost:8000/models
```

## Features

- **48 Pre-configured Model Variants** - 10 model families including RF-DETR, HuggingFace Transformers, PaddleOCR, EasyOCR, Florence-2, BLIP VQA, and more
- **Automatic Memory Management** - Lazy loading, usage tracking, automatic cleanup
- **Multi-Framework Support** - Unified API across different ML frameworks
- **PixelFlow Integration** - Detection models return unified format for filtering and annotation
- **Thread-Safe** - Concurrent request handling with per-model locks
- **Production Ready** - Multiple workers, configurable timeouts, health checks

## Installation

```bash
# Basic installation
pip install mozo

# Per-family dependencies (install as needed)
pip install 'mozo[rfdetr]'      # RF-DETR
pip install 'mozo[paddleocr]'   # PaddleOCR + PP-Structure
pip install 'mozo[easyocr]'     # EasyOCR
pip install 'mozo[qwen]'        # Qwen2.5-VL
```

Depth Anything, Florence-2 and BLIP VQA run on the core `transformers` dependency
and need no extra.

## Available Models

### RF-DETR (8 variants)
Real-time transformer detection and instance segmentation by Roboflow. Apache 2.0.

- Detection: `nano`, `small`, `medium`, `large`
- Segmentation: `seg-nano`, `seg-small`, `seg-medium`, `seg-large`

Output: PixelFlow `Detections` — boxes, masks, class names, confidence scores

### Depth Anything V2 (3 variants)
Monocular depth estimation.

- `small` - Fastest, lowest memory
- `base` - Balanced performance
- `large` - Best accuracy

Output: PNG grayscale depth map

### Florence-2 (8 variants)
Microsoft Florence-2 multi-task vision.

- Captioning: `captioning`, `detailed_captioning`, `more_detailed_captioning`
- OCR: `ocr`, `ocr_with_region`
- Detection: `detection`, `detection_with_caption`, `segmentation`

Output: JSON. Detection and segmentation are not fully implemented.

### PaddleOCR (5 variants)
PP-OCRv5 scene text recognition, 80+ languages.

- `mobile`, `server`, `mobile-chinese`, `server-chinese`, `mobile-multilingual`

Output: PixelFlow `Detections` with recognised text

### PP-StructureV3 (4 variants)
Document structure analysis — layout, tables, formulas.

- `layout-only`, `full`, `table-analysis`, `formula-analysis`

Output: JSON document structure

### EasyOCR (4 variants)
General-purpose OCR, 80+ languages.

- `english-light`, `english-full`, `multilingual`, `chinese`

Output: PixelFlow `Detections` with recognised text

### BLIP VQA (2 variants)
Salesforce BLIP visual question answering.

- `base`, `capfilt-large`

Output: JSON with text response

### Qwen2.5-VL (1 variant)
Vision-language understanding for VQA, captioning, and image analysis.

- `7b-instruct` - 7B parameter model (requires 16GB+ RAM)

Output: JSON with text response

### Qwen3-VL (1 variant)
Vision-language understanding with chain-of-thought reasoning.

- `2b-thinking`

Output: JSON with text response and reasoning trace

### Detectron2 (12 variants) — currently unavailable
Object detection, instance segmentation and keypoint detection on COCO. FPN
backbones only.

- Faster R-CNN: `faster_rcnn_R_50_FPN_1x`, `faster_rcnn_R_50_FPN_3x`, `faster_rcnn_R_101_FPN_3x`, `faster_rcnn_X_101_32x8d_FPN_3x`
- Mask R-CNN: `mask_rcnn_R_50_FPN_1x`, `mask_rcnn_R_50_FPN_3x`, `mask_rcnn_R_101_FPN_3x`, `mask_rcnn_X_101_32x8d_FPN_3x`
- Keypoint R-CNN: `keypoint_rcnn_R_50_FPN_1x`, `keypoint_rcnn_R_50_FPN_3x`, `keypoint_rcnn_R_101_FPN_3x`, `keypoint_rcnn_X_101_32x8d_FPN_3x`

These are listed by `/models` but raise `NotImplementedError` on load. The adapter
is being rebuilt so the family no longer requires a per-platform Detectron2 build.

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
- `family` - Model family (e.g., `rfdetr`, `depth_anything`, `qwen2.5_vl`)
- `variant` - Model variant (e.g., `medium`, `small`, `7b-instruct`)
- `file` - Image file
- `threshold` - Confidence threshold (detection models only)
- `labels` - Comma-separated class labels overriding the model defaults (detection models only)
- `prompt` - Text prompt (VLM models only)

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

Returns currently loaded models with usage information.

### Get Model Info
```http
GET /models/{family}/{variant}/info
```

Returns detailed information about a specific model variant.

### Unload Model
```http
POST /models/{family}/{variant}/unload
```

Manually unload a model to free memory.

### Cleanup Inactive Models
```http
POST /models/cleanup?inactive_seconds=600
```

Unload models inactive for specified duration (default: 600 seconds).

## How It Works

**Lazy Loading**
Models load on first request, not at server startup. This keeps startup time instant regardless of available models.

**Smart Caching**
Loaded models stay in memory and are reused across requests. First request is slower (model download + load), subsequent requests are fast.

**Usage Tracking**
Each model access updates a timestamp. Models inactive for 10+ minutes are automatically unloaded.

**Thread Safety**
Per-model locks ensure only one thread loads a given model. Other threads wait and reuse the loaded instance.

Example flow:
```bash
# Server starts instantly (no models loaded)
mozo start

# First request loads model
curl -X POST "http://localhost:8000/predict/rfdetr/medium" -F "file=@test.jpg"
# Output: [ModelManager] Loading model: rfdetr/medium...

# Subsequent requests reuse loaded model
curl -X POST "http://localhost:8000/predict/rfdetr/medium" -F "file=@test2.jpg"
# Output: [ModelManager] Model already loaded, reusing existing instance.

# After 10 minutes of inactivity, model auto-unloads
# Output: [ModelManager] Cleanup: Unloaded 1 inactive model(s).
```

## Python SDK

For direct integration in Python applications:

```python
from mozo import ModelManager
import cv2

manager = ModelManager()
model = manager.get_model('rfdetr', 'medium')

image = cv2.imread('image.jpg')
detections = model.predict(image)

# Filter results
high_confidence = detections.filter_by_confidence(0.8)

# Manual memory management
manager.unload_model('rfdetr', 'medium')
manager.cleanup_inactive_models(inactive_seconds=300)
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
# Works the same for RF-DETR, OCR models, or custom models
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

Models automatically unload after 10 minutes of inactivity. Adjust this:

```bash
curl -X POST "http://localhost:8000/models/cleanup?inactive_seconds=300"
```

Or in Python:
```python
manager.cleanup_inactive_models(inactive_seconds=300)
```

## Extending Mozo

Add new models in 3 steps:

1. Create adapter in `mozo/adapters/your_model.py`
2. Register in `mozo/registry.py`
3. Use via HTTP or Python API

## Architecture

```
HTTP Request → FastAPI Server → ModelManager → ModelFactory → Adapter → Framework
                                      ↓
                               Thread-safe cache
                               Usage tracking
                               Auto cleanup
```

Components:
- **Server** - FastAPI REST API
- **Manager** - Lifecycle management, caching, cleanup
- **Factory** - Dynamic adapter instantiation
- **Registry** - Central catalog of models
- **Adapters** - Framework-specific implementations

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
