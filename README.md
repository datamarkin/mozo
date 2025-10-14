# Mozo

Universal computer vision model server with automatic memory management and multi-framework support.

Mozo provides HTTP access to 25+ pre-configured models from Detectron2, HuggingFace Transformers, and other frameworks. Models load on-demand and clean up automatically.

## Quick Start

```bash
pip install mozo
mozo start
```

Server starts on `http://localhost:8000` with all models available via REST API.

### Examples

Object detection:
```bash
curl -X POST "http://localhost:8000/predict/detectron2/mask_rcnn_R_50_FPN_3x" \
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

- **25+ Pre-configured Models** - Detectron2, HuggingFace Transformers, custom adapters
- **Automatic Memory Management** - Lazy loading, usage tracking, automatic cleanup
- **Multi-Framework Support** - Unified API across different ML frameworks
- **PixelFlow Integration** - Detection models return unified format for filtering and annotation
- **Thread-Safe** - Concurrent request handling with per-model locks
- **Production Ready** - Multiple workers, configurable timeouts, health checks

## Installation

```bash
# Basic installation
pip install mozo

# Framework dependencies (install as needed)
pip install transformers torch torchvision
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Available Models

### Detectron2 (22 variants)
Object detection, instance segmentation, keypoint detection trained on COCO dataset.

Popular variants:
- `mask_rcnn_R_50_FPN_3x` - Instance segmentation
- `faster_rcnn_R_50_FPN_3x` - Object detection
- `faster_rcnn_X_101_32x8d_FPN_3x` - High-accuracy detection
- `keypoint_rcnn_R_50_FPN_3x` - Keypoint detection
- `retinanet_R_50_FPN_3x` - Single-stage detector

Output: JSON with bounding boxes, class names, confidence scores (80 COCO classes)

### Depth Anything (3 variants)
Monocular depth estimation.

- `small` - Fastest, lowest memory
- `base` - Balanced performance
- `large` - Best accuracy

Output: PNG grayscale depth map

### Qwen2.5-VL (1 variant)
Vision-language understanding for VQA, captioning, and image analysis.

- `7b-instruct` - 7B parameter model (requires 16GB+ RAM)

Output: JSON with text response

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
- `family` - Model family (e.g., `detectron2`, `depth_anything`, `qwen2.5_vl`)
- `variant` - Model variant (e.g., `mask_rcnn_R_50_FPN_3x`, `small`, `7b-instruct`)
- `file` - Image file
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
curl -X POST "http://localhost:8000/predict/detectron2/faster_rcnn_R_50_FPN_3x" -F "file=@test.jpg"
# Output: [ModelManager] Loading model: detectron2/faster_rcnn_R_50_FPN_3x...

# Subsequent requests reuse loaded model
curl -X POST "http://localhost:8000/predict/detectron2/faster_rcnn_R_50_FPN_3x" -F "file=@test2.jpg"
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
model = manager.get_model('detectron2', 'mask_rcnn_R_50_FPN_3x')

image = cv2.imread('image.jpg')
detections = model.predict(image)

# Filter results
high_confidence = detections.filter_by_confidence(0.8)

# Manual memory management
manager.unload_model('detectron2', 'mask_rcnn_R_50_FPN_3x')
manager.cleanup_inactive_models(inactive_seconds=300)
```

### PixelFlow Integration

Detection models return PixelFlow Detections objects - a unified format across all ML frameworks:

```python
# Works the same for Detectron2, YOLO, or custom models
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

See [CLAUDE.md](CLAUDE.md) for detailed implementation guide.

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
