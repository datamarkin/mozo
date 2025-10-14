# Mozo

Universal computer vision model serving library with dynamic model management.

Mozo provides a unified interface for deploying models from multiple ML frameworks (Detectron2, HuggingFace Transformers) with intelligent lifecycle management, lazy loading, and automatic memory cleanup.

## Key Features

- **Multi-Framework Support** - Detectron2, HuggingFace Transformers, and custom adapters
- **PixelFlow Integration** - All detection models return PixelFlow Detections for unified filtering and annotation
- **Lazy Loading** - Models load on-demand, not at startup
- **Thread-Safe** - Concurrent requests handled safely with per-model locks
- **Automatic Cleanup** - Inactive models are automatically unloaded to free memory
- **REST API** - FastAPI server for HTTP access
- **Registry-Factory-Manager Pattern** - Clean separation of concerns for model management

## Installation

```bash
# Basic installation
pip install mozo

# For development (editable mode)
pip install -e .

# Framework dependencies (install as needed)
pip install transformers torch torchvision
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Quick Start

### Python API

```python
from mozo.manager import ModelManager
import pixelflow as pf
import cv2

# Initialize manager
manager = ModelManager()

# Load model (lazy loads on first call)
model = manager.get_model('detectron2', 'mask_rcnn_R_50_FPN_3x')

# Run prediction - returns PixelFlow Detections
image = cv2.imread('example.jpg')
detections = model.predict(image)

# Filter with PixelFlow
detections = detections.filter_by_confidence(0.7)

# Annotate with PixelFlow
annotated = pf.annotate.box(image, detections)
annotated = pf.annotate.label(annotated, detections)
cv2.imwrite('output.jpg', annotated)

# Cleanup
manager.unload_model('detectron2', 'mask_rcnn_R_50_FPN_3x')
```

### REST API

```bash
# Start server
python -m mozo
# or
uvicorn mozo.server:app --reload --port 8000

# Run prediction
curl -X POST "http://localhost:8000/predict/detectron2/mask_rcnn_R_50_FPN_3x" \
  -F "file=@image.jpg"

# List available models
curl http://localhost:8000/models

# List loaded models
curl http://localhost:8000/models/loaded

# Cleanup inactive models
curl -X POST "http://localhost:8000/models/cleanup?inactive_seconds=600"
```

## Supported Models

### Detectron2 (Object Detection & Instance Segmentation)
- **Task Types:** Object detection, instance segmentation, keypoint detection
- **22 variants** including Mask R-CNN, Faster R-CNN, RetinaNet
- **Recommended:** `mask_rcnn_R_50_FPN_3x` (instance segmentation), `faster_rcnn_R_50_FPN_3x` (detection)
- **Output:** PixelFlow Detections object (80 COCO classes)

### Depth Anything (Depth Estimation)
- **Task Type:** Monocular depth estimation
- **3 variants:** `small`, `base`, `large`
- **Recommended:** `small` for speed, `large` for accuracy
- **Output:** PIL Image (grayscale depth map)

### Qwen2.5-VL (Vision-Language)
- **Task Type:** Visual question answering, image captioning, VQA
- **1 variant:** `7b-instruct` (requires 16GB+ VRAM)
- **Output:** dict with `text`, `prompt`, `variant` keys

## Architecture

```
Client → ModelManager → ModelFactory → MODEL_REGISTRY → Adapter Classes
                ↓
         Thread-safe caching
         Usage tracking
         Automatic cleanup
```

**Components:**
- **Registry** (`mozo/registry.py`) - Central catalog of available models
- **Factory** (`mozo/factory.py`) - Dynamic adapter instantiation
- **Manager** (`mozo/manager.py`) - Lifecycle management, caching, cleanup
- **Adapters** (`mozo/adapters/`) - Framework-specific implementations
- **Server** (`mozo/server.py`) - FastAPI REST API

## Development

```bash
# Install in development mode
pip install -e .

# Start server with auto-reload
uvicorn mozo.server:app --reload --port 8000

# Test model loading
python -c "from mozo.manager import ModelManager; import cv2; \
  manager = ModelManager(); \
  model = manager.get_model('detectron2', 'mask_rcnn_R_50_FPN_3x'); \
  print('✅ Model loaded successfully')"
```

### Adding New Model Families

1. Create adapter class in `mozo/adapters/your_model.py`
2. Register in `mozo/registry.py` with variants and parameters
3. Test with ModelManager

## PixelFlow Integration

All detection models return PixelFlow Detections objects, providing a unified interface for filtering, annotation, and export across different ML frameworks.

**Learn more:**
- **Documentation:** https://pixelflow.datamarkin.com
- **GitHub:** https://github.com/datamarkin/pixelflow

## Memory Management

```python
# Manual unload
manager.unload_model('detectron2', 'mask_rcnn_R_50_FPN_3x')

# Automatic cleanup (models inactive for >10 minutes)
count = manager.cleanup_inactive_models(inactive_seconds=600)

# Unload all models
manager.unload_all_models()
```

## Model ID Format

Models are identified by `{family}/{variant}`:
- `detectron2/mask_rcnn_R_50_FPN_3x`
- `depth_anything/small`
- `qwen2.5_vl/7b-instruct`

## Environment Variables

```bash
# Enable MPS fallback for macOS (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Configure HuggingFace cache location
export HF_HOME=~/.cache/huggingface
```

## Documentation

- **CLAUDE.md** - Detailed implementation guidance, common patterns, integration examples
- **Repository:** https://github.com/datamarkin/mozo
- **Issues:** https://github.com/datamarkin/mozo/issues

## License

MIT License
