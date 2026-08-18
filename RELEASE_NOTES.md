# Mozo v0.4.0: Real-Time Detection

## What's New in v0.4.0

### 🎉 New Model Family

**RF-DETR:** Real-time transformer-based object detection and instance segmentation by Roboflow
- 8 variants: nano, small, medium, large + seg-nano, seg-small, seg-medium, seg-large
- Pretrained COCO models with fine-tuned checkpoint support
- Apache 2.0 license

### 📊 Model Count

**Total Models:** 63 variants across 10 model families
**Growth:** +8 variants (RF-DETR), +1 family

### 🔧 Infrastructure Improvements

**Automatic device detection:** CPU/GPU (CUDA)/Apple MPS auto-selection integrated into `get_model()` and ModelFactory

**Unified image loading:** New `load_image` utility allows all adapters to accept both file paths and numpy arrays as input

**Simplified API:** New top-level `get_model()` function for direct access with shared ModelManager instance

### 🗑️ Removed

- **StabilityInpainting** adapter removed (cloud-based inpainting service discontinued)
- **Datamarkin** adapter removed (the hosted Vision Service is no longer available)
- **SAM3** adapter removed — it was never registered and therefore unreachable; it will
  return once implemented properly

## Installation

```bash
pip install --upgrade mozo
mozo start
```

## Links

- PyPI: https://pypi.org/project/mozo/0.4.0/
- Documentation: https://github.com/datamarkin/mozo
- Issue Tracker: https://github.com/datamarkin/mozo/issues
