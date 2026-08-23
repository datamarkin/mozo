# Mozo v0.5.0: Describe It In Words

## What's New in v0.5.0

### 🎉 New Model Family

**Grounding DINO:** Open-vocabulary detection by IDEA Research
- 2 variants: `tiny` (Swin-T, 48.4 box AP COCO zero-shot) and `base` (Swin-B, 56.7)
- Both checkpoints upstream publishes; there is no third
- Apache 2.0 code *and* weights

It answers the same question OWLv2 does, through the same endpoint and the same response shape,
so the two are substitutable. Reach for it when the prompt is a *description* rather than a noun:
OWLv2 embeds each phrase independently, while Grounding DINO fuses the text into the image
features six times over and lets the decoder attend back to the words, so `"the mug on the left"`
is read as a phrase rather than a bag of words.

```python
model = mozo.get_model("grounding_dino/tiny")
found = model.predict("kitchen.jpg", ["a kettle", "the mug on the left"])
found[0].class_name        # 'a kettle' -- the phrase you asked for
```

**No `transformers` dependency**, despite the BERT text encoder: the published checkpoint carries
its own fine-tuned BERT tower, and the WordPiece vocabulary ships in the wheel. Tokenizing a
prompt needs no network.

**Torch only, and that is structural.** The image is resized rather than letterboxed, so there is
no fixed input size and no graph to export. `runtime="auto"` will not offer one.

### 📊 Model Count

**Total Models:** 54 variants across 12 model families
**Growth:** +2 variants (Grounding DINO), +1 family

### ✅ Verification

138 comparisons against the authors' own implementation, both variants, exact equality with no
tolerance. Every intermediate is hooked — preprocessing, token ids, the BERT tower, all three
Swin levels, encoder memory, and each of the six decoder layers — not just the final boxes, since
two implementations can agree on the last tensor and disagree in the middle.

### 🔧 Infrastructure

**Stated counts are now tested.** `tests/test_stated_counts.py` holds the package docstring and
the README's model counts against the manifest. The docstring had said "Seventeen published
variants across two families" for eight families longer than it was true.

---

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
