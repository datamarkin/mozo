# rfdetr_deploy

Inference-only RF-DETR. Detection, instance segmentation, and keypoints, in one package that
depends on **`torch`, `torchvision`, `numpy`, and `pillow`** and nothing else.

```python
from rfdetr_deploy import Predictor

predictor = Predictor.from_pretrained("rfdetr-small")
results = predictor.predict("photo.jpg", threshold=0.5)

boxes = results[0]["boxes"]    # (N, 4) xyxy, source-image pixels
scores = results[0]["scores"]  # (N,)
labels = results[0]["labels"]  # (N,)
```

Weights download on first use into `~/.cache/rfdetr-deploy` (override with
`RFDETR_DEPLOY_HOME`) and are verified against a pinned MD5.

## Models

| Name | Task | Resolution | Notes |
|---|---|---|---|
| `rfdetr-nano` | detection | 384 | |
| `rfdetr-small` | detection | 512 | |
| `rfdetr-medium` | detection | 576 | |
| `rfdetr-large` | detection | 704 | |
| `rfdetr-seg-nano` | segmentation | 312 | adds `masks` |
| `rfdetr-seg-small` | segmentation | 384 | adds `masks` |
| `rfdetr-seg-medium` | segmentation | 432 | adds `masks` |
| `rfdetr-seg-large` | segmentation | 504 | adds `masks` |
| `rfdetr-keypoint-preview` | keypoints | 576 | adds `keypoints`, `keypoint_precision_cholesky` |

XLarge / 2XLarge are not included — they are PML 1.0 components of the separate `rfdetr_plus`
package, not Apache 2.0.

## Output

`predict()` returns one dict per input image, straight from the post-processor. There is no
result class: the wrapper type belongs to whatever consumes this package.

| Key | Shape | Meaning |
|---|---|---|
| `scores` | `(N,)` | Detection confidence |
| `labels` | `(N,)` | Class ids |
| `boxes` | `(N, 4)` | `xyxy`, source-image pixels |
| `masks` | `(N, 1, H, W)` | Segmentation variants only |
| `keypoints` | `(N, K, 3)` | Keypoint variants only, as `(x, y, confidence)` |
| `keypoint_precision_cholesky` | `(N, K, 3)` | Per-keypoint precision, when emitted |

Class ids follow the checkpoint. The published COCO detection checkpoints emit **sparse COCO
category ids (1-90, with gaps)**, not indices into a dense 80-name list — map by category id.
Fine-tuned checkpoints emit contiguous 0-based ids.

## Inputs

File paths, PIL images, HWC `numpy` arrays, and CHW tensors already scaled to `[0, 1]`. Any PIL
mode converts to RGB automatically.

## Notes for anyone modifying this

- **Resizing must stay `antialias=False`.** It matches the antialias-free bilinear resize the
  models were trained under. Turning antialias on costs accuracy silently — nothing raises.
- **Input side must be divisible by `patch_size * num_windows`.** `Predictor.from_pretrained`
  checks this when given an explicit `resolution`.
- **Loading refuses to leave a parameter uninitialized.** A checkpoint that does not fully
  populate the model raises rather than producing a randomly initialized head.
- **`num_classes` and the keypoint schema come from the checkpoint**, so fine-tuned weights load
  without restating their shape.

See `PROVENANCE.md` for the upstream commit, exactly what was changed, and the parity results.
See `NOTICE` for attribution.
