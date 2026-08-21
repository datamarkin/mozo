# edgetam_deploy

Deployment-only EdgeTAM promptable segmentation for single images. Apache-2.0 code *and* weights,
extracted from `facebookresearch/EdgeTAM` and reduced to the image path, with no `hydra`,
`omegaconf`, `iopath`, `torchvision` or `timm` dependency.

EdgeTAM is SAM 2 with a RepViT-M1 trunk instead of Hiera, distilled to run on a phone. Its image
path is SAM 2's, so everything below reads the same as `sam2_deploy` — the difference is size:
a 9.1 M-parameter image path against SAM 2 tiny's 38.9 M.

```python
from mozo.vendors.edgetam_deploy import Segmenter
from mozo.image import load_image

segmenter = Segmenter("edgetam.pt")                    # device="cpu"
image = load_image("photo.jpg")

found = segmenter.predict(image, points=[[820, 640]], labels=[1])
found = segmenter.predict(image, boxes=[40, 60, 300, 480])

found.masks     # (b, c, h, w) bool, in the source image's pixels
found.scores    # (b, c) the model's predicted IoU for each mask
found.logits    # (b, c, 256, 256) low-res logits
```

## Prompting

Every prompt is a set of points. A click is a point with a label; a box is its two corners
carrying reserved labels, because EdgeTAM has no separate box input.

| Label | Meaning |
|---|---|
| `1` | include this — the thing you want is here |
| `0` | exclude this — you got too much, cut here |
| `2` | a box's top-left corner |
| `3` | a box's bottom-right corner |

You pass `1` and `0` yourself; `2` and `3` are written for you when you pass `boxes`.

Three things decide whether you get the mask you meant:

- **`labels` is required with `points` and has no default.** A missing label is not an omission
  this package can fill in — guessing between include and exclude returns a confident mask of the
  wrong thing, so it raises instead.
- **Order does *not* matter, despite looking like it should.** `predict` places box corners
  before clicks because upstream's image predictor does, and matching it is what keeps the
  parity gate exact — but each token's learned embedding is chosen by its **label**, not by its
  position, and the sparse tokens are only read through attention. Putting the click first
  instead moves the logits by 3.1e-05: float summation order, not a different mask.
- **Negative points need something to subtract from.** A prompt of only `0` labels does not mean
  "everything else".

`multimask_output=True` (the default) returns three candidate masks with their predicted IoU.
That is the right setting for a single click, which is genuinely ambiguous about whether you
meant the handle, the door, or the car. Take the highest-scoring one, or show all three. With a
box, or several points, the prompt is usually unambiguous and `multimask_output=False` is
tighter — and it is the only setting that reaches the decoder's stability fallback, which swaps
an unstable single mask for the best of the three candidates.

To refine, pass one of the previous call's `logits` back as `mask_input` with an extra click —
one channel, not all three, so choose the candidate first:
`found.logits[:, found.scores[0].argmax()]`.

## The encode/decode split

EdgeTAM's cost is mostly the image encoder, and the encoder depends only on the image. So
`predict` caches encoder output keyed on pixel content, and a second prompt on the same image
pays only for the decoder. On this machine, at 1281×1920 on CPU, that is 286 ms for the first
prompt and 33 ms for every one after it.

The cache holds five images, keyed on a sha256 of the pixels rather than on a filename — the same
photograph arriving twice over HTTP is two different arrays and should still be one encode.

## What this is not

The video tracker. EdgeTAM's paper is about tracking, and its contribution — a 2-D Spatial
Perceiver that compresses the memory bank — is not in this package, along with memory attention,
the memory encoder and object pointers. 149 of the checkpoint's 982 tensors are dropped at load.
See `PROVENANCE.md`.
