# ben2_deploy

Deployment-only BEN2 for single images. MIT code, MIT weights, extracted from
`PramaLLC/BEN2`'s single 1,368-line `BEN2.py`, with no `timm` and no `einops` dependency at
runtime.

> **Verified bit-identical to the published model** at every stage — the preprocessed tensor, the
> raw matte, the postprocessed alpha, and both documented end-to-end paths — on CPU in float32.
> `PROVENANCE.md` says where the guarantee stops and why it has to be conditional.

## What BEN2 is for

Every other family in mozo answers with a *decision*: this box, this class, this character, this
mask. BEN2 answers with an **opacity** — a per-pixel number saying how much of this pixel is
foreground. That is the number a compositor needs and the one a segmenter cannot give. A binary
mask cuts hair off; a matte keeps it.

```python
from ben2_deploy import Predictor

predictor = Predictor.from_pretrained("torch-fp32.pth")
alpha  = predictor.matte(rgb)                 # (H, W)    uint8
cutout = predictor.cutout(rgb, refine=True)   # (H, W, 4) uint8 RGBA
```

## One image is five forward passes

This is the shape of the model, not a batching choice:

```
input 1024x1024
  ├─ 4 quadrants  512x512  ─┐
  └─ 1 global     512x512  ─┴─→ Swin-B backbone, batch 5 → decoder splits [4, 1] at every rung
```

The decoder's two halves mean different things — the global view produces a saliency gate that
multiplies the quadrants, and the refreshed quadrants are added back into the global. An outer
batch of *N* sends `5N` images through the backbone at once, so *N* is a number the gate pins
rather than a free parameter.

The resolution is frozen by construction. `INPUT` is 1024 in `config.py` and changing it changes
the pooling grid in every MCLM and MCRM block; the published weights would stop meaning anything.

## The alpha is not a probability, by default

`postprocess` reproduces upstream's per-image min-max stretch, so the most-foreground pixel in
*this* image becomes 255 and the least becomes 0:

```python
alpha = predictor.matte(rgb)                   # contrast-stretched, upstream's default
alpha = predictor.matte(rgb, stretch=False)    # the calibrated sigmoid, 0.5 means something
```

Compare the stretched alpha within an image, never across two. An image whose most confident
pixel scored 0.6 still comes back with pixels at 255. Use `stretch=False` when thresholding, or
when combining this model with another.

Upstream divides by `max - min` unguarded, so a frame it reads as uniform produces `nan` cast to
uint8. Here the stretch is skipped below `ALPHA_EPSILON` and the constant is returned scaled.

## `refine` changes colour, not opacity

`cutout(refine=True)` runs Photoroom's blur-fusion foreground estimator before compositing, so a
soft edge does not carry a fringe of the background it came from. It costs two box blurs at full
resolution, which is why it is off by default.

It also takes a **different alpha**, and that is upstream's doing rather than a choice made here:
the unrefined path bilinearly resizes and then stretches, while the refined path casts the raw
1024×1024 sigmoid to uint8 and resizes *that* with PIL. Both are reproduced exactly, which is why
`stretch` has no effect on the refined path — there is nowhere in upstream's refined path for it
to apply.

## Dependencies

`torch`, `numpy`, `cv2`, `Pillow`. Nothing else.

Pillow is load-bearing: upstream resizes with PIL's LANCZOS, whose filter support scales with the
downsampling factor, and neither `cv2.resize` nor `F.interpolate(antialias=True)` reproduces it.
`cv2` is load-bearing for the same kind of reason — the foreground estimator is two `cv2.blur`
passes and a box blur is not a Gaussian.
