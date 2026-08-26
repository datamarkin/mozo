# moebius_deploy

Moebius, extracted for deployment: an image and a mask in, the masked thing gone.

A 226M-parameter latent diffusion inpainter that matches FLUX.1-Fill-dev (11.9B) across six
benchmarks. This package is its forward path and nothing else — no training code, no distillation
teacher, no dependency beyond torch, numpy, OpenCV and Pillow.

See `PROVENANCE.md` for what this derives from, what it deliberately leaves behind, and the parity
numbers. This file is about driving it.

## Two things to know before you start

**It runs at 512×512 and cannot run at anything else.** That is a property of the published
weights, not a setting — the cross-attention's positional table is stored with a row per latent
cell. `Predictor.predict` resizes for you and composites back at your resolution;
`Predictor.sample` refuses anything else outright.

**The answer is a sample, not an estimate.** There is no score, no confidence and no "best" result.
Change the seed and you get a different, equally valid removal. This is the only family in mozo
where re-running the same call with a different seed is a normal thing to do.

## Using it

```python
from mozo.vendors.moebius_deploy import Predictor

model = Predictor("torch-fp32-unet.pth", "torch-fp32-vae.pth", "general")
clean = model.predict(frame, mask, seed=0)
```

`frame` is `(H, W, 3)` uint8 RGB; `mask` is `(H, W)` and may be bool, uint8 or float — thresholded
by its dtype *and* its values, so a uint8 mask holding `{0, 1}` works as well as one holding
`{0, 255}`. The result is the same shape as `frame`, and **every pixel the feathered seam does not
reach is byte-identical to the input** — roughly 8 px beyond the selection at the default
`feather=3`, exactly the mask's edge at `feather=0`.

An empty mask returns the frame unchanged without running the model. Several disjoint regions are
removed in one pass over their union.

### The knobs that matter

| argument | default | what it does |
|---|---|---|
| `seed` | `0` | Which sample to draw. Not a quality setting — just a different valid answer. |
| `steps` | `20` | **Runs nineteen.** Upstream trims one; mozo reproduces that rather than quietly disagreeing about what twenty means. |
| `guidance` | `2.0` | Upstream's README value. Its pipeline says 4.5 and its argparse says 2.5. |
| `dilate` | `0` | Grow the mask first. Usually what you want — see below. |
| `feather` | `3` | Blur radius on the mask before compositing. |

### If the thing is still faintly there

Two causes, and they have different fixes.

**The mask stopped at the object's edge.** An object's shadow and its antialiased rim are outside
most segmentation masks, and removing the object without them leaves an outline. Raise `dilate`.

**The feather ate into a small mask.** A radius-3 Gaussian pulls the blend below full strength
several pixels *inside* the selection, so on a mask only a few pixels across you get a mix of
original and generated everywhere, including the centre. The fix is a larger `dilate`, not a
smaller `feather` — the feather is what stops the seam looking cut out.

## Two variants

`general` is upstream's `pretrained`, for arbitrary photographs. `places2` is tuned on scenes and
backgrounds. They are the same architecture at the same shape; only the training differs.

Upstream also publishes two face-specific checkpoints, which mozo does not carry — `PROVENANCE.md`
says why.

## What is in here

| file | |
|---|---|
| `config.py` | The geometry, frozen. Every number read off the published tensors. |
| `vae.py` | The autoencoder — pixels to latents and back. |
| `attention.py` | The λ layers, the MixFFN and the depthwise convolution. The novel part. |
| `network.py` | The denoiser: nine latent channels in, four channels of predicted noise out. |
| `scheduler.py` | DDIM, at the one configuration these weights were trained under. |
| `image.py` | Preprocessing, and the composite that is the family's actual contract. |
| `predictor.py` | The seam. The only file here with an opinion about *how* to run the model. |

## Exporting it

`fold_positional()` rewrites the layer's `Conv3d` — a depth-1 kernel that is algebraically a 2-D
convolution — into the `Conv2d` that ExecuTorch and CoreML will accept. It is **not** used on the
torch path: the two forms sum in a different order and land 2.1e-06 apart, which is a divergence
worth taking for a mobile graph and not worth taking for nothing.
