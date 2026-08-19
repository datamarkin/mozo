# Provenance

This package is a deployment-only extraction of
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

| | |
|---|---|
| Upstream repository | `https://github.com/DepthAnything/Depth-Anything-V2` |
| Upstream commit | `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf` |
| Extracted | 2026-08-19 |
| Verified against | `torch` 2.11.0, `torchvision` 0.26.0, Python 3.10 |

Record the commit before re-syncing: without it, diffing a later upstream against this
copy means guessing which revision it started from.

## What was taken

Verbatim, byte for byte. Upstream's imports are already relative, so nothing needed rewriting:

```
dinov2.py
dinov2_layers/{__init__,attention,block,drop_path,layer_scale,mlp,patch_embed,swiglu_ffn}.py
util/{blocks,transform}.py
```

Modified:

- `dpt.py` — two changes. The metric-depth head is merged in behind a `max_depth` argument:
  `None` keeps upstream's ReLU output, a float switches to the sigmoid-and-scale head. Upstream
  ships that difference as a **second copy of the whole package** under `metric_depth/`, which
  differs from the first by these six lines and nothing else — `dinov2.py`, `dinov2_layers/`
  and `util/` are identical between the two trees. The second change removes `infer_image` and
  `image2tensor`, which moved to `predictor.py`; that also removes this file's `cv2` and
  `torchvision.transforms` imports.

Added:

- `util/__init__.py` — empty. Upstream relies on `util/` being an implicit namespace package;
  setuptools' package discovery does not, and would leave `blocks.py` and `transform.py` out
  of the wheel.

Written for this package:

- `config.py` — the nine released variants as frozen dataclasses. Upstream repeats the same
  four-row `model_configs` dict in seven scripts and stores `max_depth` in none of them.
- `predictor.py` — pre-processing, forward, post-processing. Step for step upstream's
  `infer_image`, including the `cv2.INTER_CUBIC` resize, the aspect-preserving multiple-of-14
  sizing, and the bilinear resize back to the input resolution.

## Why `predictor.py` exists at all

Upstream's `image2tensor` probes the host for a device and moves the tensor there itself:

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
```

That makes the model's placement unobservable and a same-device comparison impossible to set
up, which is precisely what the verification below depends on. The device is a parameter here.
Nothing else about the pipeline changed.

## Aspect ratio is preserved, so the input shape varies

The shorter side is resized to `input_size` (518 by default) and the longer side follows,
each rounded to a multiple of 14. A 1920x1281 photograph is fed as 770x518, not 518x518. This
is upstream's behaviour and what its published numbers were measured with, so it is this
package's behaviour too.

It has a consequence worth knowing before exporting anything: `interpolate_pos_encoding` in
`dinov2.py` returns early when the input is square at the trained resolution, so a traced graph
bakes in whichever resolution it was traced at. `torch.onnx.export` accepts `dynamic_axes` for
this model and produces a graph that fails at every shape but the traced one. Any published
"dynamic" Depth Anything V2 ONNX has that property whether or not it says so.

## What was deliberately left behind

Training and its datasets (`metric_depth/{train.py,dataset/,util/}`), the Gradio app, the
`run.py` / `run_video.py` / `depth_to_pointcloud.py` scripts, and the duplicate
`metric_depth/depth_anything_v2/` tree.

`vitg` is not extracted: upstream lists it as "coming soon" and has published no checkpoint.

The `xformers` fast path in `dinov2_layers/attention.py` is kept verbatim but never taken —
this package does not depend on `xformers`, so `MemEffAttention` falls through to the plain
attention its own base class defines. Keeping the file byte-identical means a future
re-extraction has nothing to re-apply; `__init__.py` silences the import-time warning instead.

## Verification

`tools/verify/depth_anything_v2.py` reproduces all of the following, against a checkout of the
commit above, with both sides pinned to the same device:

- **Standalone** — no import outside stdlib, `torch`, `torchvision`, `numpy` and `cv2`, and no
  absolute self-import. `cv2` is not incidental: upstream resizes with `INTER_CUBIC`, and
  torchvision's bicubic is a different resample, so dropping it would change every depth map.
- **Structural** — state-dict keys and shapes identical to upstream for all nine variants.
- **Forward pass** — same weights and input, `max|delta| = 0.0`. Bitwise identical, not close.
- **End to end** — real photographs through `Predictor.predict` against upstream's
  `infer_image`, `max|delta| = 0.0`.

As of the extraction date, all nine variants pass all four, on MPS, over ten of upstream's own
`assets/examples` photographs fed at their natural aspect ratio (518x784 to 518x798):

| variant | encoder | unit | state dict | forward | end to end | ms | fps |
|---|---|---|---|---|---|---|---|
| small | vits | relative | 239 identical | 0 | 0 | 70.5 | 14.2 |
| base | vitb | relative | 239 identical | 0 | 0 | 146.9 | 6.8 |
| large | vitl | relative | 407 identical | 0 | 0 | 405.6 | 2.5 |
| indoor-small | vits | metres | 239 identical | 0 | 0 | 72.5 | 13.8 |
| indoor-base | vitb | metres | 239 identical | 0 | 0 | 151.5 | 6.6 |
| indoor-large | vitl | metres | 407 identical | 0 | 0 | 407.3 | 2.5 |
| outdoor-small | vits | metres | 239 identical | 0 | 0 | 74.3 | 13.5 |
| outdoor-base | vitb | metres | 239 identical | 0 | 0 | 145.3 | 6.9 |
| outdoor-large | vitl | metres | 407 identical | 0 | 0 | 389.4 | 2.6 |

Latency is the median over three passes after twelve warm-up passes, and covers the whole
pipeline -- cv2 resize in, model, resize out -- not the forward pass alone.

Building the baseline for the six metric variants means importing upstream's *other*
`depth_anything_v2` package, the one under `metric_depth/`, because only that copy's
`DepthAnythingV2` accepts `max_depth`. The verify script loads whichever of the two same-named
trees matches the variant under test. Needing both copies to check one model is the clearest
argument for the six-line merge that replaced them.

Comparisons must pin both sides to the same device. An MPS-vs-CPU comparison measures backend
kernels rather than the two implementations, and shows up as a spurious drift that is easy to
mistake for a real one.

The baseline must not be older than the extraction. An outdated upstream will argue confidently
for behaviour upstream itself has since abandoned, and it is convincing while it does so — the
verify script refuses to run against a different commit unless told to.
