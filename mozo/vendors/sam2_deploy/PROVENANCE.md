# Provenance

This package is a deployment-only extraction of [SAM 2](https://github.com/facebookresearch/sam2),
reduced to single-image promptable segmentation.

| | |
|---|---|
| Upstream repository | `https://github.com/facebookresearch/sam2` |
| Upstream commit | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| Extracted | 2026-08-20 |
| Verified against | `torch` 2.11.0, `torchvision` 0.26.0, Python 3.10, on CPU |

Record the commit before re-syncing: without it, diffing a later upstream against this copy
means guessing which revision it started from.

## What was taken

Verbatim, apart from rewriting imports to relative form:

```
backbones/{hieradet,image_encoder,utils}.py
sam/{mask_decoder,prompt_encoder,transformer}.py
position_encoding.py
```

Modified:

- `sam2_utils.py` — truncated after `LayerNorm2d`. Everything below it (`sample_box_points`,
  `sample_random_points_from_errors`, `sample_one_point_from_error_center`, `get_next_point`)
  samples training clicks, and it was the only thing importing `sam2.utils.misc.mask_to_box`.
  The image path uses three names from this file: `DropPath`, `MLP`, `LayerNorm2d`.
- `backbones/hieradet.py` — `iopath`'s `g_pathmgr.open` replaced with the builtin `open`. It
  appeared once, in a `weights_path` branch that loads a pretrained trunk on its own; mozo loads
  the whole checkpoint instead, so the branch never runs. Dropping it removes `iopath` from the
  install.

Written for this package:

- `config.py` — the four variants' geometry as frozen dataclasses, replacing the Hydra YAML.
  Values were read out of `sam2/configs/sam2.1/sam2.1_hiera_*.yaml`. This is what removes
  `hydra-core` and `omegaconf`.
- `network.py` — the image-mode model, replacing `SAM2Base`, split into `encode` and `decode`.
- `image.py` — preprocessing, prompt scaling, and mask resizing, replacing
  `sam2/utils/transforms.py`. This is what removes `torchvision`.
- `predictor.py` — checkpoint loading, prompt assembly, and the encoder-output cache.

## What was deliberately left behind

The whole video tracker: `memory_attention.py`, `memory_encoder.py`, `sam2_video_predictor.py`
and its legacy twin, and on `SAM2Base` the memory bank, object pointers, temporal position
encoding and frame bookkeeping. Also the training code, the evaluation harness,
`automatic_mask_generator.py`, the CUDA `connected_components` extension used for hole filling,
and the Hydra build path.

Dropping these removes `hydra-core`, `omegaconf`, `iopath` and `torchvision` from the install,
and leaves 90.7 percent of the checkpoint's parameters in use — the video weights are 7.5 M of
80.9 M for base_plus, and they are filtered out at load rather than being carried dead.

One tensor that *looks* like tracker state is kept. `no_mem_embed` is trained to mean "there is
no memory to attend to", which is exactly the situation a single image is in, and upstream's
image predictor adds it to the lowest-resolution feature map. Leaving it out costs 4.5e-02 on the
embedding and a few hundred pixels per mask — close enough to look correct, which is why the
parity suite pins it.

## Measured parity

Against upstream `SAM2ImagePredictor` built through Hydra from the same checkpoint, same process,
same torch, on CPU. Eight prompt configurations per variant: a single positive point, a box, a
box combined with a point, positive-and-negative pairs, and five random multi-point sets.

| Variant | Encoder features | IoU predictions | Low-res logits | Mask pixels differing |
|---|---|---|---|---|
| tiny | 0.0 | 0.0 | 0.0 | 0 / 2,459,520 |
| small | 0.0 | 0.0 | 0.0 | 0 / 2,459,520 |
| base_plus | 0.0 | 0.0 | 0.0 | 0 / 2,459,520 |
| large | 0.0 | 0.0 | 0.0 | 0 / 2,459,520 |

Bit-identical, not merely within tolerance. Two things had to be right for that, and neither was
obvious:

- **The resize antialiases.** Upstream resizes with `torchvision.transforms.Resize`, which on a
  tensor defaults to `antialias=True`. `cv2.INTER_LINEAR` does not, and mozo decodes with cv2
  everywhere else. `image.py` calls the same `F.interpolate` torchvision delegates to.
- **Prompt coordinates are normalised before they are scaled.** Upstream divides by the image
  width and then multiplies by 1024. Multiplying by the combined ratio instead differs in the
  last bits of a float, which was enough to move one box corner across a pixel boundary and put
  one mask pixel on the wrong side of the threshold.

## Runtimes

`torch-fp32` and `onnx-fp32` are exported here. **CoreML is not: Apple publishes it.**

`apple/coreml-sam2.1-{tiny,small,large}` are Apache-2.0 converted packages, and mozo
redistributes them rather than producing a second set of the same thing. They differ from the
other artifacts in three ways worth knowing before using them: they are **fp16**, they split
**three** ways rather than two (image encoder, prompt encoder, mask decoder), and their encoder
takes a **CoreML image input** with normalisation baked into the graph rather than a normalised
tensor. Hence the `coreml-fp16-*` names. `base_plus` has no CoreML at all, because Apple did not
publish one.

The fp16 costs accuracy against the fp32 reference, measured on the fixture with a box prompt:

| Variant | Mask pixels differing from torch-fp32 |
|---|---|
| tiny | 49,064 / 2,459,520 (1.99%) |
| small | 36,047 / 2,459,520 (1.47%) |
| large | 19,955 / 2,459,520 (0.81%) |

Some of that is not fp16 rounding. Apple's encoder scales the image by the single scalar
`0.01735207`, which is `1 / (255 * 0.226)` -- the *mean* of the three ImageNet standard
deviations -- where the correct scaling is per channel, `[0.017125, 0.017507, 0.017429]`. CoreML's
image input accepts a scalar scale with a per-channel bias and nothing richer, so the conversion
could not express it. The bias is right; only the scale is averaged. Pre-compensating the input
recovers little of the gap, so this is a contributing error rather than the whole of it.

Speed, tiny variant at 1024x1024 on this machine:

| Runtime | Encode | Decode | Agreement with torch-fp32 CPU |
|---|---|---|---|
| torch fp32, CPU | 463 ms | 30 ms | — |
| torch fp32, MPS | 109 ms | — | **wrong: max abs 9.7e-01** |
| onnx fp32, CPU | 940 ms | 20 ms | 1.2e-05 |
| CoreML fp16 (Apple's), all units | 109 ms | 46 ms | see the table above |

The two halves have different best runtimes, and because they are separate artifacts they can be
mixed: CoreML encodes fastest and ONNX decodes fastest. ONNX is slower than torch on the encoder
and faster on the decoder, so neither runtime wins outright.

**MPS is refused for this family.** Not because it is slow or unsupported — it runs, and it is
the fastest torch path — but because on `torch` 2.11.0 it returns wrong numbers, and only under
`torch.no_grad()` or `torch.inference_mode()`, which is exactly how inference runs. Under
`enable_grad` the same code on the same device agrees to 2.6e-05. The divergence is inside the
Hiera trunk; the neck and the decoder are clean. A silent 0.37 percent of mask pixels move, so
nothing raises and nothing looks broken.

Both halves export separately, which is the point of the `encode`/`decode` split: the encoder is
a fixed-shape graph and the decoder takes a variable number of prompt points, as a dynamic axis in
ONNX. The decoder graph returns all four mask tokens and the slice that picks one or three is done
in Python, so a single graph serves both `multimask_output` settings: **token 0 is the
single-mask output and tokens 1 to 3 are the multimask candidates**, so a caller driving
`onnx-fp32-decoder.onnx` wants `low_res_masks[:, 1:]` to match `multimask_output=True` and
`[:, :1]` to match `False`. That convention is recorded here because it is not in the graph.
Apple's decoder does not offer the choice -- it emits three masks only.

Converting SAM 2 to CoreML ourselves does work, if a reason to ever appears: Hiera interpolates
its position embedding with `bicubic`, which coremltools cannot convert, but at a fixed 1024 input
that interpolation is a constant and folds away exactly. That is the only obstacle.
