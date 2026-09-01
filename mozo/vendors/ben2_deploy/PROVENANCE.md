# Provenance

This package is a deployment-only extraction of BEN2's **image** path.

Upstream commit: `2c99a5da477b5523585bfa5c893888a6e818a8f6` (2026-01-02), from
`https://github.com/PramaLLC/BEN2`, file `BEN2.py` (1,368 lines).

Weights revision: `e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f` on
`https://huggingface.co/PramaLLC/BEN2` (2025-12-31).

## The reference, and what mozo guarantees

**The reference is the GitHub repository, not the Hugging Face copy.** Both ship a `BEN2.py` and
they are not the same file. The HF copy imports `DropPath` from `timm.models.layers` (the
deprecated path) and defines `PositionEmbeddingSine` **twice**; the GitHub file at the commit
above imports from `timm.layers` and defines it once. The most recent upstream commit is titled
"Remove PositionEmbeddingSine class from BEN2.py" — the duplicate was deleted on GitHub and not
on Hugging Face. Forty diff lines, no arithmetic among them. GitHub is pinned because it is the
project repository and carries the `LICENSE`.

Upstream claims nothing numerical about anything: no port, no second implementation, no
conversion script. The checkpoint reproduces this code and no other, so the chain is one link
long.

**The guarantee is conditional, and the condition is a device check.** `BEN_Base.forward` is
decorated `@torch.autocast(device_type="cuda", dtype=torch.float16)` and `inference()` chooses
its normalisation with

```python
if torch.cuda.is_available():
    img_tensor = img_transform(image)      # ConvertImageDtype(torch.float16)
else:
    img_tensor = img_transform32(image)    # ConvertImageDtype(torch.float32)
```

so the published model is **two models**, selected by a global property of the machine rather
than by an argument. mozo takes the dtype as a parameter and never probes the host.

The guarantee reads: *on CPU, in float32, at batch 1, this package reproduces upstream bit for
bit at every stage.* The fp16 path is a different computation and is not claimed.

## What the published files actually contain

| file | bytes | contents |
|---|---|---|
| `model.safetensors` | 380,577,976 | 535 tensors, 94.63M parameters, float32 and int64 |
| `BEN2_Base.pth` | 1,134,584,206 | a **training** checkpoint — see below |
| `BEN2_Base.onnx` | 222,932,053 | a float16 graph — see "Export" |

`BEN2_Base.pth` is not a weights file. It is `torch.save` of a training state at **epoch 5**:

```
epoch                5
model_state_dict     535 entries, 380.5 MB
optimizer_state_dict 753.1 MB of Adam moments over 511 parameters
scaler_state_dict    5 entries
loss                 9.64396223146713
metrics              MAE 0.0399  DICE 0.8736  IOU 0.8236  BER 0.0588  ACC 0.9646  FM 0.8501
learning_rate        3.999999998087702e-06
```

380.5 + 753.1 MB accounts for the file. Its `model_state_dict` and `model.safetensors` were
compared tensor by tensor: 535 keys each, no key in one and not the other, no shape or dtype
mismatch, and `torch.equal` on all 535. mozo publishes the safetensors, repacked.

## Licence

**Code: MIT.** `LICENSE` at the repository root, 1,066 bytes, "Copyright (c) 2025 Prama LLC".
GitHub's API reports `spdx_id: MIT`. This is the strong case: stated in the project, in a file,
by the copyright holder.

**Weights: MIT.** `license:mit` on the model card, `gated: False`. Also the strong case: stated
on the checkpoint itself.

**Where the chain stops.** BEN2 was trained on DIS5K plus a 22K proprietary set. DIS5K's own
terms restrict commercial use pending an agreement with its authors. mozo takes the publisher's
MIT grant on the weights as stated and does not track the data behind them — that obligation was
Prama's, discharged before publication. This paragraph is the end of the chain, recorded as §1.2
of `plans/vendoring.md` requires rather than left as silence.

The model card points commercial users at a sales address for a different, unpublished model.
That is an offer of another artifact, not a restriction on this one.

## What was taken

| from `BEN2.py` | what | lands in |
|---|---|---|
| lines 38–616 | Swin-B backbone, window attention, patch merging, patch embedding | `swin.py` |
| lines 618–842 | MCLM, MCRM, `PositionEmbeddingSine`, cbr/cbg, the patch plumbing | `blocks.py` |
| lines 844–967 | `BEN_Base` and its forward | `network.py` |
| lines 1161–1368 | the square resize, the normalise, `postprocess_image`, the foreground estimator | `image.py` |
| lines 969–1041 | `loadcheckpoints`, the useful half of `inference` | `predictor.py` |

## What was deliberately left behind

**The entire video path** — `segment_video`, `pil_images_to_mp4`, `pil_images_to_webm_alpha` and
`add_audio_to_video`, about 250 lines. Three of the four shell out to `ffmpeg` through
`subprocess`. A library that invokes a binary it never checked for is not something to ship in a
wheel; mozo's video story is `pixelflow.VideoReader` over the image path.

**`timm` and `einops`, which upstream requires and inference does not.** `DropPath` is the
identity at eval and is reimplemented in eight lines; `trunc_normal_` is initialisation that the
strict load overwrites before a forward runs; `to_2tuple` is one line. Every `einops.rearrange`
had a fixed pattern with literal group sizes, so each became a `view`/`permute`/`reshape`.
`tools/verify/ben2.py` checks all nine rewrites against `einops` itself, and the round trip
`patches2image(image2patches(x)) == x`.

**Weights that are built and never run.** `sideout1`..`sideout5` are deep-supervision heads from
training: five `Conv2d(128, 1, 3, padding=1)`, 5,765 parameters between them, never mentioned in
`forward`. They are built because a strict load needs somewhere to put them. The four
`token_attention_map` tensors the decoder rungs return are likewise computed and discarded at
every call site.

## Where this diverges from the obvious reading, and why

Each of these is a place where writing the sensible thing produces confident, wrong output.

| divergence | cost of the obvious reading |
|---|---|
| `rgb_loader_refiner` is defined **twice**, at lines 1161 and 1350, and Python takes the second. This package reproduces the second. | The first converts to RGB *after* the resize instead of before — different pixels for any image with an alpha channel, which is most of a background remover's input. |
| The second definition computes `ImageOps.exif_transpose(original_image)` into a local that the next line overwrites. **The EXIF correction is dead**, and the call in `inference` is commented out besides. | "Fixing" it rotates every matte for every photo carrying an orientation tag, away from what the weights were exercised with. |
| `h, w = original_image.size` puts the **width** in `h`, and `postprocess_image(res, im_size=[w, h])` therefore passes `(H, W)` — which is what `F.interpolate` wants. Two swapped names cancelling. | Renaming either one transposes the matte on every non-square image. Upstream's own `onnx_run.py` has exactly this bug; see below. |
| The 1024×1024 resize does not preserve aspect ratio. | Letterboxing moves every pixel: the weights were trained on squashed images. |
| `postprocess_image` **min-max normalises** the matte per image. | The returned alpha is a contrast stretch of a probability, not a probability. Treating it as calibrated is wrong across images, and the undefended denominator is `nan` on a flat matte. mozo guards it and offers `stretch=False`; see `image.py`. |
| `MCLM` derives its pooling target from the **quadrant** height while pooling the reassembled `2h × 2w` image, so ratio 1 halves rather than preserving. | Deriving the target from the tensor being pooled is the natural reading and pools the wrong grid at every ratio. |
| Every attention call is `self.attention[i](q, k, v)[0]`, leaving `need_weights=True`. | That default takes torch's unfused branch, where the query is scaled before the matmul and `scaled_dot_product_attention` is never reached. Rewriting to SDPA is a real numerical change that no attention-implementation pin governs. |
| `rescale_to` defaults to `nearest` and `resize_as` to `bilinear`, and both are used on the same path in the head. | Unifying them is a filter change on half the upsampling path. |

## Upstream behaviour that is reproduced, not fixed

- The dead EXIF transpose, above.
- Five unused deep-supervision heads, above.
- `if final_input == None` compares a tensor to `None` with `==`; it works only because the first
  iteration compares `None` to `None`. Rewritten to `is None`, which cannot change a number.
- `for m in self.modules(): if isinstance(m, (nn.GELU, nn.Dropout)): m.inplace = True` — `nn.GELU`
  has no such argument and ignores the attribute, and `Dropout(inplace=)` is inert at eval. Dead
  code that reads as load-bearing. Dropped.

## What is not carried, because a library may not do it

- `set_random_seed(9)` runs **at import** (line 31) and again inside `inference()`. It writes
  `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed`,
  `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic` and
  `torch.backends.cudnn.benchmark`. Nothing on the inference path is stochastic — the model is in
  `eval()`, so every `Dropout` is the identity — and the gate proves the removal changes no
  number rather than assuming it.
- `torch.set_float32_matmul_precision('highest')` at import. It happens to be torch's own default
  today, so it is a no-op that pins a default against a future change. Pinning global torch state
  is still not a library's to do.
- `inference()` calls `original_image.putalpha(mask)`, writing into the caller's own `PIL.Image`.
  This package returns new arrays and mutates nothing it was given.
- The two decorators on `forward`. `@torch.inference_mode()` returns tensors that cannot be
  saved, re-entered or exported; `@torch.autocast(device_type="cuda")` silently halves precision
  on one class of machine. Both moved to the predictor, where they are visible and optional.

## Two changes made to enable an export, both proven to move nothing

Neither is a divergence from upstream's arithmetic; both are the tracer being unable to see a
constant that is one. `tools/verify/ben2.py` gates the result, and eager parity was re-measured
after each: `torch.equal`, `max|delta| = 0.0`, on every fixture.

**`round(h / pool_ratio)` became `round(int(h) / pool_ratio)`.** Under a trace `x.size()` yields
Tensors, and `round()` on a Tensor raises `TypeError: type Tensor doesn't define __round__`. In
eager `int()` is the identity, so nothing moves.

**`F.adaptive_avg_pool2d` became `F.avg_pool2d` where the division is exact**, via `_pool` in
`blocks.py`. ONNX cannot lower adaptive pooling once the tracer has lost the input's static shape,
which happens here because `image2patches` reaches the pooled tensor through a reshape with
computed sizes: `Unsupported: ONNX export of operator adaptive_avg_pool2d, input size not
accessible`. When the input divides evenly by the target the two are the same operation — the
adaptive window `[floor(i*in/out), ceil((i+1)*in/out))` collapses to `[i*k, (i+1)*k)` — and every
ratio in this model divides evenly. Checked at all fifteen (shape, target) pairs the model uses:
`torch.equal`, zero delta at each. Upstream's operator is kept for any case where the division
would not be exact, so the substitution is never a guess.

This is §6 of `plans/vendoring.md` landing the opposite way from EasyOCR, where the only
substitution that traced was a mean over the same axis and *was* different in float, so it was
reverted.

## Export

**mozo exports an fp32 ONNX graph and does not publish it.** The result is recorded here because
§6 says a measured "no" is a finding and an unmeasured one is a gap.

The graph builds: 408.9 MB, opset 17, fixed `(1, 3, 1024, 1024)` in and `(1, 1, 1024, 1024)` out,
10 s to export once the two changes above are in place. Then, on CPU against the torch model it
came from:

| | |
|---|---|
| parity | `max abs` **4.879e-05**, MAE 6.112e-07 — **not** bit-exact |
| effect on the returned alpha | 1 grey level on 0.0146% of pixels |
| speed | **6180 ms against torch's 5455 ms — 0.88x, i.e. slower** |
| peak RSS | 10.8 GB |

Two independent reasons not to publish it, either sufficient. It is **slower**, so
`select_runtime` would never choose it and users would download a second 409 MB copy to run at
0.88x. And it does **not hold parity**, so publishing it would put two artifacts of the same model
in users' hands that disagree — the one failure the whole scheme exists to prevent. mozo's bar for
a published graph is exactness, and 4.9e-05 on the matte reaches the uint8 alpha a caller receives.

The likely cause of both is the same line: MCLM and MCRM leave `need_weights=True`, so the
attention is unfused and materialises a full 16,384 x 5,376 matrix per quadrant at the shallowest
rung. That is what makes the graph large, the memory high, and the op ordering diverge.

### CoreML: faster, and wrong in the one place a matte cannot be

`coremltools` converts this architecture directly — 30 s to trace, 15 s to convert, no custom op
registration, no deformable convolution to work around. It is also the fastest thing measured
here. It is still not published.

| | torch cpu | torch mps | CoreML (GPU) | CoreML (ALL) |
|---|---|---|---|---|
| forward | 5455 ms | 601 ms | **386 ms** | 386 ms |
| vs the CPU reference | — | 3.13e-05 | 7.85e-01 | 7.85e-01 |

**1.56x faster than torch on Metal**, and `CPU_AND_GPU` and `ALL` are identical to within a
millisecond, so the Neural Engine contributes nothing.

But the matte disagrees: `max abs` 0.785, MAE 6.7e-03, and on the returned uint8 alpha

```
differing by >  1 grey level : 7.801%
differing by >  8            : 3.329%
differing by > 32            : 1.472%
differing by >128            : 0.142%
```

**The difference is concentrated entirely on the edges.** A difference image is black across every
interior and glows along every silhouette. That is the shape of a sub-pixel shift in the alpha
ramp rather than a structural error — the matte is visually correct — and it is precisely the
wrong place for a matting model to differ. The soft edge *is* the product; a segmenter could
absorb this and a matte cannot.

So: not published, and the reason is parity rather than speed. This is the one artifact here worth
revisiting, and anyone doing so should start from these numbers. The likely lever is the same
unfused attention named above, and the first thing to try is pinning `compute_precision` per op
rather than globally.

**Upstream also publishes `BEN2_Base.onnx`, and mozo does not republish that either.** The graph was inspected
rather than trusted, and two things rule it out as a parity-holding artifact:

1. **It is float16.** 325 of its 513 initializers are `FLOAT16` and 188 are `FLOAT`, totalling
   220.8 MB — which is why the file sits between fp16's 190 MB and fp32's 380 MB. It is an export
   of the CUDA autocast path, so it cannot agree with the fp32 model to better than fp16.
   Measured against this package on CPU: MAE 0.0024, with 0.01% of pixels differing by more
   than 0.5.
2. **Its own runner script feeds it the wrong input.** `onnx_run.py` preprocesses with
   `transforms.Resize((1024, 1024))` and `ToTensor` and **no normalisation at all**, while
   `BEN2.py::inference` normalises with ImageNet statistics. They cannot both be right. Tracing
   the graph settles it: `input.1` is consumed directly by a `Conv` (the `shallow` stem), a
   `Resize` (the half-scale global) and a `Gather` (the quadrant split), with no `Sub`/`Div`
   anywhere. The graph expects an already-normalised tensor, so the runner script is wrong.
   Feeding it unnormalised input measures MAE 0.0080 against this package — 3.3× worse, with 28×
   more grossly-wrong pixels — and never raises.

`onnx_run.py` additionally transposes the matte on non-square images: it unpacks
`(w, h) = image.size` and then passes `im_size=[w, h]` to `F.interpolate`, which wants `(H, W)`.
The torch path avoids this by accident, through the swapped names recorded above.
