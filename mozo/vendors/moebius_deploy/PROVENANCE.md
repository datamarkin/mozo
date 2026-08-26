# Provenance

This package is deployment-only Moebius inference: a 226M-parameter latent diffusion inpainter,
plus the autoencoder it denoises inside. It is an extraction of `hustvl/Moebius` — the forward path
only — rewritten to depend on nothing but torch, numpy, OpenCV and Pillow.

| | |
|---|---|
| Upstream | https://github.com/hustvl/Moebius |
| Authors | HUST Vision Lab and vivo AI Lab |
| Paper | *Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level Performance*, ECCV 2026, arXiv:2606.19195 |
| Released | 2026-06-18, last touched 2026-06-22 |
| Code licence | Apache-2.0 |
| Weights licence | Apache-2.0 (GitHub) / MIT (HF tag) — **both stated by the authors**, see below |
| Autoencoder | `hustvl/PixelHacker`, MIT — a separate work, published separately here |
| Harvested into mozo | 2026-08-26 |
| Verified with | `torch` 2.11.0, `diffusers` 0.38.0, Python 3.10, on CPU |
| Reference version | `diffusers` 0.38.0 — upstream's own pin |
| Resolution | 512×512, and nothing else — see *One shape* |

## The reference, and what mozo guarantees

The reference is upstream's own code: `removal/v1_2/pipeline.py` (`RemovalSDXLPipeline_BatchMode`)
built by `infer/utils.py::build_pipeline`, standing on `diffusers` for the autoencoder, the DDIM
scheduler and the UNet scaffolding its model subclasses.

**Upstream claims nothing numerical about anything.** There is no port and no second
implementation, so the chain is one link long: these checkpoints reproduce this code and no other.
That is the strongest position any mozo family has started from, and it is worth saying plainly
because most of them are weaker.

**The guarantee is conditional, and the condition is the point.** Upstream's encoder is stochastic
— `latent_dist.sample()`, not `.mode()` — and its initial noise is a second draw. Both come from
the global RNG, seeded by `torch.manual_seed`. So:

> Given the same seed, drawing in the same order at the same shapes, at batch 1 on CPU, mozo
> reproduces upstream bit for bit at every stage.

Anything looser is not a claim about this extraction. In particular **batch size changes the
output** — not the rounding, the *sample*: `randn_like` on a batch of four consumes a different
slice of the stream than four calls on a batch of one.

## Where this diverges from upstream, and why

**`torch.Generator` instead of `torch.manual_seed`.** Upstream is a script and may write the
process-wide RNG; mozo is a library and may not. The generator is threaded through the encode and
the noise explicitly. This is a divergence in *mechanism* that produces identical *numbers*, and
the gate proves it rather than asserting it.

**The composite is mandatory.** Upstream's `_post_process` defaults to `paste=False`, which returns
the decoder's reconstruction of the entire frame — every pixel changed, including the ones nobody
selected. mozo always returns the caller's own array byte for byte wherever the feathered mask does
not reach — about 8 px past the selection at the default radius of 3, and exactly the mask's edge at
`feather=0`. This selects upstream's `paste=True` branch rather than inventing one, but it *is* a
selection.

**Non-512 inputs are resized rather than refused.** Upstream's `resize_image_to_multiple_of_64`
scales the short side to 512 and floors both sides to a multiple of 64 — which can hand upstream's
own model a shape it raises on (see *One shape*). mozo resizes to 512×512, runs, and composites the
result back at the caller's resolution. Detail inside the hole is therefore capped at 512. Cropping
a 512 window around the mask would avoid that and is the intended improvement.

**The spatial shape is passed, not inferred.** Upstream recovers it with `int(N ** 0.5)` in three
places. It is always square here, but a value derived from an assumption and a value passed in fail
differently when the assumption stops holding, and only one of them fails loudly.

**Two things upstream carries are not extracted.** `enable_migan` — an optional MI-GAN pre-fill
whose licence is a separate question, off by default and not part of the published result. And
`paste_compensate`, a colour-matching composite that is a third behaviour again.

## One shape

**Moebius runs at 512×512 and cannot run at anything else, and that is a property of the weights.**
Two independent reasons, both visible in the published tensors:

- `attn2.rel_pos_emb` is stored as `(4096, 10, 40, 1)` at the top level — a row per latent
  position. There is nowhere to put the positions of a larger image.
- `MQSλ_FwdWrapper.forward` and SANA's `GLUMBConv.forward` both recover the spatial shape with
  `int(N ** 0.5)`, so a non-square latent raises.

Nothing upstream documents this. Its own preprocessing can produce a shape its own model rejects,
which is evidence the repository is a benchmark harness over square datasets.

## Licensing

**The authors state weight terms twice and the two differ.** Both are recorded; neither is chosen:

- The GitHub README: *"Both the code and the pretrained model weights of Moebius are released under
  the Apache License 2.0, the same license used by the Qwen model family. Commercial use of the
  weights and the images produced with them is permitted."* This is the strong form — it names the
  weights separately from the code and grants commercial use explicitly.
- The Hugging Face model card metadata: `license: mit`.

Both are permissive and both permit commercial use, so the discrepancy is a documentation
inconsistency rather than a risk.

**The autoencoder is a separate work with a longer chain.** Upstream's config names `sdvae_f8d4`
and its README points at `hustvl/PixelHacker`, whose card says MIT. The config that travels with it
is `stabilityai/sdxl-vae`'s config with `sample_size` changed from 1024 to 512 — every other value
matches, including `scaling_factor` `0.13025` to five digits. So this is SDXL's autoencoder,
fine-tuned or not. That matters only for terms and does not change them: `stabilityai/sdxl-vae` is
published as its own MIT repository, separate from SDXL base's OpenRAIL++. **The chain terminates in
MIT whichever link is followed.**

**Third-party code on the extracted path**, all permissive:

| origin | what is used | licence |
|---|---|---|
| NVIDIA SANA | `GLUMBConv`, as the MixFFN | Apache-2.0, header verbatim in the file |
| timm (Ross Wightman) | `DepthwiseSeparableConv` | Apache-2.0, source URL in the file |
| diffusers | UNet, transformer and resnet scaffolding | Apache-2.0 |
| lambda-networks lineage | the λ layers, rewritten by the authors | Apache-2.0 with the repo |

### `flash-linear-attention` is not a dependency, because the teacher is not extracted

`requirements.txt` pins `flash-linear-attention[cuda]==0.3.2`, which reads like a hard CUDA
dependency. It is imported by exactly one file — `model_lib/nets/layers/gla/gla.py` — used by
exactly one model, `unet_gla.py`, the **PixelHacker teacher** used for distillation.
`model_lib/__init__.py` imports both student and teacher eagerly, which is the only reason it looks
mandatory.

Measured: stub that one import and the student builds with **neither `fla` nor `transformers` in
`sys.modules`**. The dependency is not worked around; it is never reached. This is worth stating
precisely because "we removed a CUDA-only dependency" and "we never took the file that imports it"
are different claims, and only the second is true.

## What the checkpoints contain

| artifact | bytes | sha256 (prefix) | contents |
|---|---|---|---|
| `general` / `torch-fp32-unet` | 905,297,927 | `6c9ee08c…` | 226.19M fp32, upstream's `pretrained` |
| `places2` / `torch-fp32-unet` | 905,298,356 | `6525afb8…` | upstream's `ft_places2` |
| `torch-fp32-vae` | 167,394,306 | `a59d7ea6…` | 83.65M **fp16**, cast at load |

Every digest is read from Hugging Face's API by `tools/fetch/moebius.py`, not transcribed.

Upstream also publishes `ft_celebahq` and `ft_ffhq`. They are face-specific, identically shaped,
and exercise no code path the other two do not — 1.7 GB for a task mozo has no other model for.
Named so that adding one is a decision rather than a discovery.

**Two things the config declares and the checkpoint does not have.** `mid_block_type: null`, so
there is no mid block at all — the class name `_prune_down_mid_up_block_8x8` is describing what was
removed. And `projection_class_embeddings_input_dim: 2560` is **inert**: `class_embed_type` is
never set, so `diffusers` builds no class embedding and no such tensor exists.

## Parity

`tools/verify/moebius.py`, exact — `torch.equal`, no tolerance.

| stage | result |
|---|---|
| autoencoder, 8 stages | **0.000e+00** |
| self-λ (local, r=15) | **0.000e+00** |
| cross-λ (global) | **0.000e+00** |
| UNet full forward, 1203 tensors | **0.000e+00** |
| DDIM, all 19 steps incl. the final-alpha branch | **0.000e+00** |

The gate is falsifiable and has been falsified: four perturbations, each caught, each at the stage
its constant reaches. The most useful is `GroupNorm` eps 1e-6 → 1e-5, which moves the encoder trunk
by 1.9e-03 — inside any tolerance a reasonable person would pick, which is the argument for having
none.

## Traps found by measuring

**A strict load cannot catch a missing activation.** The UNet loaded 1203/1203 keys — zero missing,
zero unexpected, zero shape mismatches — and was **4.46 off** at the output. The cause was one
SiLU: SANA's `ConvLayer` carries `act=("silu", "silu", None)`, so `MixFFN.inverted_conv` is
followed by an activation that has no parameters and therefore no tensor to miss. A clean strict
load is evidence about tensors, not about arithmetic.

**Layout is arithmetic, in three places.** `einops.rearrange` ends in `.contiguous()`; a permuted
view carries the same values with a different layout; and torch picks a different vectorised path
for each. Left as views, the self-λ landed 4.8e-07 from upstream *while every one of its
contractions was individually exact*, and the cross-λ landed 4.5e-08 through `BatchNorm1d`. This is
OWLv2's page-alignment trap arriving through a different door, and no operator-by-operator check
can see it.

**A gate at random weights is necessary and not sufficient.** The cross-λ passed bit-exact at random
weights and only diverged once trained values went through it.

**The Conv3d fold is not free.** `pos_conv` is `Conv3d(1, 40, (1, 15, 15))` — a depth-1 kernel,
algebraically a 2-D convolution. Folding it is what ExecuTorch and CoreML require, and it is
**2.1e-06** away on CPU, because torch dispatches two- and three-dimensional convolutions to
different kernels. So the torch path keeps the 3-D form and `fold_positional()` performs the rewrite
for export only, as a deliberate divergence with a number against its name.

**The cross-λ's positional gather is the identity.** Upstream indexes `rel_pos_emb[n, m]` where
`n, m` come from `meshgrid(arange(N), arange(M))`, so the gather returns what it was given. The
machinery is inherited from the self-λ's *global* branch, where the indices really are relative
offsets. Dropping it also disposes of a latent bug: `rel_pos` is a plain attribute rather than a
registered buffer, so it never followed `.to(device)`.

**The depthwise blocks activate with ReLU** — timm's default, never overridden. Every other
nonlinearity in the network is SiLU, so it reads like a transcription error and is not.

## The known optimisation that is not taken, and why

`MixFFN.forward` hands `inverted_conv` a **permuted view**, which is channels-last. `conv2d`
propagates that layout, so `depth_conv` — a 3x3 convolution with 3200 groups at the top level —
lands on a kernel that is catastrophically slower for it. Adding one `.contiguous()` after the
permute fixes it. Measured, at the published geometry:

| | as written | with `.contiguous()` | |
|---|---|---|---|
| `MixFFN` alone, CPU | 592 ms | **37.6 ms** | 15.7x |
| whole UNet forward, CPU | ~9.3 s | ~2.3 s | ~4x |
| one 512x512 removal, **MPS** | 14.13 s | 13.56 s | **1.04x** |
| output pixels changed | — | **0 of 262,144** | max byte delta 0 |

Three things follow, and only the third is a judgement call.

**It is not taken because it is not bit-exact.** Upstream's `GLUMBConv` has the same permute and no
`.contiguous()`, so being exact with upstream means being as slow as upstream here. The drift is
2.6e-06 on the UNet output — the same order as the Conv3d fold this package already declines on the
torch path.

**It is a CPU-only win.** Metal does not care about the layout: 4%, against 1470% on CPU. And
`tools/bench/moebius.py` establishes that CPU is not a serious target for this model at 204 s per
removal against Metal's 13.7 s. So the optimisation is large on the path nobody should be using and
absent on the path they should.

**It changes no output.** A full 19-step removal produced an identical image — every one of 262,144
pixels, to the byte. So the exactness being protected here is the *gate's* diagnostic power, not
the picture: 2.6e-06 on a latent is roughly 1500x below one 8-bit level.

If CPU inference ever matters, this is the first thing to take and the trade is one `.contiguous()`
against a gate that would need an explicit tolerance at this one site. It is recorded rather than
applied because that is the maintainer's call, not the extraction's.

Two smaller levers found at the same time, both also non-exact and both unapplied: `CrossLambda`
recomputes everything derived from the conditioning on all 19 steps though it is constant (186 ms
per forward, but caching it costs 780 MB), and the positional product is associated so it
materialises a 52 MB intermediate that the other association avoids entirely (186 ms → 85 ms, ~4e-07).

## Things that are upstream's behaviour, not bugs to fix

Reproduced, and recorded:

- **A run of twenty steps runs nineteen.** `strength=0.99` makes `get_timesteps` trim one from the
  front. Asking for twenty and running twenty produces a different image and raises nothing.
- **The clean image is encoded too.** Because `strength < 1`, the initial latent is the real image
  partially noised, not pure noise — so the autoencoder's encoder runs twice per call.
- **The "unconditional" branch is not null.** Ids 10–19 are ten *trained* embeddings, as much part
  of the model as ids 0–9. There is no empty prompt to substitute.
- **Three different guidance defaults ship in one repository**: the pipeline signature says 4.5,
  argparse says 2.5, the README example passes 2.0. mozo uses 2.0 — the documented invocation.

## Dead code upstream, not reproduced

- `no_cfg_after_step=1000` is a parameter of `__call__` and is never read.
- In `_post_process`'s `paste and compensate` branch, `m_img` is computed and then unused.
- `torch._dynamo.config.suppress_errors = True` executes at import in two files on the path — a
  process-wide compiler setting written by a library. Not carried. Its presence is also a hint
  that these layers do not compile cleanly, which matters for export.
- `build_removal_model` reaches its architecture by `eval()` on a string from a YAML file. mozo
  uses a frozen dataclass; there is one architecture and it does not need looking up by name.

## Export

Nothing is published beyond `torch-fp32-*`. `EXECUTES = ("torch",)`.

The groundwork is done and the reasons are recorded: `fold_positional()` handles the Conv3d that
ExecuTorch and CoreML both refuse, `einsum` and `einops` are gone from the forward path, and the
model has exactly one input shape — which is the thing that killed every previous mozo export
attempt and here cannot vary. A graph artifact will be published when it has a parity number and a
latency number beside it, and **not publishing one is a result worth writing down**.
