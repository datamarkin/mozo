# Provenance

This package is a deployment-only extraction of [EdgeTAM](https://github.com/facebookresearch/EdgeTAM),
reduced to single-image promptable segmentation.

EdgeTAM is SAM 2 with two changes: a RepViT-M1 trunk in place of Hiera, and a 2-D Spatial
Perceiver that compresses the video memory bank. The second is the paper's contribution and is
entirely about video, so what is left here is SAM 2's image path running on a smaller trunk.

| | |
|---|---|
| Upstream repository | `https://github.com/facebookresearch/EdgeTAM` |
| Upstream commit | `7711e012a30a2402c4eaab637bdb00a521302c91` |
| Trunk repository | `https://github.com/huggingface/pytorch-image-models` |
| Trunk commit | `0e968e1cb8b18ea66e6f492ea9536ce21cf062b1` (v1.0.27) |
| Extracted | 2026-08-21 |
| Verified against | `torch` 2.11.0, Python 3.10, on CPU |

Record the commits before re-syncing: without them, diffing a later upstream against this copy
means guessing which revision it started from.

## Why two upstreams

EdgeTAM does not carry its own trunk. Both of its published entry points reach for timm:
`sam2/modeling/backbones/timm.py` calls `timm.create_model("repvit_m1.dist_in1k",
pretrained=True, features_only=True)`, and the `transformers` port defers to
`AutoConfig.from_pretrained("timm/repvit_m1.dist_in1k")` through `timm_wrapper`. Neither is
available to a vendor here — one adds a dependency, both fetch from the network at construction —
so `backbones/repvit.py` carries the modules instead, from timm, which is Apache-2.0.

The checkpoint decides the naming, not this package. EdgeTAM stores the trunk under timm's own
key names, wrapped twice: `body` is `TimmBackbone`'s attribute, and `stages_0` rather than
`stages.0` is timm's `FeatureListNet` flattening a feature-extracting model into a `ModuleDict`.
Reproducing both is what makes the load strict with no rename table at all.

## What was taken

Verbatim from `facebookresearch/EdgeTAM`, apart from rewriting imports to relative form and
adding docstrings:

```
sam/mask_decoder.py          <- sam2/modeling/sam/mask_decoder.py
backbones/image_encoder.py   <- sam2/modeling/backbones/image_encoder.py
```

Reduced, with everything the image path cannot reach removed:

- `layers.py` — `LayerNorm2d` and `MLP` from `sam2/modeling/sam2_utils.py`. Everything below them
  samples training clicks (`sample_box_points`, `sample_random_points_from_errors`,
  `sample_one_point_from_error_center`, `get_next_point`) and was the only thing importing OpenCV
  and `mask_to_box`; everything above belongs to the tracker. `DropPath` goes too — SAM 2's Hiera
  trunk uses it, RepViT does not.
- `position_encoding.py` — `PositionEmbeddingSine` and `PositionEmbeddingRandom`. The sine one's
  `encode_boxes` and `encode_points` encode object pointers and are unreachable from an image.
  The rotary helpers below them (`init_t_xy`, `compute_axial_cis`, `reshape_for_broadcast`,
  `apply_rotary_enc`, and EdgeTAM's own `apply_rotary_enc_v2`) belong to memory attention.
- `sam/transformer.py` — `TwoWayTransformer`, `TwoWayAttentionBlock`, `Attention`.
  `RoPEAttention` and EdgeTAM's `RoPEAttentionv2` are memory attention. `sdp_kernel_context`,
  `get_sdpa_settings` and the module-level `warnings.simplefilter` are discussed below.
- `backbones/repvit.py` — from timm: `ConvNorm`, `RepVggDw`, `RepVitMlp`, `RepViTBlock`,
  `RepVitStem`, `RepVitDownsample`, `RepVitStage`, the body of `RepVit`, and `SqueezeExcite` from
  `timm/layers`. Left behind: the classifier head and its distillation twin, the model registry
  and the other seven variants, `forward_intermediates`, gradient checkpointing, the
  `device`/`dtype` construction kwargs, the `legacy=False` branch (EdgeTAM's weights are
  `legacy=True`), and every `fuse()` — folding the reparameterisable branches into one
  convolution rewrites the published weights, and running them unchanged is this package's
  whole claim.

Written for this package:

- `config.py` — the geometry as plain data, replacing the Hydra YAML. Values read out of
  `sam2/configs/edgetam.yaml`, plus three from `build_sam2` (see below). This is what removes
  `hydra-core` and `omegaconf`.
- `image.py` — preprocessing, prompt scaling and mask resizing, replacing `sam2/utils/transforms.py`.
  That module builds a `torch.jit.script`-ed `nn.Sequential` of torchvision's `Resize` and
  `Normalize`; doing the two operations directly is what removes `torchvision` and keeps a
  deprecated `torch.jit` call out of the package.
- `network.py` — the image-mode model, replacing `SAM2Base`, split into `encode` and `decode`.
- `predictor.py` — checkpoint loading, prompt assembly, and the encoder-output cache.

## What was deliberately left behind

The whole video tracker: `memory_attention.py`, `memory_encoder.py`, `perceiver.py`,
`sam2_video_predictor.py`, and on `SAM2Base` the memory bank, object pointers, temporal position
encoding and frame bookkeeping. Also the training code, `automatic_mask_generator.py`, the
CoreML export scripts, and the Hydra build path.

Dropping these removes `hydra-core`, `omegaconf`, `iopath`, `torchvision` and `timm` from the
install. 833 of the checkpoint's 982 tensors are used; the 149 dropped are 4.78 M parameters of
9.90 M, and they are filtered out at load rather than being carried dead:

| dropped | tensors | parameters |
|---|---|---|
| `memory_attention` | 54 | 2.961 M |
| `memory_encoder` | 40 | 1.385 M |
| `spatial_perceiver` | 44 | 0.231 M |
| `obj_ptr_proj` | 6 | 0.197 M |
| `mask_downsample`, `maskmem_tpos_enc`, `no_mem_pos_enc`, `no_obj_ptr` | 5 | < 0.001 M |

What is left is a 9.121 M-parameter image path.

One tensor that *looks* like tracker state is kept. `no_mem_embed` is trained to mean "there is
no memory to attend to", which is exactly the situation a single image is in; upstream's config
says so directly with `directly_add_no_mem_embed: true`, and its own image predictor adds it in
`set_image` before caching the features.

## Deliberate divergences

Three, all measured rather than assumed.

| divergence | why, and what it costs |
|---|---|
| `sam/prompt_encoder.py` selects label embeddings with `torch.where` instead of `point_embedding[labels == 0] += ...`. | A boolean mask index is data-dependent and does not trace, so an export would bake in one prompt's labels or refuse to convert. EdgeTAM forked from SAM 2 before upstream made this same change for the same reason. **Bit-identical**: the prompt encoder's sparse and dense outputs agree exactly. |
| `sam/transformer.py` does not pin the SDPA backend. | Upstream's `sdp_kernel_context` derives flags from local CUDA capability *at import time*: `(True, False, True)` on any machine without a modern CUDA GPU, which pins the math kernel, and flash on a recent one. Upstream's own logits therefore differ between a laptop and an A100. Dropping the pin costs **2e-07 per attention layer**, compounding to **9.2e-05** on the decoder's mask logits against upstream-on-CPU; every other stage stays bit-identical. It is dropped because it makes the answer depend on the card in the machine, because `torch.backends.cuda.sdp_kernel` is deprecated in torch 2.11, because it probes a device at module scope, and because the kernel it pins is 17 percent slower here (38.7 ms against 33.0 ms per decode). `sam2_deploy` made the same call on the same code. |
| The stability fallback is enabled from `config.py` rather than from the YAML. | `dynamic_multimask_via_stability`, its delta and its threshold are not in `edgetam.yaml` at all — `build_sam2` appends them as Hydra overrides when `apply_postprocessing` is true, which is its default. Reading only the config file would leave them off. They carry no weights, so a strict load cannot catch it, and they change nothing unless a caller asks for a single mask. `tools/verify/edgetam.py` runs one prompt with `multimask_output=False` for exactly this reason. |

## How it is checked

`tools/verify/edgetam.py` runs both implementations on the same photograph and compares every
stage exactly — preprocessing, the image embedding, both high-resolution feature maps, and the
logits and predicted IoU of seven prompts. It needs a checkout of upstream, and upstream needs
`hydra-core` and `timm` to run at all; neither is a dependency of mozo, of this package, or of
the test suite.

### A third implementation, on the prompt shape specifically

Bit-exactness against upstream proves this package agrees with *one* way of driving the model,
and it inherits that way's assumptions. The assumption worth doubting is the box: this package
spells one as two corner tokens folded into the point list with `boxes=None`, because that is
what upstream's `SAM2ImagePredictor._predict` does, so the gate cannot tell the difference
between "correct" and "wrong in the same way as the reference".

`transformers`' `EdgeTamModel` is an independent check on exactly that. It carries its own weight
conversion, its own preprocessing, and a prompt encoder that builds box tokens the *other* way —
`_embed_boxes` emits `[top-left, bottom-right, padding]` directly and appends them *after* any
clicks, rather than folding corners in ahead of them. Its parameter count matches this package's
image path to the tensor (9.120641 M), and on the fixture it agrees at **0.995–0.999 mask IoU**
across clicks, click pairs and three boxes, with predicted IoU matching to three decimals.

The two orderings are the same multiset of tokens, which is why they land in the same place: each
token's learned embedding is selected by its label, not by its position, and the sparse tokens
are only read through attention rather than sliced by index. Reversing the order here moves the
mask logits by 3.1e-05 — float summation, not a different mask. This is recorded because the
opposite is the natural assumption, and it is stated as fact in some SAM documentation.
