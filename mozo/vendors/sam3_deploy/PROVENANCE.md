# Provenance

This package is a deployment-only extraction of SAM 3's **image** path.

It derives from `transformers/models/sam3` (Apache-2.0, © The Meta AI Authors and The
HuggingFace Team) — **not** from `facebookresearch/sam3`, whose code ships under Meta's SAM
License. Choosing the Apache-2.0 implementation is what keeps `pip install mozo` free of
SAM-Licensed source. The published model is used as a black box for verification: it is run, and
its tensors are compared against, but its source is not the source of this code.

The tokenizer and its vocabulary are OpenAI's CLIP (MIT). See `NOTICE`.

**The weights are a different matter.** SAM 3's checkpoints are covered by the SAM License,
which is not open source and carries field-of-use restrictions that flow to downstream users.
This package does not redistribute them.

## What was taken

| from | what |
|---|---|
| `modeling_sam3.py` | the RoPE ViT trunk, the FPN neck, the geometry encoder, the DETR encoder and decoder, the dot-product scoring and the mask head |
| `configuration_sam3.py` | the numeric geometry, rewritten as frozen dataclasses in `config.py` |
| `convert_sam3_to_hf.py` | the checkpoint key-rename table, adapted in `checkpoint.py` |
| `openai/CLIP` | the byte-pair tokenizer and `bpe_simple_vocab_16e6.txt.gz` |

The checkpoint is consumed **exactly as Meta publishes it** — no repacking, no pruning, no
mozo-format artifact. `checkpoint.py` translates key names at load time instead.

## What was deliberately left behind

The entire video path: memory attention, the memory encoder, temporal disambiguation, tracklet
bookkeeping, multi-GPU inference, the training code and the evaluation harness.

Weights present in the checkpoint that this package does not build:

- `detector.backbone.language_backbone.encoder.text_projection` — upstream builds it, applies it,
  and the caller discards the result. Dropping it saves a matmul per prompt and 2 MB. The parity
  gate covers the claim: if it were live, the outputs would not match.
- `detector.geometry_encoder.points_{direct,pool,pos_enc}_project` — the geometry encoder supports
  **point** exemplars as well as boxes, but neither `Sam3Processor` nor `transformers` exposes
  them, and there is no Apache-licensed implementation to derive the behaviour from.

Weights that are built, because a strict load needs somewhere to put them, but never run:

- `segmentation_head.pixel_decoder.{conv_layers,norms}.2` — the pixel decoder allocates three
  upsampling stages and runs one per *gap* between pyramid levels. Three levels means two gaps.
- The lowest-resolution FPN level in both neck stacks, which upstream builds and then scalps.

## Where this diverges from `transformers`, and why

Every item below was found by comparing numbers against the published model, not by reading.
Each one is a place where following the Apache-2.0 source faithfully would have produced
confident, plausible, wrong output.

| divergence | cost of following `transformers` |
|---|---|
| The geometry encoder runs on **every** prompt, emitting a CLS token even with no boxes. `transformers` skips it when `input_boxes` is empty. | The prompt becomes 32 tokens where the weights expect 33 — on every text prompt. |
| The decoder's normalisations are numbered in a different order than they run: `norm2` follows self-attention, `norm1` follows cross-attention. | Two normalisations on the wrong residuals. Measured **6.2e-01** on boxes. |
| `freqs_cis` is loaded from the checkpoint and applied in complex arithmetic. `transformers` rebuilds real `cos`/`sin` tables. | Rebuilding is 5.96e-08 off; the real-valued rotation compounds to **1.35e-02** by block 31. |
| `qkv` stays fused, as the checkpoint stores it. | Splitting is algebraically identical, numerically not. |
| Layout is chosen per module — sequence-first in the geometry encoder and decoder, batch-first in fusion. | `nn.MultiheadAttention` transposes internally for `batch_first=True`, selecting a different kernel. One ulp at layer 0, growing through the stack. |
| `need_weights` differs per attention call: default for the two unbiased attentions, `False` for the box-biased one. | Each selects a different path inside PyTorch. Getting it uniform costs 5e-06 either way. |
| Padding masks are `True` for padding. `transformers` uses `True` for valid. | Attends to exactly the tokens meant to be ignored. |
| The trunk hands the neck a permuted view, not a contiguous tensor. | `conv2d` picks a different kernel for channels-last. **2.15e-05** on the neck output while the trunk stays bit-identical. |
| `layer_norm_eps` is 1e-5. `transformers`' `Sam3VitConfig` defaults to 1e-6. | The weights were trained at 1e-5. |
| Preprocessing multiplies by `1/255` rather than dividing by 255, and resizes in uint8 with antialiasing. | Dividing moves 1.2M of 3M pixels by one ulp; skipping antialias reaches 72 grey levels. |
| The presence logit is returned **unclamped**. `transformers` clamps it to ±10 in the decoder loop. | Upstream carries a clamp setting and leaves the returned logit untouched by it — measured −10.719295 for an absent concept, where clamping gives −10. Only visible on prompts that find nothing. |
| The pixel decoder is restored to channels-last before each convolution. | `F.interpolate` drops the format; `conv2d` and `GroupNorm` are layout-sensitive while interpolation and addition are not. 9.57e-06 on pixel embeddings, 2.7e-04 on masks. |
| `need_weights` in the mask head's prompt cross-attention is `False`, where the decoder's prompt cross-attention takes the default — despite both carrying a key-padding mask. | 2.5e-06 at that call, compounding through the pixel decoder. Which path reproduces upstream is not derivable from the call signature; it was measured at each of the four sites. |

## Measured parity

Against the published model on `tests/fixtures/images/example.jpg`, CPU, float32.

| stage | result |
|---|---|
| preprocessing | max abs diff **0** |
| vision trunk + dual neck (all 6 pyramid tensors + positions) | max abs diff **0** |
| tokenizer, 9 prompts incl. casing, punctuation, non-ASCII, empty, truncation | identical ids |
| text tower (mask, features, embeddings) | max abs diff **0** |
| geometry encoder | max abs diff **0** |
| fusion encoder | max abs diff **0** |
| DETR decoder (`pred_boxes`, `presence`) | max abs diff **0** |
| dot-product scoring (`pred_logits`) | max abs diff **0** |
| mask head (`pred_masks`) | max abs diff **0** |
| **whole concept path, end to end** — 5 prompts incl. a multi-word phrase and an absent concept | masks, boxes, logits and presence all max abs diff **0** |
| exemplar boxes — positive, negative, and two with mixed labels | max abs diff **0** |
| `Segmenter`, through both caches — masks, boxes and scores after thresholding | max abs diff **0** |

Every other gate prompts with no boxes, so the exemplar row is what exercises `roi_align` and the
three box projections at all. The two-box case is what makes it meaningful: with a single box the
reference's sequence-first prompt is indistinguishable from a batch-first one, so a transpose
would pass unnoticed.

The click path is gated too, on seven prompt shapes: one positive point, positive with negative,
two positives, a box, a box with a negative point, a lone negative point, and
`multimask_output=False`. The last two exist because a simpler set misses them -- a lone negative
is the only prompt with nothing to include, and the single-mask call is the only one that reaches
the stability fallback. Scores and low-resolution logits are bit-identical to the published model
on all seven; binary masks are excluded from the comparison for the hole-filling reason below.

## The click path

Built on `mozo.vendors.sam2_deploy`'s `PromptEncoder`, `MaskDecoder` and `TwoWayTransformer`
rather than on new modules. This is the one place mozo lets two vendor trees import each other,
and it is deliberate: the checkpoint stores these weights under `tracker.sam_prompt_encoder` and
`tracker.sam_mask_decoder` with SAM 2's module names, shapes and token counts, and Meta names the
neck that feeds them `sam2_convs`. A second copy of ~600 lines that had to stay bit-identical to
the first forever is a worse failure mode than the coupling -- a divergence between the copies
would surface as wrong masks, not as an import error.

Configured for 1008 pixels over a 72x72 grid instead of 1024 over 64x64. Every value was read
back from the published model's own attributes and then confirmed by a strict load:
`image_size 1008`, `backbone_stride 14`, `sam_image_embedding_size 72`, `low_res_mask_size 288`,
`sam_prompt_embed_dim 256`, four mask tokens, object scores on, high-resolution features on.

One prompt structure, not several. Every click prompt is a set of labelled points: `1` include,
`0` exclude, `2`/`3` reserved for a box's two corners, since the network has no box input.
Points, a box, and a box with points are the same array filled differently.

Three things about it could not be read off the weights, and each was found by measuring:

- **`dynamic_multimask_via_stability` is on.** It carries no parameters, so the strict load could
  not catch it, and it only changes an answer when the single-mask token is unstable. It survived
  23 of 24 parity prompts before one image caught it.
- **The two heads do not share a preprocessing**, and therefore cannot share an image encode.
  The concept path rounds the resize back to uint8 and multiplies by `1/255`; the click path stays
  in float and divides by `255`. Both choices are the opposite of the other's, they differ by half
  a grey level, and that is worth 9e-03 of predicted IoU and several thousand mask pixels. The
  published model runs the trunk twice for this reason and so does mozo, with a second cache.
- **`no_mem_embed` applies here too**, as it does on SAM 2's image path -- trained as "there is no
  memory to attend to", which is a single image's situation exactly.

`tracker.*` is 309 tensors; 149 of them (4.2 M parameters) are the click path and are loaded. The
other 160 (7.5 M) are memory attention and mask-memory fusion, which have nothing to attend to on
a still image, and are filtered at load.

**Hole filling is not implemented**, matching `sam2_deploy`, which records the same omission: it
needs a CUDA connected-components extension. Upstream fills holes up to 256 low-resolution pixels
before thresholding, so mozo's binary masks come out a *strict subset* of the reference's --
verified as 0 pixels present in mozo and absent upstream, with every difference lying inside a
filled hole. The low-resolution logits the masks are thresholded from are bit-identical.

## Dependencies

`regex` and `ftfy` are core dependencies, needed by the tokenizer: CLIP's split pattern uses
Unicode categories (`\p{L}`, `\p{N}`) that the standard library's `re` cannot express, and `ftfy`
repairs mis-decoded input the way upstream does.

`torchvision` is a declared core dependency, and `grounding/geometry.py` imports it at module
scope rather than defending against its absence. It was already a de-facto dependency before SAM 3
— `rfdetr_deploy` and `depth_anything_v2_deploy` both import it at module scope — and SAM 3 needs
`roi_align` for exemplar-box pooling. Declaring it is what makes a missing install fail at install
time instead of inside a forward pass.

Writing `roi_align` against `grid_sample` to drop the dependency was considered and rejected: it
would remove nothing from the install, since two other families already require torchvision.
