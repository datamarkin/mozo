# Provenance

This package is a deployment-only extraction of **Grounding DINO**'s detection path.

It derives from `IDEA-Research/GroundingDINO` (Apache-2.0, © 2023 – present IDEA Research), which
is the authors' own release. That makes this the opposite of OWLv2's case: there, the authors
publish JAX and the extraction had to come from HuggingFace's PyTorch port; here the authors'
PyTorch code *is* what the published checkpoints reproduce, so it is the reference and
`transformers/models/grounding_dino` is used only as a cross-check on the key mapping.

The tokenizer's vocabulary is Google's `bert-base-uncased` (Apache-2.0). See `NOTICE`.

**The weights are Apache-2.0**, on the authors' say-so — stated on the Hugging Face repositories
that serve the files, and nowhere in the GitHub project. `NOTICE` records exactly where.

## What was taken

| from | what |
|---|---|
| `models/GroundingDINO/groundingdino.py` | the assembly, the input projections, the box and contrastive heads |
| `models/GroundingDINO/transformer.py` | the deformable encoder, the cross-modality decoder, language-guided query selection |
| `models/GroundingDINO/fuse_modules.py` | `BiAttentionBlock` and its bi-directional attention, rewritten in `fuse.py` |
| `models/GroundingDINO/bertwarper.py` | the phrase-isolating attention mask and per-phrase position ids, rewritten as `network.phrase_masks` |
| `models/GroundingDINO/backbone/swin_transformer.py` | the Swin backbone, reduced to the branches the published configs take |
| `models/GroundingDINO/backbone/position_encoding.py` | `PositionEmbeddingSineHW`, rewritten in `position.py` |
| `models/GroundingDINO/ms_deform_attn.py` | the `grid_sample` deformable attention, without the CUDA extension |
| `models/GroundingDINO/utils.py` | `MLP`, `ContrastiveEmbed`, `gen_encoder_output_proposals`, the two sine embedders |
| `datasets/transforms.py` | the 800/1333 aspect-preserving resize, rewritten in `image.py` |
| `config/GroundingDINO_Swin{T_OGC,B_cfg}.py` | the numeric geometry, written out as frozen dataclasses in `config.py` |
| `transformers` BERT | the encoder architecture, rewritten in `text/bert.py` — **weights come from the Grounding DINO checkpoint**, not from Google |
| `bert-base-uncased` | `vocab.txt`, and the WordPiece algorithm, rewritten in `text/tokenizer.py` |

The checkpoint is consumed **exactly as IDEA Research publishes it** — `.pth`, no repacking, no
pruning, no mozo-format artifact. `checkpoint.py` renames one prefix and drops three at load time.

**There is no `transformers` dependency at run time.** Upstream builds
`BertModel.from_pretrained("bert-base-uncased")` and then overwrites every tensor from the
checkpoint, which carries its own fine-tuned copy under `bert.*` — 200 tensors, most of the file's
694 MB. The download is only ever used for its shape. Rebuilding that shape here removes the
dependency, the network round trip, and the version drift.

## What was deliberately left behind

**The CUDA/C++ deformable-attention extension.** Upstream ships `groundingdino._C` and falls back
to `grid_sample` when it is absent. Only the fallback is carried: it needs no compiler, it runs on
every device mozo targets, and it is the path the parity gate can compare. The extension is a
faster route to the same arithmetic, not a different model.

**Denoising training** (`dn_number`, `dn_box_noise_scale`, `dn_label_noise_ratio`,
`dn_labelbook_size`) and the `label_enc` embedding it needs — 2,001 × 256 present in every
published checkpoint, never read at inference.

**BERT's pooler.** Present in the checkpoint, frozen by upstream, and the detection path reads only
`last_hidden_state`. Nothing here is built and left unrun; everything that loads, runs.

**The one-stage branch** (`two_stage_type="no"`), `num_patterns`, encoder layer sharing, gradient
checkpointing (`use_checkpoint`, `use_transformer_ckpt`), auxiliary per-layer outputs, and the
encoder-output head. All are training or ablation paths, none reachable from a published config.

**The RoBERTa text encoder and the ResNet backbone.** Upstream can build both; neither published
checkpoint uses either.

**Upstream's phrase decoding** (`get_phrases_from_posmap`). Replaced rather than dropped — see
below.

## Where this diverges from upstream, and why

Every item below was found by comparing numbers against the reference, not by reading.
`tools/verify/grounding_dino.py` is where they stay found: **138 comparisons across both
variants, exact equality, no tolerance.**

### 1. The encoder's box head is its own module, not the decoder's

`two_stage_bbox_embed_share = False` in both published configs, so upstream *deep-copies* the box
head for query selection and trains it separately. Aliasing the two is the obvious reading of
`self.transformer.enc_out_bbox_embed = _bbox_embed` a few lines above — and it is wrong.

This is the one divergence that a strict state-dict load cannot catch. Both key sets still match,
so nothing is reported missing or unexpected; one simply overwrites the other depending on
traversal order. It moved the initial reference boxes by **12.9** and every prediction with them,
while `load_state_dict(..., strict=True)` returned silently.

### 2. The reference must be pinned to eager attention

`transformers` changed `BertModel`'s default `attn_implementation` to SDPA. That is the same
arithmetic in a different order and it moves `last_hidden_state` by **1.5e-06** — small, and it
propagates through six fusion layers and six decoder layers into visibly different boxes. The
published checkpoint predates the change; eager is what it was trained and released against, and
what this package reproduces bit-exactly. The gate pins it.

This is the wrapping tax in one line: the same code, the same weights and the same input produce
different numbers on a `transformers` upgrade nobody asked for.

### 3. `strict=False` is not carried

Upstream loads with `strict=False`, which cannot distinguish a tensor deliberately not built from
one renamed by mistake — the second leaves a module at its random initialisation, runs, and
returns confident wrong boxes. `checkpoint.py` drops the three genuinely-absent prefixes by name
and loads strictly.

### 4. The name on a detection is the caller's phrase, not a decoded span

**This is a deliberate behavioural difference, and the only one.** The numbers are identical; what
changes is the string.

Upstream takes each surviving query, keeps every text token whose similarity clears
`text_threshold`, and decodes those tokens back into a string. Nothing constrains the kept tokens
to lie inside one prompt, so it can return `"yellow school"` for `"a yellow school bus"`, or a span
running across a separator.

mozo instead reports **which prompt** the query matched, and returns that prompt verbatim. This is
exact rather than heuristic: `phrase_masks` already computes per-phrase token membership — it has
to, to build the attention mask — so the query's peak token maps to exactly one phrase. A query
whose peak lands on a separator names no prompt and is dropped rather than guessed.

Three reasons. It matches OWLv2, mozo's other `open_vocabulary_detection` family, where
`class_name` is the phrase you searched for; two families answering one task through one endpoint
must agree on what a class name is. mozo owns the caption — the endpoint takes `?text=` repeated
and joins it — so it knows where the separators are rather than inferring them. And a fragment is
a worse answer than the phrase asked for.

`text_threshold` is therefore not used, and is not exposed. It exists upstream only to select the
tokens that get decoded.

### 5. `phrase_masks` restarts its separator scan per batch row

Upstream's `generate_masks_with_special_tokens_and_transfer_map` carries `previous_col` across
rows of a batch, so a second caption inherits the first one's last separator position. mozo runs
one caption per image, where the two are identical — the reset is a correctness property that
costs nothing here, not a numerical change. Recorded rather than silently matched, because a
future batched path would need it.

### 6. The padding mask stays commented out

Upstream disables the padding mask in the phrase-isolation function, so padded tokens are attended
to. It looks like an oversight and it is what the published weights were trained and evaluated
with. Reproduced exactly. mozo never pads a caption anyway — one caption per call — so it is
unreachable in practice and would matter the moment a batched path existed.

## Traps that survive as tests

Each of these runs, produces plausible output, and is wrong.

- **`pe_temperatureH/W = 20`**, not the DETR default of 10000. Every other DETR in this repository
  uses 10000, so a copied sine position encoding brings the wrong constant.
- **The 800/1333 resize can exceed 1333.** A 500×4000 image comes back 167×**1336**: upstream
  rounds the short side to an integer first, then scales the long side from it, so the cap binds
  the ratio rather than the result. Clamping to 1333 is a reasonable-looking correction that moves
  every box on a panoramic image. Pinned in `tests/families/test_grounding_dino.py`.
- **The resize goes through PIL**, whose bilinear filter has support that scales with the
  downsampling factor. Neither `cv2.resize` nor `F.interpolate(antialias=True)` reproduces it.
- **The caption is lowercased** and gains a trailing `"."`. Proper nouns do not survive.
- **`"?"` is a separator**, alongside `[CLS]`, `[SEP]` and `"."`. A prompt containing one splits;
  mozo refuses such a prompt rather than reporting its detections against the wrong phrase.
- **`max_text_len = 256` truncates silently upstream** — roughly 60 phrases, dropped without a
  word. mozo raises, naming the count and the cap.
- **The fusion block's residual is added to the normalised input**, not to the input. Upstream
  reassigns `v` and `l` before the attention call, so reading it as the usual `x + f(norm(x))`
  gives a different model.
- **The fusion softmax subtracts a global maximum**, not a per-row one. The per-row spelling is
  the usual stable-softmax idiom and a different number.

## Numbers

Parity, on `tests/fixtures/images/example.jpg` with four prompt sets, against the pristine upstream
checkout on CPU:

| stage | tiny | base |
|---|---|---|
| preprocessing tensor | exact | exact |
| token ids | exact | exact |
| BERT `last_hidden_state` | exact | exact |
| Swin features, all 3 levels | exact | exact |
| projected text (`feat_map`) | exact | exact |
| encoder memory, fused text | exact | exact |
| every decoder layer (6) | exact | exact |
| `pred_logits`, `pred_boxes` | exact | exact |

`138 comparisons, all identical.`

Every row is hooked in the gate rather than checked once by hand: two implementations can agree
on the last tensor and disagree in the middle, and when they do, the stage name is the difference
between "something moved" and knowing which of eleven rewritten pieces moved it. Perturbing
`pe_temperature_h` from 20 to 21 fails 40 of the 65 per-variant comparisons and leaves the
backbone, BERT and projected-text rows passing -- which is where that constant does and does not
reach.

Extraction size: ~2,300 lines across 13 modules, against roughly 7,200 in the upstream files it
was taken from.
