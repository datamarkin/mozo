# Provenance

This package is a deployment-only extraction of **CLIP**'s inference path.

It derives from `openai/CLIP` (MIT, © 2021 OpenAI), pinned at commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. That repository is the authors' own and is what the
published checkpoints reproduce, so it is the reference; `transformers/models/clip` is a port and
is not used here at all.

**The weights are MIT.** The repository publishes one `LICENSE` and no separate weights licence,
so the checkpoints are covered by it. That is an inference from silence and is recorded as one,
here and in the `NOTICE` beside every checkpoint — unlike Grounding DINO, where the authors state
the weights' terms explicitly on the repositories that serve them.

## What was taken

| from | what |
|---|---|
| `clip/model.py` | `ResidualAttentionBlock`, `Transformer`, `VisionTransformer`, the text tower's forward, `LayerNorm`, `QuickGELU` |
| `clip/simple_tokenizer.py` | the byte-pair encoder, rewritten in `text/tokenizer.py` |
| `clip/clip.py` | `tokenize`'s wrapper, and `_transform`'s resize/crop/normalise, rewritten in `image.py` |
| `clip/model.py` `build_model` | the geometry it infers, written out as frozen dataclasses in `config.py` |
| `bpe_simple_vocab_16e6.txt.gz` | the vocabulary, byte-identical (sha256 `924691ac…`, 1,356,917 bytes) |

## What was deliberately left behind

**The five ResNet variants.** RN50, RN101, RN50x4, RN50x16 and RN50x64 replace the Vision
Transformer with a modified ResNet and an attention-pooling head. A second image tower, not a
second configuration. Nothing here is in the way of adding it.

**Training.** The contrastive loss, the temperature's gradient, `convert_weights`, and the
distributed plumbing.

**`build_model`'s inference of geometry.** Upstream reads shapes out of the state dict to decide
what to build. `config.py` writes the geometry down instead, and the strict load is what holds the
two in step — a spec that is inferred cannot be checked.

**Upstream's `jit=True` path.** `clip.load` can return the scripted archive and run it directly.
mozo repacks to plain tensors at publish time instead; see the `NOTICE`.

**`truncate=True`.** `tokenize` can silently keep the first 76 tokens and overwrite the last with
the end marker. Not carried: a prompt shortened without saying so is a different prompt, and only
the caller can decide what to drop. `Tokenizer.__call__` raises with the count and the cap.

**`logit_scale` on the inference path.** It is loaded — `read_logit_scale` — but nothing in
`predictor.py` multiplies by it, because mozo returns cosine similarities rather than the logits
upstream softmaxes. Its only reader is `tools/verify/clip.py`, which needs it to compare against
upstream's `logits_per_image`. Recorded here rather than left as a tensor nobody can account for.

## Where this diverges from upstream, and why

Every item was found by comparing numbers against the reference, not by reading.
`tools/verify/clip.py` is where they stay found.

### 1. The towers are built and loaded separately

Upstream builds one `CLIP` module holding both towers and loads the checkpoint into it whole. This
partitions the state dict by the `visual.` prefix and loads each tower on its own, so an ingest job
never allocates the text tower and a query service holds 63.4M parameters rather than 151.3M.

The load stays strict within each half — a key belonging to neither partition, or a module left
unfilled, is an error. `input_resolution`, `context_length` and `vocab_size` are dropped by name:
they are scalars recording how the model was built, not weights, and upstream deletes them for the
same reason before its own strict load.

This is a structural difference with no numerical one. Both towers are bit-exact against upstream.

### 2. Vectors leave L2-normalised

Upstream's `encode_image` and `encode_text` return unnormalised features and leave normalising to
the caller — its own `forward` does it immediately afterwards. mozo normalises inside the vendor,
so a dot product between any two vectors it emits is a cosine similarity with no convention to
document. Two callers normalising differently is a class of bug that never raises.

The gate compares the unnormalised features too, so this is an addition rather than a change.

## Traps

Each of these runs, produces plausible output, and is wrong.

- **`QuickGELU`, not `nn.GELU`.** `x * sigmoid(1.702 * x)`, at `model.py:166-168`. A sigmoid
  approximation predating the erf one. Substituting the standard activation is the commonest CLIP
  reimplementation error and it neither raises nor warns.
- **Operator precedence changes the logits.** Upstream writes
  `logit_scale * image_features @ text_features.t()`. In Python `*` and `@` share precedence and
  associate left to right, so the scale multiplies the *features* before the matmul, not the
  product after it. Same arithmetic, different rounding: measured at **1.9e-06** on ViT-B/32, and
  the gate is only exact when it scales in upstream's order.
- **`LayerNorm` is subclassed** at `model.py:157` to normalise in fp32 and cast back. Under fp16 a
  plain `nn.LayerNorm` diverges. mozo publishes fp32, where the cast is a no-op — kept because it
  is what upstream runs.
- **`need_weights=False`** at `model.py:187` selects torch's fused attention path, which is the
  same arithmetic in a different order from the unfused one. Grounding DINO's gate measured that
  class of difference at 1.5e-06.
- **The normalisation constants are CLIP's, not ImageNet's.** Mean
  `(0.48145466, 0.4578275, 0.40821073)`, std `(0.26862954, 0.26130258, 0.27577711)`. Close enough
  to look like a typo when they differ.
- **Bicubic, through PIL.** `_transform` resizes a `PIL.Image` with `InterpolationMode.BICUBIC`.
  PIL's filter has support that scales with the downsampling factor; neither `cv2.resize` nor
  `F.interpolate` reproduces it, with or without `antialias`.
- **`Resize` takes one int, not a pair.** `Resize(224)` scales the *short* side and keeps the
  aspect ratio; `Resize((224, 224))` squashes the image and is a different picture.
- **Pooling is by `argmax` over token ids, not by position.** The end-of-text marker is the highest
  id in the vocabulary, so the largest id marks where the prompt stopped. It survives zero padding
  because 0 is the lowest id. It would *not* survive a prompt containing a literal
  `<|endoftext|>`, which would pool there instead; upstream behaves the same way.
- **Single digits are separate tokens.** The split pattern is `[\p{N}]`, one digit at a time, so
  "2024" is four tokens and a numeric prompt eats context faster than it looks.
- **`html.unescape` is applied twice.** Web-scraped captions carry double-escaped entities, and one
  pass leaves `&amp;amp;` as `&amp;`.

## Numbers

Parity on `tests/fixtures/images/example.jpg` against the pinned upstream checkout, CPU, `base`:

| stage | |
|---|---|
| preprocessed tensor | exact |
| token ids | exact |
| image features | exact |
| text features | exact |
| cosine similarities | exact |
| `logits_per_image` | exact, once scaled in upstream's order |

Extraction size: ~950 lines across 11 modules, against roughly 1,500 in the upstream files it came
from — a smaller reduction than most families here, because CLIP's reference implementation is
already close to deployment-only.
