# Provenance

This package is a deployment-only extraction of **SigLIP 2**'s inference path.

It derives from `transformers/models/siglip` (Apache-2.0, © The Google Research Authors and The
HuggingFace Team) — **not** from `google-research/big_vision`, which is the authors' own release
and is written in JAX/Flax. Both are Apache-2.0, so this is not a licensing choice; it is that the
PyTorch implementation is the one whose numbers the published PyTorch checkpoints reproduce, and
the one mozo can be checked against on every run. `owlv2_deploy` chose the same way for the same
reason.

**The weights are Apache-2.0**, stated on every model card and served ungated. Unlike CLIP, where
MIT over the checkpoints is an inference from a repository that publishes no separate weights
licence, nothing here is inferred.

**SigLIP 1 is not in this package and never will be.** It is a different release with a different
tokenizer, a different vocabulary and a different resize filter, and if it is ever wanted it
becomes `siglip_deploy` with its own gate — the way `yolov8` and `yolov11` are two packages.

## What mozo guarantees, exactly

**mozo's numbers equal the published PyTorch model's, bit for bit.** `tools/verify/siglip2.py`
proves it on every run with `torch.equal` and no tolerance.

**What those checkpoints owe the authors' JAX originals is not stated by anyone, and is not
claimed here.** It is worth being precise, because the obvious citation is wrong:
`models/siglip2/convert_siglip2_to_hf.py` does verify at `atol=1e-3, rtol=1e-3` — but its
`MODEL_NAME_TO_CHECKPOINT_PATH` contains only the two `-naflex` checkpoints, which this package
does not carry. The fifteen fixed-resolution models are converted by
`models/siglip/convert_siglip_to_hf.py`, whose verification block holds expected outputs for eight
SigLIP **1** model names and none for any SigLIP 2. So upstream publishes no JAX-to-PyTorch
agreement, at any tolerance, for a single checkpoint mozo ships.

mozo cannot close that gap without a JAX runtime, and does not pretend to. It claims the link it
can measure.

## The reference is pinned

Parity was established against **torch 2.11.0, torchvision 0.26.0, transformers 5.8.0**. This is
not ceremony: `SiglipImageProcessor` was refactored from a PIL/numpy implementation to a
torchvision backend, which *changed the pixels it produces*. A zero-tolerance gate against a
moving package needs an anchor, and the gate prints these versions on every run.

## What was taken

| from | what |
|---|---|
| `modeling_siglip.py` | the encoder block, the eager attention, the patch embedding, the attention-pooling head, the text tower's forward and pooling |
| `image_processing_siglip.py` and `image_processing_backends.py` | the resize-and-normalise, rewritten in `image.py` |
| `tokenization_siglip2.py` | the normalisation, and the choice of tokenizer class |
| `configuration_siglip.py` and the fifteen `config.json` | the geometry, written out as frozen dataclasses in `config.py` |
| `google/siglip2-base-patch16-224/tokenizer.json` | the vocabulary, re-encoded into `assets/gemma_bpe.json.gz` |

The checkpoint is consumed as Google publishes it, modulo one repack: `tools/fetch/siglip2.py`
reads the safetensors and writes plain tensors. No tensor is altered, renamed, cast or dropped.

## What was deliberately left behind

**The two `-naflex` variants.** Variable resolution through `Siglip2Model` — patch-attention masks,
spatial shapes, a different image tower. A second tower, not a second configuration. They are 20%
of SigLIP 2's downloads and their absence is a decision, not an oversight. Nothing here is in the
way of adding them.

**Training.** The sigmoid loss, the captioning decoder, the self-distillation and masked-prediction
heads, and the temperature's gradient.

**`interpolate_pos_encoding`.** Upstream offers it to run at a resolution the position embedding
was not trained for. This package always runs at the published one, so there is nothing to
interpolate and no branch that can silently take the wrong path.

**`SiglipForImageClassification`.** A supervised head on the vision tower, and a different task.

**Silent truncation.** Upstream's own preprocessing truncates with `eos="sticky"`, keeping the end
marker. Not carried: a prompt shortened without saying so is a different prompt, and only the
caller can decide what to drop. `Tokenizer.__call__` raises with the count and the cap, as
`clip_deploy` does.

## Where this diverges from the reference, and why

### 1. The tokenizer lowercases, and the published config does not ask it to

This is the one place mozo deliberately does something the published `tokenizer_config.json` does
not describe, and it is not cosmetic.

Those files declare `tokenizer_class: GemmaTokenizer` with `do_lower_case: true` — but
`do_lower_case` is not a `GemmaTokenizer` parameter and nothing acts on it. So
`AutoTokenizer.from_pretrained("google/siglip2-…")` returns a tokenizer that **preserves case**,
while `transformers`' own `Siglip2Tokenizer` prepends a `Lowercase` normaliser for what its
docstring calls the *"SigLIP2 training default"*.

The authors settle it. Their demo notebook preprocesses with:

```python
# ‼️ NOTE: SigLIP 2 models work best with lowercase texts
pp_txt = pp_builder.get_preprocess_fn(
    f'lower(key="text")|tok(length={SEQLEN}, model="gemma", bos="no", eos="sticky", key="text")')
```

Measured on `base-224`, `"A Photo Of People"` against the fixture photograph scores **0.0349**
lowercased and **0.0001** as written — a factor of 350. Gating against `AutoTokenizer` would have
pinned the wrong behaviour bit-exactly and shipped a family that silently mis-encodes every
capitalised prompt, with a green gate.

The gate's reference is therefore `Siglip2Tokenizer`.

### 2. The case table is carried, not taken from the interpreter

Lowercasing is not `str.lower()`. `assets/gemma_bpe.json.gz` carries the 1,488 codepoints the
reference folds and what it folds them to, and the tokenizer applies that table.

The reason is that Python's case mappings are a property of the interpreter rather than of this
package. Python 3.10 ships Unicode 13.0.0; the reference's Rust normaliser ships a later one; and
they disagree about **95 codepoints** — the case mappings added for Vithkuqi, Latin Extended-D and
Cyrillic Extended-C in Unicode 14 and 15. In every one of the 95, Python leaves the character alone
and the reference folds it.

This was found by fuzzing the tokenizer against the reference, not by reading, and it is the kind
of bug that would never show up in review: two callers on different interpreters would tokenize the
same phrase differently, and so write different vectors for it into the same index. Since a stored
embedding outlives the process that made it, an interpreter upgrade would silently split an index
in two.

The table costs 20 KB and makes the answer a property of the package. Verified against the
reference on 2,815 phrases, including the 95 disputed codepoints.

### 3. The towers are built and loaded separately

Upstream builds one `SiglipModel` holding both and loads the checkpoint into it whole. This
partitions the state dict by the `vision_model.` and `text_model.` prefixes and loads each tower on
its own. It matters more here than it did for CLIP: Gemma's 256,000-piece vocabulary is 786 MB of a
`base` checkpoint and 1,180 MB of an `so400m` one, so an image-encoding job would otherwise
allocate most of the file for a table it never reads.

The load stays strict within each half. This is a structural difference with no numerical one.

### 4. Vectors leave L2-normalised

Upstream's `get_image_features` and `get_text_features` return unnormalised vectors and leave
normalising to the caller — its own `forward` does it immediately afterwards. mozo normalises
inside the vendor, so a dot product between any two vectors it emits is a cosine similarity with no
convention to document. The gate compares the unnormalised features too, so this is an addition
rather than a change.

### 5. mozo requires RGB where the reference does not convert

`do_convert_rgb` is `null` in all fifteen published configs and the reference reads it as false, so
the reference does not convert a greyscale or RGBA input. mozo's `load_image` guarantees RGB before
the vendor sees anything, so the two agree on every input mozo can produce; `preprocess` requires
what it is handed to be RGB `uint8` and says so rather than guessing.

## Traps

Each of these runs, produces plausible output, and is wrong. Every one is covered by the gate;
seven were confirmed by perturbing the constant and watching the gate fail at the right stage.

- **The image resize runs on `uint8` with `antialias=True`, and the rescale is folded into the
  statistics.** `TorchvisionBackend._fuse_mean_std_and_rescale_factor` multiplies mean and standard
  deviation by `1/rescale_factor` and sets `do_rescale = False`, so nothing is ever divided by 255.
  Measured cost of the plausible alternatives: `(x/255 - 0.5)/0.5` is **5.9e-08** out,
  `antialias=False` is **9.4e-01** out, and resizing the float tensor rather than the uint8 one is
  **3.9e-03** out.
- **`layer_norm_eps` is 1e-6.** torch defaults to 1e-5 and no published config overrides the 1e-6.
  Worth **5.1e-04** on the image features.
- **The activation is `gelu_pytorch_tanh`.** Not the exact erf GELU `nn.GELU()` gives, and not
  CLIP's QuickGELU. Worth **5.5e-03**.
- **The attention-pooling head takes its residual *before* the layernorm.**
  `h = attn(probe,x,x); r = h; h = ln(h); h = r + mlp(h)` — the opposite of every other block here.
  Worth **7.1e-01**.
- **Pooling reads the last slot, not an end marker.** `hidden[:, -1, :]`, whatever is there;
  upstream's own comment says it "may be padding". Nothing like CLIP's argmax over token ids.
  Worth **1.0e+01**.
- **`padding="max_length"` to 64 is not optional.** The model trained on fully padded sequences,
  the text tower attends the padding, and the pooling above is coherent only because of it.
- **The head dimension is not 64.** CLIP fixes it and divides; here `so400m` is 72 and `giant-opt`
  is 96. Nor is the MLP `4 × width`: `so400m` is 1152 → 4304, and `giant-opt`'s *text* tower is
  so400m's rather than its own vision tower's. Seven of the fifteen break one rule or the other.
- **`so400m-384` does not divide evenly.** 384 over a 14-pixel patch is 27 patches and a remainder
  of six pixels, which the convolution strides past and the model never sees. Upstream computes it
  the same way; `resolution % patch == 0` looks like an invariant and is not.
- **`giant-opt`'s towers are asymmetric** — vision 1536×40, text 1152×27 — and its text head
  projects *up* into the 1536-wide space the image tower defines.
- **The text tower is bidirectional** and has a learned 64×width position embedding. No causal
  mask, unlike CLIP; upstream flags it in a comment because it is what a CLIP reader assumes.
- **There is no class token**, and there are no CLIP-style projection matrices — but there are two
  projections. The vision projection *is* the attention-pooling head; the text projection *is*
  `text_model.head`. Adding a `visual_projection` invents weights that do not exist; dropping
  `text_model.head` loses weights that do.
- **The pooling head's attention is `nn.MultiheadAttention` with a fused `in_proj_weight`**, while
  the encoder blocks use three separate projections. Keeping torch's module is not only about
  weight layout: `need_weights` is left at its default `True`, which sends torch down its unfused
  branch, where the query is scaled *before* the matmul rather than the product after it. That is
  CLIP's operator-precedence trap in a second costume, and it also means the gate's
  `attn_implementation` pin does not govern this head.
- **`logits_per_image` is a transpose of `logits_per_text`**, not a second matmul.
- **The word split never happens.** The normaliser replaces every space with `U+2581` *before* the
  pre-tokeniser's `Split(" ")` runs, so byte-pair merging covers the whole caption and merges cross
  word boundaries. CLIP merges within a word; copying that structure gives different ids.
- **There is no prefix space.** `"a photo"` opens with `'a'` (235250), not `'▁a'` (476).
- **Byte fallback is neither complete nor contiguous.** `<0x00>` is id 217, `<0x09>` does not exist
  at all, and a literal `'\x01'` is an ordinary piece at 238213. The table is read out of the
  vocabulary rather than computed from a base offset.
- **249 reserved names match before normalising.** A caption containing the literal text `<eos>`
  carries a control token, and `<unused1>` prefixes `<unused10>` so the longest must win. Upstream
  behaves this way and this reproduces it; `clip_deploy` makes the same call about a literal
  `<|endoftext|>`.
- **The published config's `model_type` is `"siglip"`, not `"siglip2"`.** `AutoModel` builds the
  v1 class from a v2 checkpoint. Only `-naflex` is `"siglip2"`. The gate has to instantiate what
  the config names, or it is not comparing against the reference.

## Numbers

Parity on `tests/fixtures/images/example.jpg` against `transformers` 5.8.0, CPU, `base-224`:

| stage | |
|---|---|
| preprocessed tensor | exact |
| token ids | exact |
| image features | exact |
| text features | exact |
| `logits_per_image` | exact |
| sigmoid probabilities | exact |

Extraction size: ~640 lines across 11 modules.
