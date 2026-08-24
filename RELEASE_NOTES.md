# Mozo v0.7.1: Decide What You Serve

## What's New in v0.7.1

A server no longer has to offer everything mozo publishes. `MOZO_ENABLE` narrows a deployment to
the models you choose, which matters because the weights are separate works with their own terms:
of the 63 published variants, 20 are AGPL-3.0, 2 are CC-BY-NC-4.0 and 1 carries Meta's SAM
License, and serving those over a network places obligations on you that serving the other 40
does not.

> **PyPI goes 0.6.0 → 0.7.1.** 0.7.0 was written up and committed but never tagged or
> released, so everything below ships here for the first time.

### 🔒 `MOZO_ENABLE`

One environment variable, alongside `MOZO_CACHE` and the rest. Name a family or a single variant,
comma-separated, mix the two freely:

```bash
MOZO_ENABLE=clip,siglip2/base-224 mozo start
```

Unset offers everything, so nothing changes for anyone who does not set it.

**An allow-list, not a deny-list.** A deny-list naming today's AGPL families would serve whatever
lands tomorrow, silently. An allow-list serves nothing it was not told to.

**A name that matches nothing warns rather than refusing to start.** An unrecognised token can
only ever subtract, so a typo yields a server missing a model, never one serving something
unsanctioned.

**A refusal never names a model the server declined to offer**, so a narrowed catalogue stays
narrowed at every wrong turn rather than answering it with a menu of what you configured it to
decline.

**The refusal costs nothing.** It is the first thing `/predict` and `/encode` do — before the
image decode, before the download — so a model excluded on licence grounds is never fetched, and
its weights never reach the disk.

**Server-side only.** `mozo.get_model()` is untouched.

For a permissively licensed demo, the README carries the line that offers the 40 Apache-2.0 and
MIT variants and nothing else. Depth Anything is named variant by variant because it is the one
family whose licence is not uniform — seven of its nine are Apache-2.0 and two are CC-BY-NC-4.0 —
which is why the variable takes `family/variant` at all.

### 🔢 SigLIP 2 Carries Five Variants

mozo registered fifteen fixed-resolution SigLIP 2 models and shipped weights for three, so twelve
of them answered a request with a 500. Now five are registered and five are published:
`base-224`, `base-256`, `so400m-384`, `so400m16-256` and `giant-384` — together 89% of the
downloads across those fifteen. The other ten need a checkpoint and no new code.

### 🐛 Fixes

**The test page said "Cannot reach the server" when a deployment offered no models.** It now says
which problem it actually is.

### 📊 Model Count

**Total Models:** 63 variants across 14 model families
**Growth:** +2 variants (SigLIP 2 3 → 5), no new families

Of the 63, **36 are Apache-2.0**, 20 are AGPL-3.0 (every YOLO variant), 4 are MIT (CLIP), 2 are
CC-BY-NC-4.0 (Depth Anything `base` and `large`), and 1 carries Meta's SAM License.

### ✅ Verification

**SigLIP 2: 160 comparisons across all five variants, every one bit-identical** — pixels, token
ids, image features, text features, logits and sigmoid, at exact equality with no tolerance.

### 📖 Documentation

The README's model catalogue is restructured into per-family tables — every variant, what it
returns, and what its weights are licensed under — and the licence section now says what to do
about the terms rather than only stating them.

---

# Mozo v0.6.0: Neither Box Nor Map

## What's New in v0.6.0

The first two families that answer with neither a box nor a map. Name the classes in words and
you get a score for each; ask for the vectors instead and you get the shared space the score was
computed in. No training, no labelled data, no fixed class list.

> **PyPI goes 0.4.0 → 0.6.0.** v0.5.0 was written up and committed but never tagged or
> released, so everything in its notes — Grounding DINO included — ships here for the first
> time.

### 🎉 New Model Families

**CLIP:** Zero-shot classification and the embeddings behind it, by OpenAI
- 4 variants: `base`, `base-16`, `large`, `large-336`, all Vision Transformers
- Phrases up to 77 tokens each; a score per phrase, or 512/768-d vectors
- MIT, code and weights

The five ResNet variants are not carried: they replace the Vision Transformer with a modified
ResNet and attention pooling, which is a second image tower rather than a second configuration.

**The two towers build and load separately.** An ingest job that only encodes images never
allocates the text tower, so a query service holds 63.4M parameters rather than 151.3M.

```python
model = mozo.get_model("clip/base")
scored = model.predict("aisle.jpg", ["a forklift", "a person", "an empty aisle"])
scored[0].class_name        # 'a forklift'
scored[0].confidence        # a cosine similarity -- see below
```

**Scores are cosine similarities, not probabilities.** They are not softmaxed, so they do not sum
to one, they may be negative, and adding a phrase does not move the others — which is exactly
what makes a threshold calibratable. They are also compressed: a good match sits far below 1.0,
so 0.31 does not mean "31% sure". Softmax it yourself if you want CLIP's published closed-set
behaviour.

**SigLIP 2:** The same question, answered as a probability, by Google
- 3 variants published: `base-224`, `so400m-384`, `giant-384`
- Phrases up to 64 tokens each; a probability per phrase, or 768/1152/1536-d vectors
- Multilingual
- Apache-2.0 throughout, ungated, on the code and on every checkpoint

**The score is what makes it worth having beside CLIP.** SigLIP was trained pair by pair with a
sigmoid loss and carries a learned bias as well as a temperature, so `predict` returns a
probability for one image and one phrase on its own. Asking about a single phrase is a well-posed
question, and every phrase can be near zero when none of them fits. A cosine similarity can
express neither. It is still not a calibrated class probability, and the docstring says so.

**No `transformers` dependency**, and none on `tokenizers`, `sentencepiece` or `safetensors`
either. The Gemma vocabulary ships in the wheel — Google publishes that same tokenizer ungated
under Apache-2.0 inside every SigLIP 2 repository, which the NOTICE explains, since it looks
alarming until checked.

### 🆕 New Endpoint: `/encode`

Some models represent an image and a phrase in one shared space, so a dot product between two
vectors says how well they match. That is what makes a corpus embedded once searchable by words
afterwards — but only through a vector database, which is yours. mozo produces the vectors and
stops there.

```bash
curl -X POST "http://localhost:8000/encode/clip/base" -F "file=@aisle.jpg"
curl -X POST "http://localhost:8000/encode/clip/base?text=a%20forklift&text=a%20person"
```

Send either images or phrases, not both: they are two towers and one call runs one of them.
Repeat the part or the parameter for a batch. The response carries `model`, `revision`, `dim` and
`embeddings`, and the revision is read off the loaded model rather than re-derived — a vector
stamped with the wrong revision is undetectable afterwards.

Refusals are answered before the image decode and before the download, so naming a family that
does not embed never costs a multi-gigabyte fetch.

### 🐛 Fixes

**An installed mozo raised the moment anything encoded a phrase.** SigLIP 2 reads its tokenizer
vocabulary from a file that was missing from the wheel. Every vendored package's LICENSE, NOTICE
and PROVENANCE were missing too, for all four YOLO extractions and SigLIP 2.

**CLIP rendered with no text box on the test page**, so the page could not drive it. The page had
kept a hand-written copy of which tasks take a prompt; it now reads that from `/models`.

### 📊 Model Count

**Total Models:** 61 variants across 14 model families
**Growth:** +7 variants (CLIP 4, SigLIP 2 3), +2 families

Of the 61, **34 are Apache-2.0**, 20 are AGPL-3.0 (every YOLO variant), 4 are MIT (CLIP), 2 are
CC-BY-NC-4.0 (Depth Anything `base` and `large`), and 1 carries Meta's SAM License. MIT is a
licence mozo had not shipped before.

### ✅ Verification

**CLIP: 104 comparisons across all four variants, every one bit-identical** against
`openai/CLIP` — the authors' own repository, which is what the published checkpoints reproduce.
Preprocessing, token ids, image features, text features, cosine similarities and
`logits_per_image`, at exact equality rather than a tolerance.

**SigLIP 2: 96 comparisons across the three published variants**, bit-identical at every stage —
pixels, token ids, image features, text features, logits and sigmoid.

### ⏱️ Benchmarks

SigLIP 2 against `transformers` — same checkpoint, same photograph, same device, so any gap is
the extraction rather than a comparison between models:

- **4% slower on CPU, 11% faster on MPS**, both from the same eager-attention choice, where the
  fused path is the slower one on Metal
- **Batching wins 1.6x at 224 and nothing at 384** — one image already saturates the device
- The text tower is a quarter of a classify call rather than a rounding error, because the
  context is always padded to 64

### 📖 Documentation

`docs/models.md` — detailed notes for all 14 families: what each one answers, what it costs, and
what its weights are licensed under.

---

# Mozo v0.5.0: Describe It In Words

## What's New in v0.5.0

### 🎉 New Model Family

**Grounding DINO:** Open-vocabulary detection by IDEA Research
- 2 variants: `tiny` (Swin-T, 48.4 box AP COCO zero-shot) and `base` (Swin-B, 56.7)
- Both checkpoints upstream publishes; there is no third
- Apache 2.0 code *and* weights

It answers the same question OWLv2 does, through the same endpoint and the same response shape,
so the two are substitutable. Reach for it when the prompt is a *description* rather than a noun:
OWLv2 embeds each phrase independently, while Grounding DINO fuses the text into the image
features six times over and lets the decoder attend back to the words, so `"the mug on the left"`
is read as a phrase rather than a bag of words.

```python
model = mozo.get_model("grounding_dino/tiny")
found = model.predict("kitchen.jpg", ["a kettle", "the mug on the left"])
found[0].class_name        # 'a kettle' -- the phrase you asked for
```

**No `transformers` dependency**, despite the BERT text encoder: the published checkpoint carries
its own fine-tuned BERT tower, and the WordPiece vocabulary ships in the wheel. Tokenizing a
prompt needs no network.

**Torch only, and that is structural.** The image is resized rather than letterboxed, so there is
no fixed input size and no graph to export. `runtime="auto"` will not offer one.

### 📊 Model Count

**Total Models:** 54 variants across 12 model families
**Growth:** +2 variants (Grounding DINO), +1 family

### ✅ Verification

138 comparisons against the authors' own implementation, both variants, exact equality with no
tolerance — preprocessing, token ids, the BERT tower, all three Swin levels, encoder memory and
each of the six decoder layers, not just the final boxes.

---

# Mozo v0.4.0: Real-Time Detection

## What's New in v0.4.0

### 🎉 New Model Family

**RF-DETR:** Real-time transformer-based object detection and instance segmentation by Roboflow
- 8 variants: nano, small, medium, large + seg-nano, seg-small, seg-medium, seg-large
- Pretrained COCO models with fine-tuned checkpoint support
- Apache 2.0 license

### 📊 Model Count

**Total Models:** 63 variants across 10 model families
**Growth:** +8 variants (RF-DETR), +1 family

### 🔧 Improvements

**Automatic device detection:** CPU/GPU (CUDA)/Apple MPS auto-selection integrated into
`get_model()` and ModelFactory

**Unified image loading:** all adapters accept both file paths and numpy arrays as input

**Simplified API:** a top-level `get_model()` for direct access with a shared ModelManager

### 🗑️ Removed

- **StabilityInpainting** adapter removed (cloud-based inpainting service discontinued)
- **Datamarkin** adapter removed (the hosted Vision Service is no longer available)
- **SAM3** adapter removed — it was never registered and therefore unreachable; it will
  return once implemented properly
