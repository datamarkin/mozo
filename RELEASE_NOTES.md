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

Upstream publishes TorchScript archives. These are repacked to plain state dicts and cast to
fp32 — a scripted archive is a serialised graph that a future torch may refuse, so the version
risk is taken once, on a machine we control, rather than on your machine years from now. The
download is verified against the sha256 in OpenAI's own URL, which serves these
content-addressed, so the digest is the file's rather than one transcribed from a README.

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
- 3 variants published: `base-224`, `so400m-384`, `giant-384` — the other twelve are one run away
- Phrases up to 64 tokens each; a probability per phrase, or 768/1152/1536-d vectors
- Multilingual
- Apache-2.0 throughout, ungated, on the code and on every checkpoint

Three rather than fifteen because between them they cover every distinct code path: the patch-14
grid that floors, head dimensions of 64, 72 and 96, asymmetric towers, and the sharded download
the two giant checkpoints need. The other twelve exercise nothing new.

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

**Every refusal is answered from the registry**, before the image decode and before the download.
`/predict` can afford a late 501 because a test proves every registered task has an arm there;
here the reverse holds, and a late refusal would fetch a multi-gigabyte checkpoint in order to
say no.

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
`logits_per_image`. Exact equality, not a tolerance: both divergences found while building the
package would have been swallowed by any sane one.

Shown to be falsifiable rather than assumed. Substituting `nn.GELU` for QuickGELU fails 5
comparisons at both towers and nothing else; squashing the resize instead of scaling the short
side fails preprocessing alone; scaling the logits after the matmul rather than before fails only
the logits, at exactly 1.907e-06 — which is the whole difference between this passing and
failing.

**SigLIP 2: 96 comparisons across the three published variants**, `torch.equal` at every stage —
pixels, token ids, image features, text features, logits, sigmoid. Four things are pinned or it
means nothing: the CPU, eager attention, the reference's parameters re-allocated first (BLAS
picks different paths for page-aligned storage), and `Siglip2Tokenizer` rather than
`AutoTokenizer` — gating against the wrong one would have shipped a family that mis-encodes every
capitalised prompt, with a green gate. Shown to work by breaking the model eight ways; every
perturbation was caught, each at the stage it belongs to.

### ⏱️ Benchmarks

SigLIP 2 against `transformers` — same checkpoint, same photograph, same device, so any gap is
the extraction rather than a comparison between models:

- **4% slower on CPU, 11% faster on MPS**, both from the same eager-attention choice the gate
  requires, where the fused path is the slower one on Metal
- **Batching wins 1.6x at 224 and nothing at 384** — one image already saturates the device
- The text tower is a quarter of a classify call rather than a rounding error, because the
  context is always padded to 64

Every measurement drains the device first. Metal queues asynchronously, so an unsynchronised
timer measures submission — without this, mozo reported 23% *slower* on MPS purely because
`encode_image` ends in `.cpu()` and the reference path did not.

### 🔧 Infrastructure

**`zero_shot_classification` joins `PROMPTED`.** It shares the prompt contract with the detectors
and nothing else: no boxes, and every phrase comes back scored rather than only the ones that
hit. A classifier that drops a class has not classified.

**`ENCODES` records what each family can embed and from what**, so `/models` can say which without
loading anything. A dict rather than a set, because the kinds differ — CLIP takes both, a
re-identification embedder would take images only. The task type cannot carry this: two families
can classify while only one of them embeds.

**The catalogue's capability fields are now guarded.** A family promising `"image"` whose adapter
has no `encode_image` used to answer 500 *after* fetching gigabytes — the one refusal that costs
something. The adapter class is imported, not instantiated: the method has to exist, not run.

**The test page reads the prompt rule from `/models`.** It kept a hand-written copy of `PROMPTED`
that had already gone stale — it never learned about `zero_shot_classification`, so CLIP rendered
with no text box and the page silently could not drive it. `/models` now reports `prompted`
beside `encodes`, and the page reads both off the catalogue it already fetches.

### 📦 Packaging

**Every vendored package's terms now ship in the wheel.** `package-data` was a hand-written list
with one entry per family, and it had gone stale for five of the fourteen: all four YOLO
extractions and SigLIP 2 shipped their code with no LICENSE, no NOTICE and no PROVENANCE. One
rule over every package replaces it, so a family added tomorrow needs no packaging edit and
cannot be forgotten the way these five were.

It also covers `assets/`, which is not only a licensing matter — SigLIP 2 reads its tokenizer
vocabulary from there at construction, so without it an installed mozo raised the moment anything
encoded a phrase.

**Releases are automated.** Pushing a `v*` tag builds, checks and publishes through PyPI Trusted
Publishing — no API token in the repository or its secrets. The tag must match `mozo.__version__`
and the version must be unclaimed on PyPI, or the run stops before building. `tools/check_dist.py`
then holds both the wheel and the sdist against the working tree: every run-time file the package
reads must be in both, derived by rule rather than from a list, because a list is what went stale
above.

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
tolerance. Every intermediate is hooked — preprocessing, token ids, the BERT tower, all three
Swin levels, encoder memory, and each of the six decoder layers — not just the final boxes, since
two implementations can agree on the last tensor and disagree in the middle.

### 🔧 Infrastructure

**Stated counts are now tested.** `tests/test_stated_counts.py` holds the package docstring and
the README's model counts against the manifest. The docstring had said "Seventeen published
variants across two families" for eight families longer than it was true.

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

### 🔧 Infrastructure Improvements

**Automatic device detection:** CPU/GPU (CUDA)/Apple MPS auto-selection integrated into `get_model()` and ModelFactory

**Unified image loading:** New `load_image` utility allows all adapters to accept both file paths and numpy arrays as input

**Simplified API:** New top-level `get_model()` function for direct access with shared ModelManager instance

### 🗑️ Removed

- **StabilityInpainting** adapter removed (cloud-based inpainting service discontinued)
- **Datamarkin** adapter removed (the hosted Vision Service is no longer available)
- **SAM3** adapter removed — it was never registered and therefore unreachable; it will
  return once implemented properly

## Installation

```bash
pip install --upgrade mozo
mozo start
```

## Links

- PyPI: https://pypi.org/project/mozo/0.4.0/
- Documentation: https://github.com/datamarkin/mozo
- Issue Tracker: https://github.com/datamarkin/mozo/issues
