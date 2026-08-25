# Mozo v1.0.2: Where the Joints Are

## What's New in v1.0.2

Mozo's fifteenth family, a second way to get joints, and graphs for the ten segmentation
variants that shipped without them.

### 🕺 ViTPose++

Top-down human pose estimation, Apache-2.0, in four sizes — `small` `base` `large` `huge`.

```python
people = mozo.get_model("rfdetr/medium").predict("crowd.jpg", threshold=0.5)
posed = mozo.get_model("vitpose/base").predict("crowd.jpg", detections=people)
posed[0].keypoints[0]      # KeyPoint(id=0, name='nose', x=..., y=..., confidence=...)
```

**It is the first family whose `predict` takes detections as input.** ViTPose has no detector:
it answers a question about a box someone else found. So it takes the full frame *and* the
boxes, and returns those same detections with seventeen COCO joints attached to each.

The full frame is not a convenience. The crop it works from is deliberately larger than the box
— widened to 3:4 and then padded by a further 1.25× — so a 50×140 person becomes roughly a
131×175 crop. A caller who cropped tightly has already thrown away pixels the model wants, and a
wrist just outside the box goes with them.

**It filters nothing.** Hand it detections and it poses all of them, so filtering to people is
the caller's call and stays visible in the caller's code. Hand it an empty set and you get an
empty set back; hand it nothing at all and Python raises, because there is nothing to infer.

Heatmaps are bit-identical to `transformers` on all four variants, and joints land within
4.3e-04 px end to end. `small` `base` and `large` also publish `onnx-fp32`; `huge` does not,
because its graph does not fit in one protobuf. There is no CoreML — it converts and is correct,
but it is not faster than torch on MPS.

### 🤸 RF-DETR keypoint-preview

A second answer to the same question, in one variant, at Roboflow's published operating point of
576×576. `torch-fp32` only.

Its class-id space is its own — background at 0, person at 1 — where the detection variants emit
COCO's sparse ids running to 90.

### 📦 A task that says it must be told where to look

`pose_estimation` is a task type whose models cannot be asked to go and find something. The
registry says so once, and `/predict` grows a repeatable `box` parameter that the test UI offers
for exactly those families. Nothing carries a per-family exception.

### 🎭 ONNX for the segmentation variants

The ten `seg-` variants YOLO11 and YOLO26 shipped in v1.0.1 published `torch-fp32` only. They
now publish `onnx-fp32` as well — a graph with two outputs, the rows and the prototypes their
mask coefficients belong to.

They were held back because the export gate could compare boxes, scores and class ids but not
masks, and publishing on three quarters of the answer would have reported an agreement it had
not checked. The gate now compares masks by per-detection overlap. Worst figures across the ten:
1.83e-03 px on boxes, 2.00e-04 on scores, 0.999779 IoU on masks.

Still no CoreML for those two families, and the mask branch does not change that — the `seg-`
checkpoints are the same backbone with a different head, so they inherit the same Metal compiler
abort their detection counterparts hit.

### 📖 Documentation

`docs/models.md` gains a ViTPose section covering the crop geometry, the DARK decoding and the
mixture-of-experts backbone, and states what each YOLO generation publishes.

---

# Mozo v1.0.1: Run a Graph, Not Just a Model

## What's New in v1.0.1

Until now mozo answered one question at a time: load a model, hand it an image, get a result.
A workflow is the other half — read an image, run a family, crop what it found, run a second
family on each crop, draw the answer, write it out — as a graph you can save, version, post over
HTTP, or run from the command line.

That is what the major version is for. Nothing about `get_model()` changes, and nothing in the
0.7.x catalogue moves; what changed is that the catalogue is now something you can compose.

> **PyPI goes 0.7.1 → 1.0.1.** v1.0.0 was tagged, but its release run failed the test gate
> before it published anything, so nothing reached PyPI. Everything below ships here for the
> first time.

### 🔀 Workflows

A workflow is a graph of nodes. 38 of them ship, across ten categories — reading an image,
seven detectors, segmentation, two classifiers, depth, thirteen transforms, three exposure
adjustments, eight annotators and a writer.

```python
from mozo.workflow import Workflow

flow = Workflow.from_dict(saved)
result = flow.run(image="street.jpg")
```

**Validation is construction.** `from_dict` refuses a graph rather than failing halfway through
one: an unknown node type, a wire between mismatched ports, an input nothing feeds, a cycle.
A workflow that builds is a workflow that can run.

**Batching lives in the node.** A list on an input port fans the node out over it, while
parameters stay shared — so `crop_around_detections` turns one image into one crop per detection
and everything downstream runs per crop, with nothing in the graph saying so.

**Four endpoints**, mounted on the same server that serves `/predict`, in the same process:
`/workflow/nodes` for the catalogue, `/workflow/validate`, `/workflow/run`, and `/workflow/stream`
for per-node progress. Models are shared with `/predict` rather than loaded twice — one
`ModelManager` for the process, so a family a workflow loads is a family `/predict` already has.

**An editor**, served at `/workflow`, and a command line:

```bash
mozo run flow.json --image street.jpg
```

### 🎭 Instance Segmentation on Two More Families

Ten new variants, all boxes-plus-masks, all sitting beside their detection counterparts under the
same `object_detection` task type because what they add is a field rather than a different
question:

- **YOLO26** `seg-nano` … `seg-xlarge`
- **YOLO11** `seg-nano` … `seg-xlarge`

A mask is a boolean at the source image's resolution, one per detection, aligned with the box
beside it. Both families publish `torch-fp32` for these.

### 📊 Model Count

**73 published variants** across fourteen families, up from 63. Of those, 36 are Apache-2.0,
30 are AGPL-3.0 (every YOLO variant), 4 are MIT (CLIP), 2 are CC-BY-NC-4.0 (Depth Anything
`base` and `large`) and 1 carries Meta's SAM License.

### ✅ Verification

**YOLO11 and YOLO26 are now checked against `ultralytics` itself**, stage by stage, rather than
only against themselves — `tools/verify/yolov11_reference.py` and its YOLO26 counterpart stand the
published implementation up beside the vendored one and compare preprocessing, every layer, the
head, the prototypes, the mask coefficients and the assembled masks. 335 comparisons across
YOLO11's ten variants, every one within tolerance, with three of the five segmentation variants
pixel-identical to the reference.

Both gates are falsifiable and have been made to fail on purpose: perturbing a constant moves the
stages that constant reaches and no others, with a control that provably moves nothing.

### 🐛 Fixes

- **One model manager per process.** The server built its own, so a family reached through
  `/predict` and through a workflow loaded twice, was resident twice, and `/models/loaded`
  reported only half of what was in memory.
- **The YOLO runtime seam states its assumption.** It takes the first output of a graph, which is
  right for every graph published today and wrong for a multi-output one; it now refuses rather
  than misreading.
- **The export gate refuses what it cannot check** instead of comparing a graph against the module
  on three quarters of the answer.
- **The suite runs without weights again.** Three workflow tests asked the manifest whether a model
  was published and then ran it, which is a different question: run through a workflow, a missing
  checkpoint surfaces as a failed node rather than an exception, so a machine without the weights
  failed instead of skipping.
- **Route introspection survives FastAPI 0.137**, which stopped copying an included router's routes
  into `app.routes` and started listing the router itself.

### 📖 Documentation

`docs/workflows.md` covers what a workflow is, how validation and batching behave, and what each
node category is for. `docs/models.md` gains the segmentation sections.

---

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
