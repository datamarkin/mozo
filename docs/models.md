# Model notes

What the tables in the [README](../README.md) cannot say: why a family behaves the way it does,
what it costs, and which of its behaviours are deliberate rather than incidental.

Every claim about parity here is recorded in more detail in the family's
`mozo/vendors/<family>_deploy/PROVENANCE.md`, which also names the exact upstream commit the
extraction came from.

## RF-DETR

Real-time transformer detection and instance segmentation by Roboflow. NMS-free — the head
predicts one box per query, so there is no overlap threshold to tune.

Every variant publishes torch, ONNX and CoreML, and they are verified to return the same
detections:

```python
model = mozo.get_model("rfdetr/small")                            # torch
model = mozo.get_model("rfdetr", "small", runtime="onnx-fp32")    # ONNX Runtime
```

Class names ship with the weights, so they are the vocabulary the checkpoint was trained on
rather than an assumption. A checkpoint of your own that carries no names returns `class_id`
with `class_name` unset — pass `labels=[...]` to name them. Mozo never guesses a name.

**`keypoint-preview` returns person joints.** One variant, because Roboflow published one
operating point: 576×576, and 71.8 keypoint AP on COCO at 9.8 ms on a T4 under TensorRT FP16 by
their measurement.

```python
model = mozo.get_model("rfdetr/keypoint-preview")
found = model.predict("crowd.jpg", threshold=0.5)
found[0].keypoints[0]      # KeyPoint(id=0, name='nose', x=..., y=..., confidence=...)
```

Seventeen joints per detection in COCO's order, each `(x, y, confidence)` in source-image pixels,
named from the published `labels.json` the same way class names are. **A joint the model cannot see
still occupies its slot** — it comes back with a confidence near zero and coordinates that mean
nothing, so filter on the confidence before reading a position. That is upstream's behaviour and it
is reproduced rather than tidied: dropping the invisible joints would renumber the rest, and the
index *is* the joint's identity.

Two things differ from its siblings. Its class-id space is its own — a two-slot head, background at
0 and person at 1 — where the detection variants emit COCO's sparse ids running to 90. And it
publishes `torch-fp32` only, so `runtime="auto"` gives you torch and asking for ONNX says so.

The head also predicts a per-joint precision-Cholesky, an uncertainty ellipse rather than a score.
Mozo does not return it: there is nothing in a `Detections` that carries one, and reducing it to a
single number the model never predicted would be worse than leaving it out.

## YOLOv8, YOLO11, YOLO12, YOLO26

Real-time detection by Ultralytics. Five sizes each: `nano` `small` `medium` `large` `xlarge`.
YOLO11 and YOLO26 add five instance-segmentation variants apiece, `seg-nano` … `seg-xlarge`.

**YOLO26 is NMS-free.** Its head fires once per object and the network returns a ranked detection
list, so there is no overlap threshold to tune. The other three suppress in the usual way. Either
way `predict` takes a confidence threshold and returns the same PixelFlow result.

**YOLO11 and YOLO26 also segment.** Five `seg-` variants each, sitting beside the detection ones
in the same family under the same task type, because what they add is a field rather than a
different question:

```python
model = mozo.get_model("yolov11/seg-nano")
found = model.predict("desk.jpg", threshold=0.25)
found[0].masks          # a boolean array at the source image's resolution
```

A mask is a yes-or-no per pixel, one per detection, aligned with the box beside it. The `seg-`
variants currently publish `torch-fp32` only, so `runtime="auto"` gives you torch for them and can
still give you a graph for their detection counterparts.

Two behaviours are upstream's and are reproduced rather than tidied away. A detection whose mask
comes out empty is dropped, so a `seg-` variant can return fewer objects than its detection
counterpart on the same photograph. And the mask crop takes a different branch below 50
detections, rounding box edges to whole pixels where the other compares against the unrounded
float — so the mask you get depends slightly on how many objects are in the picture.

**Runtimes differ across the four generations.** YOLOv8 and YOLO12 publish a CoreML artifact,
which is by far the fastest way to run them on Apple silicon. YOLO11 and YOLO26 do not: the
`C2PSA` block they share makes Apple's Metal graph compiler abort the process, and the
configuration that avoids that is slower than torch on MPS. `runtime="auto"` handles this by
itself — it only ever chooses among what a variant actually publishes, so nothing in mozo carries
a per-family exception.

Class names come from the checkpoint, so a fine-tuned model publishes its own vocabulary.

**Licensing.** These weights are AGPL-3.0, or covered by a commercial licence from Ultralytics.
Mozo's own code is Apache-2.0 — they are separate works travelling together — but anything you
export from them inherits their terms, and serving predictions from them over a network places
AGPL-3.0 section 13 obligations on you. Complying is the operator's responsibility.

## OWLv2

Open-vocabulary detection. Name anything in words and it returns boxes for it — no class list, no
fine-tuning, no vocabulary agreed in advance.

`base-ensemble` and `large-ensemble` average Google's self-trained and fine-tuned checkpoints and
are what the paper reports; `base` and `large` are self-training only.

**This is mozo's permissively-licensed way to ask a model a question in words.** SAM 3 answers a
similar question with masks, but its weights carry Meta's SAM License and bind whoever you serve
predictions to. OWLv2 is Apache-2.0 on the code and on all four checkpoints.

Boxes only, no masks — pair it with SAM 2 or EdgeTAM if you want those: a box from here is a
prompt there.

Phrases go in verbatim, up to 16 tokens, and all of them share one image forward *and* one text
forward, so twenty phrases cost barely more than one. Scores are similarities through a sigmoid
rather than class probabilities, so they run low: 0.3 is confident here, and the default floor is
0.1 rather than the 0.5 the closed-vocabulary detectors use.

**There is no non-maximum suppression, because the published postprocessing has none.** A
detection is a patch: the model scores every patch against every phrase and predicts one box per
patch, so overlapping boxes on a large object are expected, and suppressing them is the caller's
policy rather than something mozo decides.

Inference is a fixed square — 960 for base, 1008 for large — reached by padding the image bottom
and right to a square first, so the aspect ratio *is* preserved and a 4:3 photograph spends a
quarter of its patches on padding. On CPU at 2 MP, `base-ensemble` is about 1.1 s, which is
within 1% of `transformers` running the same weights; the image encode is cached, so a second
vocabulary on the same photograph is about 50 ms.

Verified against `transformers` on every stage — tokenizer, preprocessing, both towers, all three
heads, and the final boxes — with no tolerance: 226 comparisons on `base-ensemble`, every one
bit-identical. `tools/verify/owlv2.py` needs neither a checkout nor a network, only
`pip install transformers` and the weights.

## Grounding DINO

Open-vocabulary detection, and mozo's second answer to the same question OWLv2 answers — same
task, same endpoint, same response shape, so the two are substitutable.

**Reach for it over OWLv2 when the prompt is a description.** OWLv2 embeds each phrase
independently and compares it to patch embeddings, which works well for nouns. Grounding DINO
fuses the text into the image features six times over and lets the decoder attend back to the
words, so `"the mug on the left"` is read as a phrase rather than a bag of words.

**It costs more, and how much more depends entirely on whether you re-query one image.** On the
fixture photograph, MPS, three prompts:

| | first look at an image | asking the same image again |
|---|---|---|
| `owlv2/base-ensemble` | 458 ms | **13 ms** |
| `grounding_dino/tiny` | 472 ms | 472 ms |
| `grounding_dino/base` | 656 ms | 656 ms |

Cold, they are comparable — 1.0x and 1.4x. Warm, OWLv2 is 40x faster, and that gap is
architectural rather than an optimisation anyone could port. OWLv2's towers are independent, so
its image encode depends only on the image and `owlv2_deploy` caches it; a second prompt against
one photograph costs almost nothing. Grounding DINO fuses text into the image features, so its
image encode depends on the prompt and there is nothing cacheable to keep. **If you ask one image
many questions, that is the whole comparison.**

Two variants, and upstream publishes no others. `tiny` (Swin-T) is 48.4 box AP on COCO zero-shot;
`base` (Swin-B) is 56.7 for 35% more weights. `tiny` is 82% of upstream's own release downloads,
but 8.3 AP is a wide enough gap that `base` is not merely the completionist option.

**Prompts have three rules.** They may not contain `.` or `?` — those separate concepts, and a
prompt carrying one would be split in two and its detections reported against the wrong phrase, so
it is refused. Case is not preserved: the caption is lowercased before tokenization, as upstream
does. And about 60 prompts fit the model's 256-token budget; upstream truncates past it silently,
mozo raises.

**The name you get back is the phrase you asked for.** Upstream decodes the matching tokens back
into a string, which can return `"yellow school"` for `"a yellow school bus"`. mozo instead reports
which prompt matched — exactly, from the phrase map the tokenizer already builds — so `class_name`
is your string and `class_id` is its index in the list you passed. This is the one place the
package deliberately differs from upstream, and it is a difference in the string rather than in any
number. `PROVENANCE.md` has the reasoning.

**Nothing is suppressed**, as with OWLv2. Two prompts describing the same thing can both find it.

**Torch only, and that is structural.** The image is resized to a short side of 800 with the long
side capped near 1333 — aspect preserved, nothing padded — so unlike every other family here there
is no fixed input size. A graph exported at one shape cannot take another, so no ONNX or CoreML
artifact is published and the adapter declares `EXECUTES = ("torch",)` so `auto` cannot pick one.

**No `transformers` at run time**, despite the BERT text encoder. The published checkpoint carries
its own fine-tuned BERT tower under `bert.*` — 200 tensors, most of the 694 MB — and upstream's
download of `bert-base-uncased` is only ever used for its shape. mozo rebuilds that shape and
ships the WordPiece vocabulary in the wheel, so tokenizing a prompt needs no network.

Verified against the authors' own implementation on every stage — tokenizer, preprocessing, the
BERT tower, all three Swin levels, the encoder, all six decoder layers, and the final logits and
boxes — with no tolerance: 138 comparisons across both variants, every one bit-identical.
`tools/verify/grounding_dino.py` needs a checkout of upstream and `transformers<5`.

## SAM 2 and EdgeTAM

Promptable segmentation: point at something and get back the thing you pointed at.
`sam2/{tiny,small,base_plus,large}` and `edgetam/edgetam`.

Every prompt is a set of points. A click is a point with a label — `1` to include, `0` to
exclude. A box is spelled as its two corners carrying reserved labels, because neither model has
a separate box input; the adapter writes those for you. Points and a box can be combined.

```python
model = mozo.get_model("edgetam/edgetam")

found = model.predict(image, points=[[820, 640]], labels=[1])   # three candidates, best first
found = model.predict(image, boxes=[40, 60, 300, 480])
found = model.predict(image, boxes=[40, 60, 300, 480], points=[[900, 700]], labels=[0])
```

`multimask_output=True` (the default) returns three candidate masks with the model's own
predicted IoU as the score, ranked. That is the right setting for a single click, which is
genuinely ambiguous about whether you meant the handle, the door or the car — take the first row,
or show all three. With a box the prompt is usually unambiguous and `multimask_output=False` is
tighter.

**Detections come back with `class_name=None`.** A click does not say what it clicked, and mozo
will not invent a name for it — a name comes from the weights or from the user. Pass `name="cat"`
if you know what you pointed at. `class_id` is the index of the prompt that produced the row, so
a batch of prompts stays separable.

The image encoder is the cost and it depends only on the image, so it is cached on pixel content
and a second prompt on the same photograph pays only for the decoder. On CPU at 2 MP: EdgeTAM
encodes in 272 ms and decodes in 33 ms; SAM 2 tiny encodes in 439 ms.

EdgeTAM is SAM 2 distilled for phones — a 9.1M-parameter image path against SAM 2 tiny's 31.4M —
and its masks agree with SAM 2 tiny's at 0.94 IoU on box prompts.

Unlike SAM 3, both families' published weights are Apache-2.0, the same as their code.

Only the torch runtime is served so far. SAM 2 also publishes ONNX and CoreML artifacts, which
this adapter refuses rather than quietly answering with torch: a promptable model exports as
several graphs and needs a runner that keeps the encode and the decode apart.

## SAM 3

Meta ships a single model rather than a size ladder, so there is one variant: `sam3/sam3`.

Two ways to prompt it, off one checkpoint:

- **Name a concept** — `predict(image, "taxi")` returns every instance, with a mask, a box and a
  score. The phrase you searched for is the class name every detection carries, so there is no
  fixed vocabulary and nothing for mozo to guess. Pass a list — `predict(image, ["car",
  "person", "dog"])` — and you get one result carrying several classes, with `class_id` indexing
  the prompts. Instances found by different prompts may overlap: ask for `"car"` and `"vehicle"`
  and the same car comes back under both names.
- **Point at one thing** — `Segmenter.segment(image, points, labels)` takes clicks (`1` include,
  `0` exclude), a box, or a previous mask to refine, and returns three candidate masks with
  predicted IoU. Reached through `mozo.vendors.sam3_deploy` rather than the adapter for now.

Prompts are up to 32 tokens. Inference is a fixed 1008×1008 square — SAM 3 squashes rather than
letterboxing, so aspect ratio is not preserved.

The model wants a GPU. On Apple silicon MPS the image encoder is about 1.2 s and on CPU about
5 s; the encode is cached, so further prompts on the same image cost only their own decode.
Concepts do not batch — the head takes one prompt at a time — so N concepts cost one encode plus
N decodes: on MPS, three concepts is about 2.1 s cold and 0.35 s each afterwards. Encoded prompts
are cached too (33 KB each), so the same three words on the next image skip the text tower
entirely.

**Licensing — read this before deploying.** These weights are **not** open source. They carry
Meta's **SAM License**, which no other family here does. It restricts what they may be used for —
military, nuclear, espionage and weapons uses are prohibited — and those restrictions flow
through to whoever you serve predictions to. It binds on use rather than on signing, and it must
travel with the weights if you pass them on. Mozo's own SAM 3 code is Apache-2.0 and derived from
`transformers`, not from `facebookresearch/sam3`; the code and the weights are separate works
travelling together. Complying is the operator's responsibility.

## EasyOCR

Text recognition. Finds every line of text on a page and reads it.

```python
model = mozo.get_model("easyocr/english")
found = model.predict("sign.jpg")
found[0].text                # 'EXIT 42' — what it says
found[0].class_name          # None — OCR reads content, it does not pick a class
found[0].segments            # the four corners as read, for rotated text
```

**A variant is a script, not a language**: `english`, `latin`, `chinese-simplified`, `japanese`,
`korean`. `latin` alone covers 41 languages and reads every character its charset holds. Upstream
instead picks a checkpoint from a language list and then suppresses characters outside those
languages at decode time, so its output depends on something that is not a property of the
weights — ask mozo's `latin` for `café` and you get `café`.

These five are 88% of upstream's own download counts, out of the seventeen recognisers it
publishes.

**Detections carry `text`, not a class name.** Every other family here names a class from a fixed
vocabulary; this one produces content that belongs to no vocabulary, and PixelFlow keeps the two
apart. `class_id` and `class_name` are `None`.

The quadrilateral is kept in `segments` with its axis-aligned hull in `bbox`, because real-world
text is rotated and a box alone throws the orientation away. **Level lines come back top to
bottom, followed by tilted ones — that is upstream's ordering, and it is not a reading order:** a
two-column page interleaves.

Two graphs: CRAFT locates the text, a CRNN reads it. There is no NMS — a detection is a connected
component of two heatmaps, one scoring "inside a character" and one "between two characters of
the same word". About 200 ms a page on CPU and 31 ms on Apple silicon, within 1% and 22%
respectively of the published package running the same weights on the same device.

**The GPU path is not bit-identical to the CPU one** — strings and quadrilaterals are exact,
confidences move by up to 2.2e-05. The verification below is a CPU claim; pass `device="cpu"` if
you need exactly those numbers.

Verified against `easyocr` at every stage — preprocessed tensor, both heatmaps, quadrilaterals,
each crop, the decoded string and its confidence — with no tolerance: 1,275 comparisons across
the five variants, every one bit-identical. `tools/verify/easyocr.py` needs `pip install easyocr`
and the weights.

### Why no ONNX or CoreML

EasyOCR is the first family in mozo whose input shape is a property of the input, so neither
export is publishable.

The detector's input is the page scaled to fit 2,560 and padded to a multiple of 32, so its shape
follows the photograph. The recogniser's width is `ceil(aspect) × 64` per line, and padding a
line out to a fixed width replicates its last column into more decode steps, which can change
what it says.

With flexible shapes neither graph converts: the detector's U-Net upsamples to another tensor's
runtime shape, and the recogniser hits an adaptive pool that only exports when the input size is
known. At fixed shapes both convert and CoreML is the fastest runtime measured — 13.8 ms against
torch-MPS's 18.8 for the detector — but parity comes out at 1.6e-05 on the recogniser, above the
1.4e-05 already known to flip a character, and enumerating shapes would mean 6,241 combinations
for the detector alone.

## Depth Anything V2

Monocular depth estimation, in two groups that are not interchangeable.

**Relative depth** — output is inverse depth on an arbitrary per-image scale: larger means
nearer, and that is all it means. Two images cannot be compared to each other, and no value is a
distance.

- `small` — fastest, lowest memory (Apache-2.0)
- `base` — balanced (**CC-BY-NC-4.0**, non-commercial)
- `large` — best accuracy (**CC-BY-NC-4.0**, non-commercial)

**Metric depth** — output is in metres, from fine-tunes on Hypersim (indoor, 0–20 m) and Virtual
KITTI 2 (outdoor, 0–80 m). All Apache-2.0 per their model cards.

- `indoor-small`, `indoor-base`, `indoor-large`
- `outdoor-small`, `outdoor-base`, `outdoor-large`

`model.unit` is `"metres"` for the metric variants and `None` for the relative ones — mozo never
guesses a unit.

Output is an `HxW` float32 array at the input's resolution. Over HTTP it is encoded as a 16-bit
PNG with the range in the headers, because six of the nine variants predict metres and quantising
those to 256 levels would discard the measurement.

## CLIP

The first family here that answers with neither a box nor a map, and the only one that will hand
back the numbers it works from rather than an answer.

CLIP is two networks trained together until their outputs land in one shared space: an image tower
and a text tower, both ending in a 512- or 768-dimensional vector. Nothing about that space is
labelled — the image of a forklift and the phrase "a forklift" simply end up near each other. Every
use follows from comparing two vectors with a dot product.

- `base` — ViT-B/32, 512-d. The one almost everyone means by "CLIP"
- `base-16` — ViT-B/16, 512-d. Same size, finer patches, slower and stronger
- `large` — ViT-L/14, 768-d
- `large-336` — ViT-L/14 at 336px, 768-d. The most accurate and by far the slowest

All four are MIT, code and weights. The five ResNet variants OpenAI also publishes use a different
image tower and are not carried; see the vendor's `PROVENANCE.md`.

### Two products, one checkpoint

**`/predict` classifies.** Name your classes in words and each is scored against the image. No
training, no labelled data, no fixed class list — which is what makes it the answer for "is there a
person in this frame" when nobody trained a person detector for your cameras.

**`/encode` returns the vectors.** That is a handoff rather than an answer: a vector is useful only
next to other vectors, which means a store mozo does not provide. The shape of the pipeline is
embed a corpus once, keep the vectors in a vector database, then encode a query phrase and let the
database find the nearest. mozo is the model in that pipeline and nothing else — it does not index,
does not search, and does not persist.

The separation matters because searching a million frames for "forklift near a person" is a
database query over stored vectors, not a million model runs. The model runs once per frame at
ingest, and once per query.

### The scores are cosine similarities, not probabilities

They are not softmaxed. They do not sum to one, they can be negative, and they are compressed —
a good match sits far below 1.0, so 0.31 does not mean "31% sure". Compare them against each other,
or against a threshold you calibrate on your own images.

A softmax would be worse rather than better here: it is relative to whichever phrases you happened
to pass, so one phrase always scores 1.00 and adding a phrase moves every other number. Because
mozo does not softmax, each phrase is scored independently and **adding a phrase does not change
the others' scores** — which is the property that makes a threshold mean anything at all.

### The towers load independently

Asking for phrases never builds the image tower, and vice versa. An ingest job holds the image
tower alone; a query service holds 63.4M parameters instead of 151.3M. Nothing to configure — ask
for what you need and the rest is never built.

### An index is tied to the weights that built it

Both towers come from one checkpoint and were trained together until their outputs agreed. A vector
from `base` means nothing against a vector from `large`, and nothing against a vector from a
different revision of `base`. So a stored index is tied to the variant *and* revision that built
it, and changing either means re-embedding the corpus. `/encode` returns both in the response for
exactly that reason. This is the operational cost of an embedding pipeline and it surprises people.

### Parity

Bit-exact against `openai/CLIP` at commit `d05afc43`, on all four variants: 104 comparisons across
preprocessing, token ids, image features, text features, cosine similarities and `logits_per_image`
— every one identical, with no tolerance.

Two traps that survive a code review and only a numeric gate catches. The activation is
**QuickGELU** (`x * sigmoid(1.702x)`), not `nn.GELU`; substituting the standard one moves the
features by 0.4 and neither raises nor warns. And upstream writes
`logit_scale * image_features @ text_features.t()`, where `*` and `@` share precedence — so the
scale multiplies the *features* before the matmul, not the product after it. Same arithmetic,
different rounding, 1.9e-06 apart.

## SigLIP 2

The second family that hands back vectors, and the answer to the thing CLIP is bad at.

Same shape as CLIP — an image tower and a text tower trained together until their outputs land in
one shared space — but trained with a different loss, and that changes what the number at the end
means. CLIP was trained to *rank* a batch: which caption in these 32,768 goes with this image. So a
CLIP score is only meaningful against the other scores you asked for. SigLIP was trained pair by
pair with a sigmoid loss: does this caption go with this image, yes or no. So its score means
something on its own.

- `base-224` / `base-256` — ViT-B/16, 768-d. The small, fast end
- `so400m-384` — the shape-optimised 400M tower at patch 14, 1152-d. The quality point
- `so400m16-256` — the same tower at patch 16 and a smaller input, so roughly a third the cost
- `giant-384` — a 1536-d image tower paired with the so400m *text* tower. The strongest, and the
  most downloaded of all of them

Apache-2.0, code and weights, ungated. Google publishes fifteen fixed-resolution variants and mozo
carries these five, which took 89% of the downloads across those fifteen; the other ten need a
manifest entry
and a checkpoint, not new code. The two `-naflex` variable-resolution variants are different — they
use a different image tower — and are not carried; see the vendor's `PROVENANCE.md`.

Multilingual by default: the text tower carries Gemma's 256,000-piece vocabulary rather than an
English byte-pair one, so there is no separate multilingual variant to choose.

### The score is a probability for one pair

`sigmoid(cos × exp(logit_scale) + logit_bias)`, which is what upstream's own examples print. Two
consequences, and they are the reason to reach for this family:

**A single phrase is a well-posed question.** Ask "is there a forklift in this frame" with one
phrase and the answer means something. With CLIP you must pass a complete set of classes and read
the ranking, because a lone cosine similarity has no absolute zero to sit against.

**Everything can be near zero.** On the fixture photograph — people at a table with a laptop —
"a photo of people" scores 0.035 and "a photo of an elephant in a pool" scores 0.000. A softmax
cannot express "none of these", because something always has to win.

Adding a phrase still does not move the others, exactly as with CLIP.

**It is not a calibrated class probability.** Nothing in the training made classes compete, so the
number is how well *this* phrase matches *this* image, not P(class | image). Absolute values run
low — a good match is often a few percent, not 90% — because the learned bias sits near −17. Read
it as a score with a meaningful zero and calibrate a threshold on your own images.

### Write your prompts in lowercase, or let mozo do it

SigLIP 2 was trained on lowercased text, and mozo lowercases for you. This matters more than it
sounds: `"A Photo Of People"` scores 0.0349 lowercased and 0.0001 as written, on the same image.

It is also a place where the published tokenizer configuration is misleading — it names a tokenizer
class that ignores its own `do_lower_case` flag, so the obvious way to load it preserves case and
quietly ruins every capitalised prompt. mozo follows what the authors trained; the vendor's
`PROVENANCE.md` records the divergence and the evidence for it.

### The towers load independently, and it matters more here

Asking for phrases never builds the image tower, and vice versa. The text tower is most of the
checkpoint for this family — Gemma's vocabulary alone is 786 MB of a `base` variant and 1,180 MB of
an `so400m` one — so an ingest job that only embeds images avoids most of the memory.

### SigLIP 2 does not replace CLIP

They are separate vendors with separate weights, separate tokenizers and separate gates, and mozo
carries both. CLIP is what a great deal of existing tooling pins, and a vector from one means
nothing against a vector from the other. Choose SigLIP 2 when you want a score you can threshold;
choose CLIP when you need CLIP's own embedding space.
