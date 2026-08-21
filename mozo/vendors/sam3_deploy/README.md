# sam3_deploy

Deployment-only SAM 3 for single images. Apache-2.0 code, derived from
`transformers/models/sam3` rather than from `facebookresearch/sam3`, with no `transformers`,
`hydra` or `iopath` dependency at runtime.

> **Both heads work end to end** and are verified bit-identical to the published model: name a
> concept and get every instance, or click and get the one thing. Every stage the gate covers is
> identical to Meta's implementation; `PROVENANCE.md` says which are compared against it directly
> and which are recorded from mozo alone.

## What SAM 3 is for

SAM 2 answers *"what is under my cursor"*. SAM 3 answers *"where is every cow"* — you give it a
noun phrase and it returns every instance, with a mask, a box and a score. That is the thing
worth deploying: point a pipeline at 10,000 images with one prompt and let it annotate them.

It also still answers the cursor question. SAM 3 carries its own click head — the same
architecture SAM 2 uses, with SAM 3's own weights and geometry — so one checkpoint and one
process serve both — an annotator can run a phrase over a corpus and then
correct a single instance by hand without a second model being deployed to do it.

## The three-way seam

SAM 3's cost is lopsided, and the split is what makes it usable:

```
encode_image   ~4900 ms   depends only on the image   -> cache per image
encode_text      ~78 ms   depends only on the phrase  -> cache per phrase, forever
decode           ~650 ms  depends on both
```

A phrase encoded once is valid for every image afterwards. An image encoded once serves every
prompt afterwards. What it does *not* buy is one encode for both heads: they preprocess an image
differently, so each runs its own trunk pass and reads its own neck stack. One checkpoint and one
process still serve both, which is the reason to deploy SAM 3 rather than two models.

The per-image cache is expensive: **111 MB** per entry in float32 for one FPN pyramid, against
SAM 2's 16.8 MB, and the two heads keep separate entries. That is a live design constraint, not
a footnote.

## Prompting

A prompt is one structure, not a menu of modes: a phrase, and a set of *exemplars* — boxes, each
carrying a label saying whether it is an example of the thing you want or of the thing you don't.
Which fields you fill decides what you get; there is no separate call per combination.

| field | |
|---|---|
| `text` | the concept, as a noun phrase — up to 32 tokens |
| `boxes` | exemplars, normalised `(cx, cy, w, h)` |
| `box_labels` | `1` this is an example, `0` this is not |

Filling only `text` finds every instance of the concept. Adding a positive exemplar says "more
like this one"; adding a negative one carves back what came out. To prompt with exemplars *only*,
pass the literal phrase `"visual"` as the text — that is what upstream uses when a caller supplies
geometry and no concept, and passing it yourself keeps the argument required rather than making a
substitution behind your back.

Two things decide whether you get what you meant:

- **`box_labels` is required with `boxes` and has no default.** Guessing between a positive and a
  negative exemplar returns a confident answer to the wrong question, so it raises instead.
- **There is no negative text.** The negative signal is a negative exemplar, not a second phrase.
  Nor does this path take points: a point prompt is the *click* head's, reached through
  `Segmenter.segment` instead.

```python
from mozo.vendors.sam3_deploy.grounding import ConceptHead
from mozo.vendors.sam3_deploy.predictor import instances

features = vision(preprocess(image))          # once per image
encoded = text(tokenizer(["cow"]))            # once per phrase, then reusable forever
result = concept(features["concept"], features["positions"],
                 encoded["features"], encoded["mask"])
found = instances(result, image.shape[:2])[0]

found["masks"]   # (N, H, W) bool, in the source image's pixels
found["boxes"]   # (N, 4) xyxy, in source pixels
found["scores"]  # (N,)
```

## Clicking

```python
found = segmenter.segment(image, points=np.array([[x, y]]), labels=np.array([1]))
found["masks"]   # (1, 3, H, W) bool -- part, subpart and whole
found["scores"]  # (1, 3) predicted IoU
found["logits"]  # (1, 3, 288, 288) -- feed one back as mask_input to refine
```

The same one-structure rule: `1` include, `0` exclude, and a box is its two corners carrying the
reserved labels `2` and `3`, because the network has no box input. `boxes=` spells that for you.
Give a box and points together and the corners go first — the encoder adds a different learned
embedding per position, so the order is meaning rather than convention.

**The click path does not share the concept path's image encode**, and this is worth knowing
before you plan around it. The two heads preprocess the same photograph differently — one rounds
the resize back to uint8, the other stays in float — and half a grey level of input moves the
predicted IoU by 9e-03. The published model runs the trunk twice for this reason; so does mozo,
with a second cache. Using both heads on one image costs two encodes, not one.

What you still save is everything after the first click. On a 1999x1510 photograph, on an M-series
GPU (`device="mps"`):

| | encode | click |
|---|---|---|
| MPS | 1.16 s | 14 ms |
| CPU | 5.1 s | 44 ms |

The trunk is 90 percent of the encode, which is why the second one costs what it does. Of a 14 ms
click, 3.7 ms is the sha256 over the image that keys the cache -- content keying is what makes the
same photograph arriving twice over HTTP one encode instead of two, and it is not free.

A held image answers a repeated click identically, which the gate checks rather than assumes.

## Supported

- The one published model. Meta ships a single SAM 3; there is no size ladder.
- Fixed 1008x1008 inference. SAM 3 squashes to a square rather than letterboxing, so aspect ratio
  is not preserved and there is no padding to undo.
- Prompts up to 32 tokens — not CLIP's usual 77.
- Clicks, boxes and mask refinement, through SAM 3's own tracker head.
- CPU and CUDA. CPU runs at about 0.2 images/second; this model wants a GPU.

Not supported: video tracking, and hole filling (which needs a CUDA extension) — so binary masks
come out a strict subset of the reference's, differing only inside holes it would have filled.
The logits they are thresholded from are bit-identical.

## Licensing

The code here is **Apache-2.0** (see `LICENSE`), and the tokenizer is MIT.

The **weights are neither**. SAM 3's checkpoints are covered by Meta's SAM License, which is not
an OSI-approved open-source licence and carries field-of-use restrictions that flow through to
whoever you serve predictions to. mozo does not redistribute them from this package. See `NOTICE`
and `PROVENANCE.md`.
