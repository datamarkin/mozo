# sam3_deploy

Deployment-only SAM 3 for single images. Apache-2.0 code, derived from
`transformers/models/sam3` rather than from `facebookresearch/sam3`, with no `transformers`,
`hydra` or `iopath` dependency at runtime.

> **Text prompting works end to end** and is verified bit-identical to the published model — a
> decoded image and a noun phrase in, masks, boxes and scores out. The caching layer and the
> click path (points, boxes, mask refinement) are not written yet. See `PROVENANCE.md`.

## What SAM 3 is for

SAM 2 answers *"what is under my cursor"*. SAM 3 answers *"where is every cow"* — you give it a
noun phrase and it returns every instance, with a mask, a box and a score. That is the thing
worth deploying: point a pipeline at 10,000 images with one prompt and let it annotate them.

## The three-way seam

SAM 3's cost is lopsided, and the split is what makes it usable:

```
encode_image   ~4900 ms   depends only on the image   -> cache per image
encode_text      ~78 ms   depends only on the phrase  -> cache per phrase, forever
decode           ~650 ms  depends on both
```

A phrase encoded once is valid for every image afterwards. An image encoded once serves every
prompt afterwards — *and* both prompt modalities, because the neck is dual: one ViT pass feeds
both the concept head and the click head. That is the whole reason to deploy SAM 3 rather than
running SAM 2 and SAM 3 side by side.

The per-image cache is expensive: **223 MB** for the two pyramids in float32, against SAM 2's
16.8 MB. That is a live design constraint, not a footnote.

## Prompting

| prompt | result |
|---|---|
| text | every instance of the concept |
| text + positive exemplar boxes | "find more like this one" |
| text + positive and negative boxes | the same, refined |
| boxes alone | upstream substitutes the literal text `"visual"` |

There is **no negative text** — the negative signal is a negative exemplar box — and the concept
path takes no points. Points and clicks belong to the click head, which shares the image encode.

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

## Supported

- The one published model. Meta ships a single SAM 3; there is no size ladder.
- Fixed 1008x1008 inference. SAM 3 squashes to a square rather than letterboxing, so aspect ratio
  is not preserved and there is no padding to undo.
- Prompts up to 32 tokens — not CLIP's usual 77.
- CPU and CUDA. CPU runs at about 0.2 images/second; this model wants a GPU.

## Licensing

The code here is **Apache-2.0** (see `LICENSE`), and the tokenizer is MIT.

The **weights are neither**. SAM 3's checkpoints are covered by Meta's SAM License, which is not
an OSI-approved open-source licence and carries field-of-use restrictions that flow through to
whoever you serve predictions to. mozo does not redistribute them from this package. See `NOTICE`
and `PROVENANCE.md`.
