# grounding_dino_deploy

Deployment-only Grounding DINO open-vocabulary detection. Apache-2.0 code *and* weights, extracted
from `IDEA-Research/GroundingDINO` and reduced to the detection path — with no `transformers`,
`timm`, `addict` or CUDA-extension dependency, and no network round trip to tokenize a prompt.

You describe a thing in words and it returns boxes for it. There is no class list, no fine-tuning,
and no vocabulary anyone had to agree on in advance.

```python
from mozo.vendors.grounding_dino_deploy import Predictor, SPECS
from mozo.image import load_image

model = Predictor("torch-fp32.pth", SPECS["tiny"])       # device="cpu"
image = load_image("kitchen.jpg")

found = model(image, ["a kettle", "the mug on the left", "a window"])

found[0].box            # (x1, y1, x2, y2) in the source image's pixels
found[0].score          # how well the prompt matched
found[0].prompt_index   # which prompt that was, indexing the list you passed
```

## Prompting

**A prompt is a description, not a label.** This is the difference from OWLv2, which embeds each
phrase independently. Grounding DINO fuses the text into the image features six times over and
lets the decoder attend back to the words, so `"the mug on the left"` is read rather than treated
as a bag of words. It costs more to run, and it is the one to reach for when the prompt is
descriptive.

**Prompts may not contain `.` or `?`.** Those separate concepts. A prompt carrying one would be
split in two and its detections reported against the wrong phrase, so it is refused instead.

**Case is not preserved.** The caption is lowercased before tokenization, as upstream does, so a
proper noun is not distinguishable from a common one.

**About 60 prompts fit.** The model has a 256-token budget. Upstream truncates past it silently;
this raises, naming the count and the cap.

## Variants

| variant | backbone | box AP (COCO, zero-shot) | weights |
|---|---|---|---|
| `tiny` | Swin-T | 48.4 | 694 MB |
| `base` | Swin-B | 56.7 | 938 MB |

Both are what upstream publishes — there is no third. `tiny` is 82% of the project's own release
downloads; `base` is 8.3 AP better, which is why both are carried.

## Shapes

Unlike every other family in mozo, **nothing is letterboxed to a square.** The short side goes to
800 and the long side is capped near 1333, aspect preserved, nothing padded — so the tensor a
photograph becomes depends on the photograph. A 1920×1281 image runs at 1199×800.

That is why this package publishes no ONNX or CoreML artifact: a graph exported at one shape
cannot take another, and the adapter declares `EXECUTES = ("torch",)` so `auto` cannot pick one.

## What comes back

Boxes, scores, and the index of the prompt that matched. **No masks** — pair it with SAM 2 or
EdgeTAM if you want one, feeding a box from here in as a prompt there.

**Nothing is suppressed.** Two prompts describing the same thing can both find it, and overlapping
boxes for one prompt are the model's answer rather than a failure to deduplicate.

Detections come back in the model's own query order, unsorted. `mozo.adapters.grounding_dino`
ranks them best-first, because every other family in mozo does.

## Fidelity

Bit-identical to upstream: 138 comparisons across both variants and four prompt sets, exact
equality, no tolerance. Every intermediate is hooked, not just the final boxes -- two
implementations can agree on the last tensor and disagree in the middle. Run the gate with

```bash
python tools/verify/grounding_dino.py tiny base --upstream /path/to/GroundingDINO
```

See `PROVENANCE.md` for what was taken, what was left, the two divergences that were deliberate,
and the eight traps that run and are wrong.
