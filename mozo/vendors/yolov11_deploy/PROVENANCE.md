# Provenance

This package is deployment-only YOLO11 inference. Like its `yolov8_deploy` sibling and unlike
mozo's other vendors it is **not an extraction of an upstream repository**: it does not contain,
import or depend on `ultralytics`, and it reproduces no model definition. It reads the checkpoint
file directly and rebuilds the network from what that file records.

| | |
|---|---|
| Relationship to `ultralytics` | none — no import, no dependency, no copied source |
| Written | independently, against the checkpoint format |
| Validated against | `ultralytics` 8.4.0, stage by stage — `tools/verify/yolov11_reference.py` |
| Harvested into mozo | 2026-08-19 |
| Segmentation added | 2026-08-25 |
| Verified with | `torch` 2.11.0, Python 3.10, on CPU |
| Checkpoint source | `ultralytics/assets` release `v8.4.0`, digests read from GitHub — see `tools/fetch/_ultralytics.py` |
| Corresponding source | https://github.com/ultralytics/ultralytics |
| Checkpoint writer | `ultralytics` 8.2.100, recorded in the checkpoints (2024-09-25) |
| Reference version | `ultralytics` 8.4.0 — see below |
| Upstream commit | not pinned; the release the checkpoints came from is |
| Heads served | `Detect` (5 variants) and `Segment` (5 `seg-` variants) |

**The version that wrote a checkpoint is not automatically the version that defines it.** These
files record `8.2.100`, and `ultralytics` refactored its detection head between then and 8.4.0 —
from per-level tensors to a dict of named branches. What a reference has to do is reproduce the
checkpoint's numbers, and 8.4.0 does: it loads an 8.2.100-written checkpoint across that refactor,
fuses it, and runs it. 8.4.0 is also the release the published bytes come from, which is the
pairing that makes "these weights were checked against this code" a statement about one thing
rather than two. The gate refuses to run against anything else.

It is a separate vendor from `yolov8_deploy` rather than a shared substrate with a second block
table, even though roughly four fifths of the two is the same design written twice. That is a
deliberate call: a vendor is meant to be readable and replaceable on its own, and the families
diverge at the head as the line moves on. The cost is real and is recorded here so it is a known
cost rather than an accident — a fix to the checkpoint reader, the builder or the box arithmetic
has to land in both.

## Why it is not a derivative

A `.pt` file is a ZIP archive holding one pickle plus the raw bytes of every tensor storage, and
that pickle records the whole module tree: each layer's class name, its wiring, and every leaf
module's hyperparameters — channel counts, kernel sizes, strides, paddings, batch-norm epsilons,
groups, attention head counts and softmax scales, DFL bin counts, per-level strides and the class
names.

`reader.py` walks that pickle with a restricted unpickler. Every class the file names resolves to
an inert placeholder that keeps its name and attributes; nothing from the framework that wrote the
file is imported or executed, and the builtins it will resolve are whitelisted. `build.py` then
constructs the matching `torch.nn` modules from those recorded numbers. There is no YAML parser, no
width or depth scaling, no `make_divisible`, no padding rule and no head-width formula, because
every number those would produce is already in the file.

The one thing a checkpoint does not record is the *dataflow* of composite blocks — which child
feeds which — and that is what `flow.py` supplies, written from an understanding of what each block
computes. Names like `cv1`, `cv2`, `m`, `nc`, `reg_max`, `num_heads` and `key_dim` appear because
they are read out of the file; they are interface facts about a format, not copied expression.

`mask.py` is the one part that argument does not cover, and it is worth naming. Mask assembly is
not in the checkpoint either, and unlike a block's dataflow it cannot be worked out from what is:
it reproduces the behaviour `ultralytics` 8.4.0 ships — the order of the four steps, the asymmetric
rounding that removes the letterbox border, the threshold against zero, and the crop rule that
changes below fifty detections — because those are what the published weights expect. The parity
table below is the check that it worked.

## Instance segmentation

The `seg-` variants are the same backbone and neck with a `Segment` head instead of a `Detect`
one: the same box and class branches, plus `cv4` predicting 32 mask coefficients per anchor and a
`Proto` branch predicting a stack of 32 prototypes at a quarter resolution. A mask is the linear
combination of the two, resized to the source image, cropped to its own box and thresholded.

Four things about it are read from the file rather than assumed, and each is somewhere a plausible
guess would have been wrong:

- **`npr` is not what the configuration says.** `yolo11-seg.yaml` declares `Segment[nc, 32, 256]`,
  and `parse_model` rewrites the 256 by the width multiplier — 64 on `nano`. A vendor that parsed
  the YAML would build a four-times-too-wide prototype branch and fail the strict load. This one
  reads it, so it cost nothing.
- **`Proto.upsample` is a learned `ConvTranspose2d`**, not an interpolation. Upstream's own source
  carries `# nn.Upsample(scale_factor=2, mode='nearest')` beside it, naming the substitution that
  looks equivalent and is not: it would drop a 64×64×2×2 weight and a bias, which the strict load
  reports rather than ignores.
- **The prototypes come from the finest feature map alone**, not from the concatenated head
  output.
- **The coefficient count is looked up, never defaulted.** `build.HEADS` names the attribute each
  head records it under. `spec.get("nm", 0)` would read the absence of a name as the number zero,
  split cleanly, return the detection shape, and serve boxes with no masks and no error anywhere.

**One thing the checkpoint records that this package deliberately does not build.** The `Segment`
head written by 8.2.100 carries an attribute `detect`, which is the *function*
`Detect.forward` — plumbing it used to reach its base class, pickled as
`getattr(Detect, "forward")`. `reader.py` resolves it to an inert `MethodReference` that keeps the
name and resolves nothing, and only ever against a class the reader has already refused to import.
It is named here rather than dropped silently, because a reader that discarded whatever it did not
recognise could not tell a checkpoint that records nothing from one whose contents it failed to
understand.

**The published segmentation checkpoints contain the detection ones.** For `medium`, `large` and
`xlarge`, *every* tensor `yolo11{m,l,x}.pt` holds is bit-identical in `yolo11{m,l,x}-seg.pt` —
649 of 649 and 1015 of 1015 — so the segmentation variant is the detection variant plus a mask
branch, and `seg-medium`'s boxes are literally `medium`'s boxes. `nano` and `small` share only 81
batch-norm counters and the DFL bin-index constant, so those two were trained apart. This is why
several rows of the parity table below are identical between a size and its `seg-` counterpart:
the same weights produce the same float noise.

## Licensing

The code here is **Apache-2.0**, like the rest of mozo.

Model weights are not. Checkpoints published by Ultralytics — including every variant mozo
publishes for this family — are **AGPL-3.0**, or covered by a commercial licence from Ultralytics.
mozo redistributes them unmodified, with the AGPL text and a NOTICE naming the exact upstream
release, in the same directory as the weights. An ONNX export contains the weights and is covered
by the same terms. Serving predictions from them over a network places AGPL-3.0 section 13
obligations on whoever runs the service; that is the operator's responsibility.

The two are separate works travelling together, which is what the GPL's aggregation clause
permits. Running an AGPL checkpoint through this code does not place this code under AGPL, and
this code does not change the checkpoint's terms.

## Measured parity

Measured by `tools/verify/yolov11_reference.py`, which stands `ultralytics` up beside the vendor
and prints the table below; `tools/verify/yolov11_reference.json` records it for a reader who
cannot run it.

All ten variants, on `example.jpg` (1281×1920), CPU, fused, `ultralytics` 8.4.0, torch 2.9.1.
Head rows are measured at `conf=0.001`; detection and mask rows at the serving threshold of 0.25,
for the reason given under *Where this diverges* below. Every figure is a maximum absolute
difference except `masks`, which is a fraction of pixels.

| Check | Tolerance | nano | small | medium | large | xlarge |
|---|---|---|---|---|---|---|
| Preprocessed input | 1e-6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Worst of all 23 layers | 2e-3 | 2.61e-05 | 2.43e-05 | 4.32e-05 | 1.10e-04 | 3.06e-04 |
| Head boxes, live anchors (px) | 1e-2 | 1.62e-03 | 1.13e-03 | 5.65e-04 | 1.37e-03 | 5.19e-04 |
| Head boxes, all 8,400 (px) | — | 1.62e-03 | 1.13e-03 | 1.04e-03 | 1.83e-03 | 4.62e-03 |
| Head class scores | 1e-3 | 4.20e-06 | 4.77e-06 | 4.08e-06 | 3.71e-06 | 1.20e-05 |
| Detections, count | exact | equal | equal | equal | equal | equal |
| Detections, boxes (px) | 1e-2 | 1.83e-04 | 1.83e-04 | 2.44e-04 | 1.53e-04 | 1.37e-04 |
| Detections, scores | 1e-3 | 1.43e-06 | 1.04e-06 | 4.77e-07 | 8.34e-07 | 5.36e-07 |
| Detections, classes | exact | identical | identical | identical | identical | identical |

And the five segmentation variants, which add three rows:

| Check | Tolerance | seg-nano | seg-small | seg-medium | seg-large | seg-xlarge |
|---|---|---|---|---|---|---|
| Head boxes, live anchors (px) | 1e-2 | 1.65e-03 | 1.40e-03 | 5.65e-04 | 1.37e-03 | 5.19e-04 |
| Head class scores | 1e-3 | 3.16e-06 | 1.17e-05 | 4.08e-06 | 3.71e-06 | 1.20e-05 |
| **Mask coefficients** | 1e-3 | 1.12e-05 | 5.60e-06 | 7.73e-06 | 7.73e-05 | 7.21e-05 |
| **Prototypes** | 1e-3 | 3.12e-05 | 1.44e-05 | 1.31e-05 | 7.73e-05 | 9.77e-05 |
| Detections, count | exact | equal | equal | equal | equal | equal |
| Detections, boxes (px) | 1e-2 | 6.10e-04 | 1.37e-04 | 2.44e-04 | 1.53e-04 | 1.37e-04 |
| **Assembled masks** | 1e-5 | 7.39e-08 | 0.0 | 0.0 | 0.0 | 2.90e-08 |

335 comparisons across the ten, every one within tolerance. Three of the five segmentation
variants are **pixel-identical** to the reference; `seg-nano` differs by **2 pixels of
27,054,720** across its 11 masks and `seg-xlarge` by **1 of 34,433,280** across its 14. Those are
diagnosed rather than waved through: each flipping pixel carries a mask logit within 2e-06 of
zero, and the threshold *is* a comparison against zero. `seg-nano`'s two read **−2.2e-07 against
+8.3e-08** and **−1.8e-07 against +1.6e-06**; `seg-xlarge`'s single one reads −1.6e-06 against
+9.0e-07. The largest logit disagreement anywhere in those 11 masks is
6.45e-05, so a boundary pixel flipping is the mask analogue of a box moving by 6e-04 px. Across
the five, the worst logit disagreement runs 3.9e-05 (`seg-xlarge`) to 6.5e-05 (`seg-nano`).

**Why the head boxes are measured twice.** Over the anchors a caller can reach, `xlarge` reads
5.19e-04 px; over all 8,400 it reads 4.62e-03, nine times larger. The worst-disagreeing anchors
are ones whose scores no threshold clears, so neither implementation's answer for them is ever
read. Gating on the whole grid would hold the family to the float noise of its dead anchors;
gating quietly on the survivors alone would hide that the grid moves at all. So both are measured
and one is gated. On `nano`, `small` and `seg-small` the two rows are identical; they separate
from `medium` up.

**The gate has been made to fail, and it localises.** `--falsify all` on `seg-nano`:

| perturbation | moved | left alone |
|---|---|---|
| constant added to every fused bias | layers 8 onward, head, prototypes, coefficients, detections | the preprocessing |
| strides scaled by 1.001 | boxes, and the masks they crop | every score, every prototype |
| `Proto.upsample` → `nn.Upsample` | prototypes and masks | every box, score and layer |
| coefficients rolled one anchor | coefficients and masks | every box and score |
| anchor cache cleared (control) | nothing | everything |

The control is what makes the other four mean anything, and the bias row is the boundary check: a
change inside the network must never move the preprocessing, and if it does, the two sides are not
being handed the same batch.

## Where this diverges, and why

Three places where this package does not do what upstream's default path does. Each is
deliberate, and each is a divergence *from the obvious reading* rather than a fix.

**Masks come back in source pixels, not network pixels.** Upstream's predictor has two paths: by
default it returns masks at the letterboxed 640×640 while scaling boxes back to the source image,
so the two do not describe the same coordinate system and the caller is expected to know. Its
`retina_masks` path returns both in source pixels, and that is the one `mask.py` follows step for
step. mozo returns boxes in source pixels, so a mask that did not match them would be unusable.

**The mask post-processing is pinned to 8.4.0's, and that is a choice.** Nothing about mask
assembly is stored in a checkpoint — post-processing is not learned — and upstream has changed it.
`ultralytics` 8.3.63 crops with a single vectorised comparison and removes the letterbox border
with `int(pad)`; 8.4.0 carries two crop branches, selected on `n < 50 and not cuda`, and rounds
the border with an asymmetric nudge, `round(pad - 0.1)` against `round(pad + 0.1)`. The two do not
agree: the loop rounds each box edge to a whole pixel where the vectorised form compares against
the unrounded float, so a box edge at `x2 = 10.4` clears column 10 in one and keeps it in the
other. This package reproduces **8.4.0**, the release its checkpoints are published in, including
the branch selection — which means the mask a caller gets depends on how many detections are in
the picture. That is upstream's behaviour, reproduced rather than tidied away.

Two consequences worth stating. The border is removed at *prototype* resolution, before the
resize, so the rounding acts on a quarter-scale padding; and the threshold is applied to logits at
zero, with no sigmoid anywhere. Sigmoid then thresholding at 0.5 picks the same side of the same
boundary only if nothing in between is non-monotonic, and the bilinear resize in the middle is
exactly that.

**Suppression separates classes differently, and it is the more exact of the two.** Upstream
shifts every box by `7680 * class` and runs one pass; this package shifts by the boxes' own span,
because a fixed 7680 is only wide enough when coordinates are positive, and a detection running
off the letterbox edge produces negative ones.

The difference is measurable, and was measured. On `seg-nano` a pair of class-56 boxes overlaps by
**0.700377**, which this package suppresses at an IoU threshold of 0.7. Shifted into upstream's
band the arithmetic happens at 430,080, where the float32 spacing is 0.031 px, and the same
overlap computes as **0.699832** — which upstream keeps. One detection, scoring 0.005, decided by
arithmetic precision rather than by either rule.

So the gate compares detections at the serving threshold of 0.25, where the two agree exactly on
all ten variants, and reports the count at `conf=0.001` without gating it — one row differs there,
on `seg-nano`, and it is the row described above. A count is a decision rather than a
measurement, so it carries no tolerance.

### Two things the table does not show on its face

**The padding convention is load-bearing in every number above.** `example.jpg` letterboxes to a
spare of 106.5 px, so it lands either side of the rounding depending on the convention used;
`tests/test_vendor_agreement.py` holds the package to the border the letterbox actually writes.
An image that padded to whole pixels would agree under either convention and could not show this.

**Unfused inference finds the same detections, and moves the boxes slightly.** Against the same
fused reference, `nano` unfused reads 1.83e-04 px on its detections and 1.92e-03 px on its head
boxes; `seg-nano` reads 1.83e-03 px and 2.32e-03 px, with prototypes at 2.92e-05 and one mask
pixel in 27 million differing. Counts and class ids are unaffected. Fusion is arithmetic
reordering, so comparing an unfused side against a fused one measures the fusion — both sides of
the table above are pinned fused for that reason.

## What mozo changed on harvesting

The package arrived shaped as a standalone distribution. Nothing about the network, the checkpoint
reader or the block arithmetic was touched; what changed is everything that assumed it was a
library rather than a vendor, plus one correction.

- **Removed:** `pyproject.toml`, `__main__.py`, the command line and the test suite. A vendor is
  not separately installable and has no command line.
- **Moved:** `export.py` to `tools/export/yolov11.py`. `__init__.py` named it, and it imports
  `onnx` at module scope, so importing the package at all required a library mozo does not depend
  on. Exporting is something you do once when publishing, not something a deployment package does.
- **Flattened:** `yolov11_deploy/yolov11/*.py` up to `yolov11_deploy/`, and split along the same
  seams as the sibling vendor: `checkpoint.py` → `reader.py`, `dataflow.py` → `flow.py`,
  `model.py` merged into `build.py`, and `detector.py` divided into `image.py` (letterbox,
  suppression, coordinate mapping) and `model.py` (the detector).
- **Added the `detect()` seam:** letterbox → *forward* → suppress → map back, with the forward pass
  passed in. A torch module and an ONNX session then differ only in that middle step and share one
  pre- and post-processing path, which is the property mozo publishes two artifacts on.
- **RGB instead of BGR:** `image.py` no longer decodes anything. It arrived reading files with
  `cv2.imread` (BGR) and flipping to RGB while building the batch; mozo decodes in one place
  (`mozo.image.load_image`, RGB), so the decode was removed and the flip with it. The network sees
  a bit-identical tensor — the change cancels — and this was verified rather than assumed: on the
  fixture photograph the harvested pre-processing and the current one produce `max|delta| = 0.0`.
  `cv2.resize` and `cv2.copyMakeBorder` stay, because letterbox geometry is model maths.
- **Corrected the padding it reports.** The harvest placed the resized image at `floor(spare)` on
  the canvas but returned `spare` unrounded, and subtracted the unrounded value when mapping boxes
  back. For any source whose scaled side is odd that is half a canvas pixel of disagreement between
  where the image was put and where the arithmetic thinks it was, or `0.5 / gain` source pixels —
  1.5 px on mozo's fixture photograph, on every box, always in the same direction. `image.py` now
  reports the floored placement, which is the one that actually happened, and
  `tests/test_vendor_agreement.py` holds both YOLO vendors to it by counting the border rows the
  letterbox really wrote. The sibling vendor already floored and did not change.

`tools/verify/yolov11.py` re-runs the vendor-against-mozo comparison over any images you give it.
`tools/verify/yolov11_reference.py` is the third path, against `ultralytics` itself.
