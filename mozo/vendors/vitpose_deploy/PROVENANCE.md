# Provenance

This package is a deployment-only extraction of **ViTPose++**'s inference path.

It derives from `transformers/models/vitpose` and `transformers/models/vitpose_backbone`
(Apache-2.0, © The University of Sydney and The HuggingFace Team) — **not** from
`ViTAE-Transformer/ViTPose`, which is the authors' own release and is built on mmpose. Both are
Apache-2.0, so this is not a licensing choice; it is that the `transformers` implementation is the
one whose numbers the published PyTorch checkpoints reproduce, and the one this package can be
checked against on every run. Where the two disagree, this package follows `transformers` — see
**The warp**, below, which is the only place they do.

**The weights are Apache-2.0 too.** All seven published checkpoints, code and weights alike.
This package does not redistribute them; `tools/fetch/vitpose.py` obtains them.

## What was taken

| from | what |
|---|---|
| `modeling_vitpose_backbone.py` | the ViT trunk, its padded patch embedding, and the mixture-of-experts MLP |
| `modeling_vitpose.py` | the classic two-deconvolution decoder and the feature reshape between trunk and head |
| `image_processing_vitpose.py` | the box-to-crop affine and the DARK decode, rewritten in `image.py` and `postprocess.py` |
| `configuration_vitpose*.py` | the numeric geometry, written out as frozen dataclasses in `config.py` |

**There is no `checkpoint.py`.** The module names here are upstream's, so a published checkpoint
loads with `strict=True` and no translation at all — no renames, no dropped tensors, nothing to
keep in step. That is why the names in `layers.py` and `network.py` are upstream's rather than
better ones.

## What was deliberately left behind

**The single-expert MLP.** `VitPoseBackboneLayer` picks between a plain MLP and the MoE on
`num_experts == 1`. Every variant published here sets six, so only the MoE is built. The original
ViTPose's checkpoints are what would need the other branch, and they are not published — ViTPose++
is better at every size, and `plus-small` is 133 MB against the smallest original's 344 MB.

**The simple decoder.** `VitPoseSimpleDecoder` — ReLU, bilinear upsample, one 3×3 convolution — is
selected by `use_simple_decoder`. No published variant here sets it.

**`dataset_index` as a parameter.** Upstream exposes which of the six experts to run. Every
published head is COCO's 17 keypoints and only expert 0 was trained against that head, so
`predictor.EXPERT` is a constant. The other five are not alternatives; they are ways to get a wrong
answer from a model that will not complain.

**The expert mask.** Upstream runs all six experts and multiplies five of them by zero, which is
what a training-time implementation looks like when one batch mixes datasets. Inference asks for
one expert for the whole batch, so `MoeMLP` indexes it: same arithmetic, six times less of it.
`tests/families/test_vitpose.py` holds the two forms equal.

**The flip test.** `flip_back` averages a prediction with its mirror image, worth a few tenths of
AP for twice the compute. It is an evaluation-time augmentation, not part of the model, and mozo
does not silently double the cost of a call.

**Training and evaluation.** No losses, no datasets, no AP harness.

## Parity

Measured against `transformers` 5.8.0 on `tests/fixtures/images/example.jpg` — a five-person
photograph — with boxes from `rfdetr/medium`. Every published variant, `torch-fp32` on CPU:

| stage | `small` | `base` | `large` | `huge` |
|---|---|---|---|---|
| `pixel_values` | 2.4e-07 | 2.4e-07 | 2.4e-07 | 2.4e-07 |
| heatmaps, same input | **0.0** | **0.0** | **0.0** | **0.0** |
| joints, same heatmaps (px) | 1.2e-04 | 2.4e-04 | 2.4e-04 | 2.4e-04 |
| confidences | **0.0** | **0.0** | **0.0** | **0.0** |
| end to end (px) | 2.4e-04 | 3.7e-04 | 4.3e-04 | 2.4e-04 |

The heatmaps are bit-identical, so the network extraction is exact. What is left is float32
rounding in preprocessing: the crops themselves are byte-identical `uint8`, and `pixel_values`
differs by one unit in the last place because upstream normalises through `torchvision`. The joints
land within 0.0005 of a pixel — PixelFlow rounds coordinates to 0.01, so the disagreement is forty
times smaller than the smallest number a result can carry.

`tools/bench/vitpose.py` reproduces the table.

## The warp

Upstream resamples each crop with `scipy.ndimage.affine_transform`. mozo does not depend on SciPy
and would not add a dependency of that size for one resample, so `image.warp` is that operation
written out: inverse map, bilinear gather, zero outside, round to `uint8`. It is bit-identical —
`np.rint` is what SciPy's cast to `uint8` does, and on real photographs the two agree on every
pixel.

`cv2.warpAffine` was the obvious alternative. It is already a mozo dependency, and it is what the
original ViTPose used — `scipy_warp_affine`'s own docstring says it exists to implement it. It is
not used here because OpenCV quantises sampling coordinates to 1/32 of a pixel: measured against an
analytically known answer, OpenCV is off by 0.032 where this is off by 3e-14, and that difference
reached **1.1 pixels** in the final joint positions. Matching the extraction source is worth more
than matching the ancestor it approximates.

Two borders had to be right, and both are places where the natural reading is wrong.

SciPy's `mode="constant"` does **not** interpolate against the constant. A sample at x = -0.2 comes
back as black, not one fifth of the way toward the first column — only `grid-constant` blends.
Since the crop is deliberately larger than the box, most crops have an edge off the frame, and
blending there is wrong by up to 68 levels along it. `tests/families/test_vitpose.py` pins this
against SciPy on boxes that run off every side.

And for the DARK blur: SciPy's `mode="reflect"` repeats the edge
sample, where `torch`'s reflect padding skips it. At σ=0.8 the first neighbour carries about a
fifth of the weight, so choosing wrong is a 0.58 error on a heatmap whose values reach 3 — not a
rounding difference. `postprocess.blur` pads with NumPy's `symmetric`, which is SciPy's `reflect`.

## The exported graphs

`small`, `base` and `large` also publish `onnx-fp32`. The forward pass is the only thing that
changes: `Predictor` takes an injected `forward`, so the crop, the affine and the DARK decode are
the same code on both paths and cannot drift by being reimplemented around a graph.
`tools/export/vitpose.py` verifies each graph against its own checkpoint before writing it, on real
people found by `rfdetr/medium` — a heatmap over an empty box is flat, and comparing `argmax` on a
flat heatmap tests tie-breaking rather than the model. Measured: joints move 0.00043-0.00061 px,
confidences under 3.2e-06.

Binding the expert to a constant lets the exporter fold away the five that never run, so the graphs
are smaller than the checkpoints by exactly `5/6` of the expert parameters — 97 MB against 133 for
`small`, 1235 against 1738 for `large`.

**`huge` publishes no graph.** After folding it is still 2.5 GB, past protobuf's single-file
ceiling, so ONNX writes the weights beside the graph instead of inside it. That stub loads
correctly where it was produced and fails wherever it is published to; the export tool refuses it
rather than writing it.

**No CoreML.** Built, measured and left out rather than never tried. `coremltools` converts this
architecture with no help — RF-DETR needs upstream's converter to register ops first, this does not
— and the joints agree to 0.0005 px. It is not faster than torch on MPS (22.9 ms against 22.6 on
`small`, 44.0 against 44.2 on `base`; fixed shapes and fp16 change nothing, the Neural Engine is
3.5x worse), and publishing it would make it the default on Apple silicon by `_PREFERENCE`. The
numbers are in `tools/export/vitpose.py` so the next attempt starts from them.

## Not verified here

**COCO AP.** The published numbers are upstream's claim, not reproduced. `images.cocodataset.org`
serves a certificate that fails hostname verification, and verification was not bypassed. What is
verified is that this package computes what `transformers` computes, which is the claim mozo is
making.
