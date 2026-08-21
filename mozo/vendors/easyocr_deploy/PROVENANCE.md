# Provenance

## Where this came from

| | |
|---|---|
| Upstream | `https://github.com/JaidedAI/EasyOCR` |
| Commit | `363afb184047ce452e436f4224f3098422df872e` (2025-12-05) |
| Version | 1.7.2, the last release to PyPI (2024-09-24) |
| Code licence | Apache-2.0 |
| Weights | Release assets of that repository, under the same licence |
| Extracted | 2026-08-21 |

Upstream is effectively frozen: 1.7.2 is two years old and the repository has seen
documentation changes since. For a parity target that is a feature — what this package is
verified against cannot move underneath it.

## What this is

Two graphs. `craft.py` locates lines of text; `crnn.py` reads one line at a time. Between them
sit `image.py` (both preprocessing paths), `boxes.py` (heatmaps to quadrilaterals to lines) and
`text.py` (CTC). `predictor.py` is the seam, `config.py` and `checkpoint.py` say which variant is
which.

This is an *extraction*, so the module and parameter names are upstream's wherever the weights
depend on them: `basenet.slice1`, `FeatureExtraction.ConvNet`, `SequenceModeling`, `Prediction`.
A key here can be grepped for over there unchanged, and the published checkpoints load with
`strict=True` after nothing but a `module.` prefix strip.

## Verified

`tools/verify/easyocr.py`, exact — no tolerance anywhere. It compares three paths: this package,
the published `easyocr` package, and mozo's own adapter through PixelFlow. Every stage is
compared, not only the final string: the preprocessed tensor, both heatmaps, the quadrilaterals,
the rescaled polygons, the grouped lines, each crop, the decoded text and the confidence.

Upstream is driven through `detect` on the same RGB array and `recognize` on the same greyscale
page, rather than through `readtext` on a path. That is deliberate — see **The input contract**
below.

## What changed, and why

Everything here changes *how the code is arranged*, not what it computes. Each item was measured
at zero delta against upstream before it was kept.

1. **The VGG encoder is written out instead of sliced from torchvision.** Upstream builds CRAFT's
   encoder by slicing `torchvision.models.vgg16_bn().features` at four hard-coded indices. Those
   indices are into a third-party model that this package must load one specific set of weights
   into forever. The layers are constructed explicitly here; the *numbering* is kept, so
   `slice2` still starts at `12`, because the checkpoint's keys say so.

2. **CRAFT returns only the heatmaps.** Upstream's `forward` also returns the 32-channel feature
   the head sits on. It exists to feed CRAFT's optional refiner — a separate network for
   polygon-accurate boundaries that EasyOCR never instantiates and publishes no weights for.

3. **Only the second-generation recogniser is vendored.** Upstream ships two networks: a VGG
   extractor with 256-wide LSTMs, and an earlier ResNet one with 512-wide LSTMs. All five
   published variants are second generation, so the ResNet extractor would be a class nothing
   constructs.

4. **`CRNN.forward` takes no `text` argument.** Upstream's signature accepts one and ignores it,
   so an attention-based head could share the call. This is a CTC model and no attention head is
   extracted.

5. **`BidirectionalLSTM` does not call `flatten_parameters()`.** Upstream calls it inside a bare
   `try`/`except` so `DataParallel` does not warn. It relays out the weights and changes no
   number; there is no `DataParallel` in a deployment package.

6. **The crop height is a constant, not a rewritable global.** Upstream keeps `imgH` at module
   scope and a custom model's YAML reassigns it for the whole process; here it is
   `image.MODEL_HEIGHT` and nothing writes to it. All five published variants train at 64, so it
   is not per-variant configuration and is not presented as any.

7. **The charsets are data files.** Upstream keeps them as literals in `config.py`. The Chinese
   one is 6,718 characters.

8. **`os.environ["LRU_CACHE_CAPACITY"] = "1"` is not reproduced.** Upstream sets it at import.
   A process-wide torch allocator setting is not a library's to write.

9. **One checkpoint per variant.** Upstream ships the detector once and a recogniser per script.
   mozo publishes a variant as one download, so `tools/fetch/easyocr.py` fuses the two state
   dictionaries into one file. The tensors are upstream's, byte for byte.

10. **Four rearrangements that compute the same numbers, kept because upstream's shapes are
    wasteful rather than meaningful.** Each was verified to produce bit-identical output before
    it was kept, and the gate re-checks all of them:

    - `boxes.quads` works inside each component's own window. Upstream allocates a
      full-heatmap array per component and scans the whole page to find pixels that can only lie
      inside that window, which costs O(words x page): **1,317 ms against 33 ms** on a 687-word
      page, for the identical points.
    - `image.for_detector` subtracts and divides through a torch view of the same buffer.
      Broadcasting a `(3,)` over a contiguous `(H, W, 3)` lands in numpy's slow inner loop --
      **34.8 ms against 4.7 ms** on a 2 MP page. Both are elementwise and correctly rounded.
    - `text.Alphabet` builds its character table once. Upstream rebuilds it inside every decode,
      which for the 6,719-symbol Chinese alphabet is 0.54 ms a line.
    - `image.line_image` takes one line rather than two lists of them, because upstream's own
      CPU path passes exactly one. The page-wide sort and width quantisation that shape is for
      are no-ops on a single element.

11. **A variant is a script, and decodes its whole alphabet.** This is the one behavioural
    difference, and it is deliberate. Upstream picks a checkpoint from a *language* list and then
    zeroes every character outside those languages before the argmax
    (`ignore_char = set(characters) - set(lang_char)`), so its output depends on something that
    is not a property of the weights. Asking upstream for `['en', 'fr', 'de', 'es']` suppresses
    203 of the Latin recogniser's 351 characters; asking for all 41 Latin languages suppresses
    far fewer. mozo's `latin` variant reads everything `latin_g2` knows, so it returns `café`
    where `Reader(['en'])` returns `cafe`. The gate asks upstream the same question by passing
    the full charset as its `allowlist`.

## The input contract

mozo's contract is an RGB array, and this package takes one and derives the greyscale page with
`COLOR_RGB2GRAY`. Upstream has two entry points that disagree with each other:

- Given a **path**, it builds RGB for the detector and reads greyscale separately with
  `cv2.imread(..., IMREAD_GRAYSCALE)` — libjpeg's direct-to-grey decode, which differs from
  converting its own decoded RGB by up to **7 levels** on a JPEG.
- Given an **array**, it documents the input as BGR and derives greyscale with
  `COLOR_BGR2GRAY`. Handed RGB — which is what mozo has — that channel-swaps the crops the
  recogniser sees, **27 levels** on a colour photograph.

Neither difference is a statement about the model, so the gate hands both sides the same two
arrays rather than comparing through either decoder. Measured: with the same inputs, the
photograph is bit-identical; through upstream's path entry its confidence differs in the third
decimal, entirely from the JPEG decode.

## Devices

The gate runs on CPU, and the parity claim is a CPU claim. mozo's default device is whatever
`get_default_device()` picks, which on Apple silicon is `mps`, and **mps is not bit-identical to
CPU**: across all five variants and every fixture, 58 of 275 comparisons differ. Every string and
every quadrilateral is exact; only the confidence moves, by at most 2.2e-05.

That is a backend difference rather than an extraction one -- the same reduction on two devices --
and it is recorded because the verified path and the default path are not the same path. Anyone
who needs the verified numbers should ask for `device="cpu"`.

It is worth the trade for most callers: 31 ms a page against 202 ms.

## Export

Neither graph is published, and both reasons are measured rather than assumed.

**CRAFT exports and is slower.** 83.1 MB, parity 1.8e-07, and 280 ms against torch's 185 ms on
CPU -- 0.66x. Against the 17 ms a page it takes on mps, an ONNX detector is not a runtime anyone
would select.

**The recogniser does not export.** `AdaptiveAvgPool2d((None, 1))` has an output size that depends
on the input's width, which cannot be traced under a dynamic width; the exporter refuses with
"adaptive pooling, since output_size is not constant". The one substitution that would trace --
a mean over the same axis -- is the same arithmetic on paper and not in float, because the pool
divides its sum by three where mean multiplies by a reciprocal. Measured at up to 1e-06 on the
confidence, which fails the gate. A fixed crop width would also trace, and would mean padding
every line to the longest one the model can take.

## Measured traps

Each of these changes the output, and each is the kind a tolerance would hide. They are recorded
because every one of them is something a later reader would reasonably "fix".

1. **`cv2.resize(..., interpolation=Image.Resampling.LANCZOS)` is bilinear.** PIL's `LANCZOS` is
   `1`; so is `cv2.INTER_LINEAR`. OpenCV's own Lanczos is `4`. Upstream passes a PIL enum to an
   OpenCV function, so that call has always been bilinear whatever it reads like. Spelled
   correctly and left bilinear here.

2. **One crop per forward, not a batch.** Upstream's reader has two paths and takes the per-line
   one whenever it is on CPU or its batch size is one — its default. Each line is padded to its
   *own* width rather than the page's widest, and a batched forward is not bit-identical to
   single ones: **1.4e-05** on the logits, enough to flip a marginal character.

3. **Padding replicates the crop's last column.** Zero padding reads as a black bar, which the
   recogniser decodes as characters.

4. **The crop is resized twice** — cv2 bilinear in `crop_to_height`, then PIL bicubic in `align`.
   Collapsing them into one resize moves pixels.

5. **The thresholds are the reader's, not `group_text_box`'s.** That function's signature says
   `width_ths=1.0, add_margin=0.05`; the reader passes `0.5` and `0.1`. They decide which crops
   exist, so the wrong pair changes the text and not merely the rectangles.

6. **`quantize=True` is upstream's default and silently does nothing here.** Its reader calls
   `torch.quantization.quantize_dynamic` inside a bare `except`. Where torch has a quantization
   engine the recogniser really is qint8; on this machine it raises `NoQEngine` and is not. The
   gate pins `quantize=False` so it compares the same model on every machine.

7. **The heatmaps are half resolution**, so `rescale` multiplies by two before undoing the input
   resize, and truncates to `int32` — upstream's cast, which the grouping then reads.

8. **The low-contrast retry is part of the model.** A crop scoring below 0.1 is read again with
   its contrast stretched and the better answer wins.

9. **Confidence is `prod ** (2 / sqrt(n))`**, over every non-blank step including the repeats the
   CTC collapse drops. It is also the number the retry compares.

## Not included

Training. The DBNet detector (`detect_network='dbnet18'/'dbnet50'`), which needs deformable
convolutions compiled with ninja. Beam-search and word-beam-search decoding, which need a
language-model file that nothing in mozo selects. `paragraph=True` grouping — geometric merging
over results mozo already returns, which also discards confidence. Right-to-left display
reordering (`python-bidi`), which changes a string's display order rather than its content.
Batched multi-image inference. `rotation_info` retries. Custom user networks loaded from YAML.
CRAFT's refiner. The first-generation recogniser.

Upstream's own dependencies on `scikit-image`, `python-bidi`, `Shapely`, `pyclipper`, `ninja` and
`PyYAML` are all on code paths listed above, so this package adds no dependency to mozo.
