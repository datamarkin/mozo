# easyocr_deploy

EasyOCR's inference path, extracted for deployment. Two graphs, no upstream package.

```python
from mozo.vendors.easyocr_deploy import Reader, SPECS

reader = Reader("torch-fp32.pth", SPECS["english"])
for region in reader(rgb_array):
    print(region.text, region.confidence, region.quad)
```

Most callers want `mozo.adapters.easyocr.EasyOCRPredictor` instead, which resolves the weights
and returns PixelFlow detections.

## How it works

| | |
|---|---|
| `craft.py` | The detector. VGG16-BN encoder, U-Net decoder, two heatmaps at half resolution. |
| `boxes.py` | Heatmaps to quadrilaterals to lines. All OpenCV, no torch. |
| `crnn.py` | The recogniser. Convolutions, two BiLSTMs, a CTC head. One line at a time. |
| `text.py` | The alphabet, and the CTC collapse that turns steps into a string. |
| `image.py` | Both preprocessing paths — the whole page, and one rectified crop. |
| `predictor.py` | The seam: `detect` finds lines, `read` reads them one at a time. |
| `config.py` | Which variant is which — which is to say, which alphabet. Charsets in `assets/`. |

A detection is a *word* to CRAFT and a *line* to the recogniser: the detector finds words, and
`boxes.group` merges them into lines before anything is read. Two channels are what make that
work — one scores "inside a character", the other "between two characters of the same word" — so
there are no anchors, no proposals and no NMS anywhere in this package.

## Variants

Five, each a script rather than a language. `latin` covers 41 languages and reads every
character its charset holds.

| Variant | Charset | Covers |
|---|---|---|
| `english` | 96 | en |
| `latin` | 351 | 41 Latin-script languages |
| `chinese-simplified` | 6,718 | ch_sim + en |
| `japanese` | 2,214 | ja + en |
| `korean` | 1,008 | ko + en |

Upstream instead selects a checkpoint from a language list and then suppresses characters
outside those languages at decode time, which makes its output depend on something that is not a
property of the weights. See `PROVENANCE.md`.

## What it costs

Median over eight fixture pages, one M-series CPU, `english`:

| | cpu | mps |
|---|---|---|
| per page | 202 ms | **31 ms** |
| detection alone | 86% of it | 56% of it |
| against the published package | 0.99x | 1.22x |

mps is 6.5x faster and **not bit-identical** -- strings and quadrilaterals are exact, confidences
move by up to 2.2e-05. The gate verifies CPU; ask for `device="cpu"` if you need the verified
numbers. See `PROVENANCE.md`.

The remaining 15% is the recogniser, and it is almost all `torch.lstm` and `conv2d`: one crop
per forward is the cost, and that is a parity constraint rather than an oversight. The
postprocessing around it was rearranged where upstream's shape was wasteful rather than
meaningful — box extraction works inside each word's own window instead of scanning the page
once per word, which is 1,317 ms against 33 ms on a 687-word page for the identical boxes. See
`PROVENANCE.md`.

Detection is the expensive half. That is what the seam between `detect` and `read` is for — not
caching: the reader holds no state between calls, because one image has one answer and there is
no second question to ask it.

## Verified

`tools/verify/easyocr.py` — exact, no tolerance. 1,275 comparisons across all five variants
against the published `easyocr` package and against mozo's own adapter, every stage from the
preprocessed tensor to the decoded string. All identical.

## Not included

Training, DBNet, beam-search decoding, paragraph grouping, RTL display reordering, batched
multi-image inference, rotation retries, custom YAML networks, CRAFT's refiner, and the
first-generation recogniser. `PROVENANCE.md` says why for each.
