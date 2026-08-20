# sam2_deploy

Deployment-only SAM 2 promptable segmentation for single images. Apache-2.0 code, extracted from
`facebookresearch/sam2` and reduced to the image path, with no `hydra`, `omegaconf`, `iopath` or
`torchvision` dependency.

```python
from mozo.vendors.sam2_deploy import Segmenter
from mozo.image import load_image

segmenter = Segmenter("sam2.1_hiera_base_plus.pt")     # device="cpu"
image = load_image("photo.jpg")

found = segmenter.predict(image, points=[[820, 640]], labels=[1])
found = segmenter.predict(image, boxes=[40, 60, 300, 480])

found.masks     # (b, c, h, w) bool, in the source image's pixels
found.scores    # (b, c) the model's predicted IoU for each mask
found.logits    # (b, c, 256, 256) low-res logits
```

## Prompting

Every prompt is a set of points. A click is a point with a label; a box is its two corners
carrying reserved labels, because SAM 2 has no separate box input.

| Label | Meaning |
|---|---|
| `1` | include this — the thing you want is here |
| `0` | exclude this — you got too much, cut here |
| `2` | a box's top-left corner |
| `3` | a box's bottom-right corner |

You pass `1` and `0` yourself; `2` and `3` are written for you when you pass `boxes`.

Three things decide whether you get the mask you meant:

- **`labels` is required with `points` and has no default.** A missing label is not an omission
  this package can fill in — guessing between include and exclude returns a confident mask of the
  wrong thing, so it raises instead.
- **Order matters when you combine them.** Box corners are placed before points, because the
  prompt encoder adds a different learned embedding per position. `predict` does this for you;
  it matters if you drive `network.decode` directly.
- **Negative points need something to subtract from.** A prompt of only `0` labels does not mean
  "everything else".

`multimask_output=True` (the default) returns three candidate masks with their predicted IoU.
That is the right setting for a single click, which is genuinely ambiguous about whether you
meant the handle, the door, or the car. Take the highest-scoring one, or show all three. With a
box, or several points, the prompt is usually unambiguous and `multimask_output=False` is
tighter.

To refine, pass one of the previous call's `logits` back as `mask_input` with an extra click —
one channel, not all three, so choose the candidate first:
`found.logits[:, found.scores[0].argmax()]`.

## The encode/decode split

SAM 2's cost is almost entirely the image encoder, and the encoder depends only on the image. So
`predict` caches encoder output keyed on pixel content, and a second prompt on the same image
skips it:

```
encode (base_plus, CPU)   ~1000 ms
decode                      ~48 ms
```

The cache holds five images, ~17 MB each. Keying on content rather than on a filename
or an object identity means the same photograph arriving twice over HTTP is still one encode.

`network.encode` and `network.decode` are the seam a graph runtime plugs into, and they export
as two separate artifacts for the same reason they are two methods.

## Supported

- The four SAM 2.1 image variants: `tiny`, `small`, `base_plus`, `large`. The variant is inferred
  from the checkpoint, so a fine-tune loads without being named.
- Points, boxes, both together, and mask refinement.
- CPU and CUDA. **Not MPS** — see below.
- Fixed 1024x1024 inference. SAM 2 squashes to a square rather than letterboxing, so aspect ratio
  is not preserved and there is no padding to undo.

Not supported: video tracking, the automatic mask generator, and hole filling (which needs a CUDA
extension upstream builds separately).

## Runtimes

`torch-fp32` and `onnx-fp32` for every variant, and `coreml-fp16-*` for all but `base_plus`.

The CoreML packages are **Apple's**, redistributed rather than converted here — `apple/coreml-sam2.1-*`,
Apache-2.0. Three things follow from that. They are fp16, so they differ from the fp32 reference on
about 1–2% of mask pixels. They split three ways (image encoder, prompt encoder, mask decoder)
rather than two, and their encoder takes an image with normalisation inside the graph. And
`base_plus` has none, because Apple did not publish one.

On Apple Silicon the encoder is about four times faster in CoreML than in torch on CPU. The
decoder is faster in ONNX than in either. Since the halves are separate artifacts, mixing them is
allowed and is the fastest combination.

**MPS is refused.** It runs, and it is the fastest torch path, but on `torch` 2.11.0 the Hiera
trunk returns wrong numbers under `torch.no_grad()` and `torch.inference_mode()` — which is how
inference runs. The same code under `enable_grad` agrees to 2.6e-05. Roughly 0.37 percent of mask
pixels move and nothing raises. See `PROVENANCE.md` for the measurements.

## Licensing

The code here is **Apache-2.0** (see `LICENSE`), and so are the checkpoints.

This makes SAM 2 unusual among the families mozo serves: there is no separate weights licence to
read, no AGPL obligation on serving predictions, and no NOTICE to ship beside the weights. Meta
licenses the code, the training code and the model checkpoints under the same terms.

See `PROVENANCE.md` for what was extracted, what was modified, and the measured parity against
the original implementation. See `NOTICE` for attribution.
