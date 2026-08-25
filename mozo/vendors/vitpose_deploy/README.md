# vitpose_deploy

Deployment-only ViTPose++ pose estimation. Apache-2.0 code *and* weights, extracted from
`transformers/models/vitpose` and reduced to the inference path, with no `transformers`, `scipy` or
`torchvision` dependency.

You say where a person is and it says where their joints are.

```python
from mozo.vendors.vitpose_deploy import Predictor
from mozo.image import load_image

model = Predictor("torch-fp32.pth", "base")   # device="cpu"
frame = load_image("street.jpg")

joints = model.predict(frame, [[40, 60, 300, 480], [510, 90, 690, 470]])

joints.shape      # (2, 17, 3)
joints[0, 0]      # the first person's nose: x, y, confidence
```

## It does not detect

ViTPose is **top-down**. It has no detector and does not want one: pair it with whatever produces
boxes — RF-DETR, a YOLO, a tracker, a rectangle someone drew. Boxes are `xyxy` in the frame's own
pixels, and the joints come back in the same coordinates, one row per box, in the order given.

Nothing here filters the boxes either. Hand it a car and it will return seventeen confident joints
on a car. Deciding which boxes are people is the caller's, because only the caller knows what the
boxes mean.

## Give it the frame, not a crop

The crop this model wants is **larger than the box it is given**. Each box is first widened or
heightened to the input's 3:4 aspect ratio, then padded by a further 1.25×. For a 50×140 person
that is roughly a 131×175 crop — about forty pixels of width and thirty-five of height that the
detector's box never contained, taken from the surrounding frame.

So `predict` takes the whole frame. Passing a pre-cropped person is not the same operation: those
pixels are already gone, the padding falls back to black, and a wrist just outside the box — which
the real crop would have recovered — is unrecoverable. If a crop is genuinely all you have, pass it
with a box covering its full extent and expect slightly worse joints near the edges.

## Batching

N boxes are N crops through **one** forward pass. Passing a frame's people together costs far less
than a call each, so pass them together.

Zero boxes returns an empty `(0, 17, 3)`. A frame with nobody in it is an answer, not an error.

## Confidences

Each joint carries the peak value of its heatmap channel. That is not a probability: it is not
calibrated, it does not sum to anything, and it is not comparable across models.

A joint the model cannot see comes back with a confidence near zero and a position that means
nothing — the argmax of a flat heatmap lands somewhere. **Filter on the confidence before reading a
coordinate.** Nothing here does it for you, because the threshold depends on what you are doing
with the answer.

## Variants

| | hidden | layers | joints |
|---|---|---|---|
| `small` | 384 | 12 | 17 |
| `base` | 768 | 12 | 17 |
| `large` | 1024 | 24 | 17 |
| `huge` | 1280 | 32 | 17 |

All four are ViTPose++ — the mixture-of-experts revision. Every block picks one of six dataset
experts; this package always runs COCO's, because COCO's is the one the published heads match. See
`PROVENANCE.md`.

## Input

RGB `uint8` `HxWx3`, as `mozo.image.load_image` returns. Nothing here reads a file, and nothing
here converts channel order — mozo decodes in one place so exactly one piece of code decides it.
