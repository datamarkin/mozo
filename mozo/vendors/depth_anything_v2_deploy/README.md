# depth_anything_v2_deploy

Depth Anything V2, reduced to inference.

```python
from mozo.vendors.depth_anything_v2_deploy import Predictor, get_spec

predictor = Predictor.from_pretrained("small", weights="torch-fp32.pth", device="mps")
depth = predictor.predict(image)          # HxW float32, same size as the input
```

Nine variants. Three predict **relative** depth — unitless inverse depth, larger meaning
nearer. The other six predict **metric** depth in metres, indoors (`indoor-*`, 0–20 m) or
outdoors (`outdoor-*`, 0–80 m).

```
small  base  large                          relative,  unit None
indoor-small   indoor-base   indoor-large   metric,    unit "metres"
outdoor-small  outdoor-base  outdoor-large  metric,    unit "metres"
```

`get_spec(variant).unit` is the only thing that says which. It is `None` for the relative
models rather than a guess, for the same reason mozo does not invent class names: a caller that
needs metres has to pick a variant that has them.

Depends on `torch`, `torchvision`, `numpy` and `cv2`. Every import inside the package is
relative, so the directory can be moved or renamed without edits.

Weights are not this package's concern — `Predictor.from_pretrained` takes a path. See
`mozo.weights` for how mozo resolves one, and note that the relative `base` and `large`
checkpoints are CC-BY-NC-4.0 while this code is Apache-2.0.

See `PROVENANCE.md` for what was extracted, what was changed, and how it is verified.
