# owlv2_deploy

Deployment-only OWLv2 open-vocabulary detection. Apache-2.0 code *and* weights, extracted from
`transformers/models/owlv2` and reduced to the detection path, with no `transformers`, `tokenizers`
or `torchvision` dependency.

You name a thing in words and it returns boxes for it. There is no class list, no fine-tuning, and
no vocabulary anyone had to agree on in advance.

```python
from mozo.vendors.owlv2_deploy import Detector
from mozo.image import load_image

detector = Detector("torch-fp32.pth", "base-ensemble")   # device="cpu"
image = load_image("kitchen.jpg")

found = detector.predict(image, ["kettle", "a mug", "the window"])

found.boxes       # (n, 4) xyxy in the source image's pixels
found.scores      # (n,) how well the best phrase matched
found.labels      # (n,) which phrase that was, indexing the list you passed
found.objectness  # (n,) how likely it is an object at all, whatever you asked for
```

## Prompting

Phrases go in **verbatim**. OWLv2's own examples read `"a photo of a cat"`, and that wrapping is
sometimes worth a point or two — but it is your wording to choose. Nothing here templates it.

| | |
|---|---|
| Length | up to 16 tokens. Longer is truncated, and still terminated. |
| Case | ignored. `"A Red Hat"` and `"a red hat"` are one prompt. |
| Cost | all phrases share one text forward and one image forward. Twenty phrases cost barely more than one. |

**Scores run low.** They are similarities put through a sigmoid, not class probabilities: 0.3 is a
confident detection here where a closed-vocabulary detector would say 0.9. The default floor is
0.1, which is upstream's.

**Each candidate keeps only its best phrase.** Ask for `"car"` and `"vehicle"` and each detection
picks a side rather than appearing twice. That is upstream's behaviour.

**There is no non-maximum suppression**, and that is not an omission — the published
postprocessing has none either. A detection *is* a patch: the model scores every patch of the
image against every phrase and predicts one box per patch, so it always returns exactly
`patches²` candidates and thresholding is the whole of the selection. You will see overlapping
boxes on a large object. Suppressing them is a policy, and it belongs to whoever is drawing.

## Variants

| variant | trunk | resolution | patches | parameters |
|---|---|---|---|---|
| `base-ensemble` | ViT-B/16 | 960 | 3,600 | 154.6 M |
| `base` | ViT-B/16 | 960 | 3,600 | 154.6 M |
| `large-ensemble` | ViT-L/14 | 1008 | 5,184 | 436.8 M |
| `large` | ViT-L/14 | 1008 | 5,184 | 436.8 M |

`-ensemble` averages the self-trained and fine-tuned checkpoints and is what the paper reports;
the plain ones are self-training only. Google also publishes two `-finetuned` checkpoints, which
mozo does not carry — see `PROVENANCE.md`.

## The seam

`encode_text` depends only on the phrases and `encode_image` only on the picture, and both are
cached. That matters in one direction more than the other: on the base geometry the image tower is
about 89% of a call, so a second vocabulary on a picture already seen costs roughly a twentieth of
the first — while a second picture on a vocabulary already encoded saves only the few percent the
text tower was. `tools/bench/owlv2.py` measures both.

```python
detector.encode_text(["kettle", "a mug"])   # valid for every image afterwards
detector.encode_image(image)                # valid for every vocabulary afterwards
```

## Boxes, not masks

Pair it with `sam2_deploy` or `edgetam_deploy` if you want a mask: a box from here is a prompt
there.

## What this is not

The video path, image-guided detection ("find more like this crop") and running at an
unpublished resolution are all absent. `PROVENANCE.md` says what was left behind and why, and
lists every place this diverges from `transformers` — each one found by comparing numbers, and
each one small enough that a tolerance would have hidden it.
