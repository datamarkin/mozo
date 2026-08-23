# clip_deploy

Deployment-only CLIP. MIT code *and* weights, extracted from `openai/CLIP` and reduced to the two
encoders, with no `clip`, `ftfy`-only-for-CLIP or TorchScript dependency at inference.

An image and a phrase become vectors in one shared space, so the dot product between them says how
well they match. That single property gives both zero-shot classification and search-by-words.

```python
from mozo.vendors.clip_deploy import Encoder, SPECS
from mozo.image import load_image

encoder = Encoder("torch-fp32.pth", SPECS["base"])     # device="cpu"

encoder.classify(load_image("aisle.jpg"), ["a forklift", "a person"])
# tensor([0.1592, 0.1774])   cosine similarities, in prompt order

encoder.encode_image(load_image("aisle.jpg"))          # (1, 512), L2-normalised
encoder.encode_text(["a forklift", "a person"])        # (2, 512), L2-normalised
```

## The towers load separately

Each is built on first use and never before, because they are usually wanted apart:

| what you call | what gets built |
|---|---|
| `encode_text` | text tower only, 63.4M parameters |
| `encode_image` | vision tower only, 87.8M |
| `classify` | both, 151.3M |

An ingest job that embeds a corpus never allocates the text tower; a query service answering
searches never allocates the vision tower. Nothing to configure.

## Scores are similarities, not probabilities

`classify` returns cosine similarities. They are not softmaxed, do not sum to one, and may be
negative. They are also compressed — a good match sits well below 1.0 — so compare them against
each other or a calibrated threshold, never against an intuition about percentages.

A softmax would be worse, not better: it is relative to whichever phrases you happened to pass, so
one phrase always scores 1.00 and adding a phrase moves every other number.

## Variants

`base` (ViT-B/32), `base-16` (ViT-B/16), `large` (ViT-L/14), `large-336` (ViT-L/14@336px).

OpenAI also publishes five ResNet variants, which use a different image tower and are not carried.

## Fidelity

Bit-identical to upstream at every stage — preprocessed tensor, token ids, image features, text
features, similarities. Run the gate with

```bash
python tools/verify/clip.py base --upstream /path/to/CLIP
```

See `PROVENANCE.md` for what was taken, what was left, the two deliberate divergences, and the ten
traps that run and are wrong — including the one where Python's operator precedence changes the
logits by 1.9e-06.
