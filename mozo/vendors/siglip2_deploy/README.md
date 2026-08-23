# siglip2_deploy

Deployment-only SigLIP 2. Apache-2.0 code *and* weights, extracted from
`transformers/models/siglip` and reduced to the two encoders, with no `transformers`,
`tokenizers`, `sentencepiece` or `safetensors` dependency at inference.

An image and a phrase become vectors in one shared space. Unlike CLIP, the model was trained pair
by pair with a sigmoid loss and carries a learned bias alongside its learned temperature — so the
score for one image and one phrase means something on its own, with no competing phrases to
normalise against.

```python
from mozo.vendors.siglip2_deploy import Encoder, SPECS
from mozo.image import load_image

encoder = Encoder("torch-fp32.pth", SPECS["base-224"])   # device="cpu"

encoder.classify(load_image("aisle.jpg"), ["a forklift", "a person"])
# tensor([0.0004, 0.0349])   probabilities, in prompt order, each independent of the others

encoder.encode_image(load_image("aisle.jpg"))            # (1, 768), L2-normalised
encoder.encode_text(["a forklift", "a person"])          # (2, 768), L2-normalised
```

Five variants — `base-224`, `base-256`, `so400m-384`, `so400m16-256`, `giant-384` — the most-used
five of the fifteen fixed-resolution models Google publishes, and 89% of the downloads across
those fifteen. See `PROVENANCE.md` for what the other ten are and why they are absent.
Multilingual: the text tower carries Gemma's 256,000-piece vocabulary.

Each tower builds on first use. An ingest job that only calls `encode_image` never allocates the
text tower, which here is most of the checkpoint — the vocabulary alone is 786 MB of a `base`
variant and 1,180 MB of an `so400m` one.

## What the score is

`sigmoid(cos × exp(logit_scale) + logit_bias)`, which is what upstream's own examples print. Adding
a phrase moves no other phrase's number and the set does not sum to one; every phrase can be near
zero, which is the useful answer when none of them fits.

It is **not** a calibrated class probability. Nothing in the training made classes compete, so it
says how well this phrase matches this image, not P(class | image). Calibrate a threshold on your
own data before deciding anything with it.

## Verified

`tools/verify/siglip2.py` compares every stage against `transformers` with `torch.equal` and no
tolerance: preprocessed pixels, token ids, image features, text features, logits and the sigmoid.
See `PROVENANCE.md` for what that guarantee does and does not extend to.
