# Provenance

This package is a deployment-only extraction of **OWLv2**'s detection path.

It derives from `transformers/models/owlv2` (Apache-2.0, © The Google Research Authors and The
HuggingFace Team) — **not** from `google-research/scenic`, whose OWL-ViT project is the authors'
own release and is written in JAX/Flax. Both are Apache-2.0, so this is not a licensing choice
the way SAM 3's was; it is that the PyTorch implementation is the one whose numbers the published
PyTorch checkpoints reproduce, and the one mozo can be checked against on every run.

The tokenizer and its vocabulary are OpenAI's CLIP (MIT). See `NOTICE`.

**The weights are Apache-2.0 too.** All six published checkpoints, code and weights alike. That is
the reason this family is in mozo: its only text-prompted sibling, SAM 3, carries Meta's SAM
License, which is not open source and whose field-of-use restrictions flow to downstream users.
This package does not redistribute Google's checkpoints; `tools/fetch/owlv2.py` obtains them.

## What was taken

| from | what |
|---|---|
| `modeling_owlv2.py` | the CLIP ViT trunk, the CLIP text tower, the class, box and objectness heads, the box-position bias and the class-token merge |
| `image_processing_owlv2.py` | the pad-then-resize preprocessing and the box descale, rewritten in `image.py` |
| `configuration_owlv2.py` | the numeric geometry, written out as frozen dataclasses in `config.py` |
| `openai/CLIP` | the byte-pair tokenizer and `bpe_simple_vocab_16e6.txt.gz` |
| the published `tokenizer_config.json` | the two settings that are not `CLIPTokenizer` defaults: `pad_token="!"` and a context length of 16 |

The checkpoint is consumed **exactly as Google publishes it** — `pytorch_model.bin`, no repacking,
no pruning, no mozo-format artifact. `checkpoint.py` translates four key prefixes at load time.

## What was deliberately left behind

**Image-guided detection.** OWLv2 can be queried with a cropped example instead of a phrase, and
`Owlv2ForObjectDetection.image_guided_detection` implements it. It is not carried: it reuses the
same trunk and heads, so it adds no weights, and its published quality is poor enough to have its
own upstream issue (huggingface/transformers#26920). If it is wanted later, nothing here is in the
way — it needs `embed_image_query` and the NMS its postprocessing applies, and neither touches a
line of this package.

**`interpolate_pos_encoding`.** Upstream offers it to run at a resolution the position embedding
was not trained for. This package always runs at the published one, so there is nothing to
interpolate and no branch that can silently take the wrong path.

**The training code, the evaluation harness, and the contrastive head.** Weights present in every
published checkpoint that this package does not build:

- `owlv2.visual_projection` (1.5 MB) and `owlv2.logit_scale` — CLIP's contrastive head. Upstream
  reaches the detection path through the full `Owlv2Model.forward`, so it projects the class
  token, dots it against the prompt, and scales by the learned temperature on **every call**,
  then discards all three. `text_projection` is *not* in this list: it is on the detection path,
  and dropping it would be wrong.

Nothing here is built and left unrun. Everything that loads, runs.

## Where this diverges from `transformers`, and why

Every item below was found by comparing numbers against the reference, not by reading. Each one
is a place where following the source faithfully — or following the obvious reading of it —
produced confident, plausible, wrong output. `tools/verify/owlv2.py` is where they stay found.

| divergence | cost of the other choice |
|---|---|
| The prompt cache is keyed on the **whole vocabulary**, not phrase by phrase. | The text tower runs the phrases as a batch, and a batched matmul does not produce bit-identical rows to the same matmul run one row at a time. **2.5e-07** on the score, and `["cat"]` then `["cat","dog"]` disagreeing about `"cat"`. |
| Pixels are **multiplied** by one two-hundred-and-fifty-fifth. | Upstream carries the reciprocal as a constant. Dividing by 255 is the same number and not the same float: **9.5e-07** on the trunk's input. |
| The resize factor is computed in **float32**, through a tensor. | Upstream divides one tensor of side lengths by another, which lands in float32. Python's float64 gives a standard deviation that differs in the seventh digit, a different kernel, and **9.5e-07** on the trunk's input — but only for images whose longest side is not an exact multiple of the model's. |
| The Gaussian kernel is a **softmax of the negated squared distance** over a span scaled by `2*sqrt(2)`. | The readable `exp` divided by its own sum is the same function and **3.0e-07** different on the blurred tensor, **1.4e-06** by the trunk. |
| The tokenizer reproduces **`CLIPTokenizer`**, not OpenAI's `SimpleTokenizer`. | OpenAI's cleaner runs `ftfy` and unescapes HTML twice; `CLIPTokenizer` does NFC, whitespace collapse and lowercase, and nothing else. Three prompts in thirty-six tokenize differently, from the second token onward. |
| `!` is split out as an added token with id **0**, and the attention mask is **returned** rather than derived from the ids. | The published config makes `!` the padding token, which puts it in the added-token table. `ids != 0` is the obvious mask and it drops a real token out of the attention on any prompt containing an exclamation mark. |

One more thing was measured and is *not* a divergence, because it is a fact about the reference
rather than about this package: `from_pretrained` places its tensors at whatever offset the
checkpoint file had, and BLAS picks different vectorised paths for a matrix whose storage happens
to be page-aligned. The 512×512 projection in the text tower moved by **1.2e-07** on that alone.
The gate re-allocates the reference's parameters before comparing, so what it compares is
arithmetic rather than addresses.

## Export

OWLv2 has a clean single-graph export and mozo publishes none. Both halves of that are deliberate.

Unlike SAM 2 and SAM 3, this model traces in one piece: `torch.onnx.export` at opset 17 produces a
619 MB graph in five seconds, with a dynamic axis for the number of prompts, and it runs under
`onnxruntime` agreeing with torch to 3.9e-03 on the logits. It was then **2× slower than torch on
the same CPU** — the attention over 3,601 tokens comes out of the trace as MatMul, Softmax, MatMul,
where torch dispatches a fused kernel. An artifact that `mozo.runtimes` would never select, at
619 MB a variant, is 2.5 GB of published bytes with no user, so there is none.

That leaves the seam in `network.py` unexploited by any graph runtime, which is not why it exists
— it exists because `encode_text` and `encode_image` depend on different things, and
`tools/bench/owlv2.py` measures what that is worth. If someone revisits this, the two-graph split
along the same seam is the shape to try, and the blocker is the same one SAM 2 has: mozo has no
multi-graph runner.
