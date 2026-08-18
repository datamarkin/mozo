# Provenance

This package is a deployment-only extraction of [RF-DETR](https://github.com/roboflow/rf-detr).

| | |
|---|---|
| Upstream repository | `https://github.com/roboflow/rf-detr` |
| Upstream commit | `85bf908534a9fc6336fae2d5b727ef0ff0e211ef` |
| Upstream version | `1.10.0.dev` (after tag `1.9.0`) |
| Extracted | 2026-08-18 |
| Verified against | `torch` 2.11.0, `torchvision` 0.26.0, Python 3.10 |

Record the commit before re-syncing: without it, diffing a later upstream against this
copy means guessing which revision it started from.

## What was taken

Verbatim, apart from rewriting imports to relative form:

```
models/{lwdetr,transformer,postprocess,math,position_encoding,_types}.py
models/heads/{segmentation,keypoints}.py
models/ops/**                      (pure-PyTorch deformable attention; no CUDA extension)
models/backbone/{backbone,base,dinov2,projector}.py
models/backbone/dinov2_configs/*.json
utilities/{box_ops,tensors,keypoints,logger}.py
```

Modified:

- `models/backbone/dinov2_with_windowed_attn.py` — reparented onto `_hf_compat`; the plain
  `...Model` and `...ForImageClassification` classes, head pruning, and the Transformers
  doc-string decorators were removed; `set_attn_implementation` moved onto the shared base
  class so the backbone keeps its eager/sdpa switch.
- `models/lwdetr.py` — truncated after `build_model()`. Everything below it built the
  criterion and matcher, which pulled in `scipy`.
- `models/backbone/dinov2.py` — the non-windowed branch raises instead of calling
  `transformers.AutoBackbone.from_pretrained`. No released variant uses that branch.
- `models/backbone/backbone.py` — dropped the `peft` LoRA merge from `export()`; this build
  has no fine-tuning path, so the encoder is never a `PeftModel`.

Written for this package:

- `config.py` — variant specs as frozen dataclasses, replacing the Pydantic `ModelConfig`
  hierarchy. Values were dumped from upstream's resolved builder namespace, not transcribed.
- `weights.py` — checkpoint resolution, download, and a load that refuses to leave any
  parameter uninitialized.
- `predictor.py` — preprocessing, forward, post-processing.
- `models/backbone/_hf_compat.py` — the Transformers base classes the backbone needs.

## What was deliberately left behind

Training and the Lightning stack, all export backends (ONNX, CoreML, TFLite, ExecuTorch,
TensorRT), LoRA, dataset loaders, evaluation, visualization, the CLI, Roboflow deployment,
distributed utilities, `optimize_for_inference` / `torch.jit.trace`, non-RGB channel
adaptation, and the `weights_only=False` pickle fallback.

Dropping these is what removes `transformers`, `supervision`, `pydantic`, `opencv-python`,
`matplotlib`, `scipy`, `requests`, `tqdm`, and `peft` from the install.

## Verification

Validated against upstream `rfdetr` at the commit above, on CPU:

- **Standalone** — no import outside stdlib, `torch`, `torchvision`, `numpy`, `PIL`; no absolute
  self-imports; importing the package and building a model pulls in none of the shed dependencies.
- **Structural** — state-dict keys and shapes identical to upstream for all nine variants.
- **Forward pass** — given the same weights and input, `max|delta| = 0.0` across `pred_logits`,
  `pred_boxes`, `pred_masks`, and `pred_keypoints`. Bitwise identical, not merely close.
- **End to end** — on a real 760x428 photograph with the published checkpoints, every variant
  returns the same detection count, identical class ids, and `max|delta| = 0.0` on scores, boxes,
  masks, and keypoints.

Comparisons must pin both sides to the same device. Upstream picks its default per host (MPS on
Apple silicon, CUDA where available); an MPS-vs-CPU comparison measures backend kernels, not the
two implementations, and shows up as a spurious ~1e-5 drift.

The `verify_vendor.py` script in the upstream working tree reproduces all of the above. It is
extraction scaffolding, not part of this package, and does not travel with it.
