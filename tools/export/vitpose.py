#!/usr/bin/env python3
"""Export ViTPose variants to ONNX and drop them into the local ``weights/`` tree.

This runs once, on a machine you control, and never ships. Users receive the ``.onnx`` file, not
the code that produced it -- which is the point of exporting at all: the graph carries the
architecture, so the trunk stays off the serving path.

    python tools/export/vitpose.py small
    python tools/export/vitpose.py small base large huge --revision 2026-08-25

Every export is checked against the torch model it came from before it is written. An export that
disagrees is not published: a silent numerical divergence between two artifacts of the same model
is the one failure this whole scheme exists to prevent.

**The check compares joints, not tensors.** A heatmap is decoded by taking each channel's brightest
cell, so two runtimes that differ in the last float bit can still pick different cells where a
heatmap is flat -- and a joint that jumps a cell is a visible change however small the tensor delta
was. Comparing positions in the frame's own pixels is what tests the model rather than the
tie-breaking of ``argmax``.

**And it uses real people.** The boxes come from ``rfdetr/medium`` run on the fixture photographs,
because a top-down model asked about a box with no person in it produces a flat heatmap, which is
exactly the near-tie the paragraph above is about. A cross-family dependency in bootstrap tooling
is worth it to avoid verifying against noise.

One artifact is produced per variant, landing in ``weights/vitpose/<variant>/<revision>/`` where
the manifest generator picks it up by stem:

``onnx-fp32``    the graph, exported from the torch model, dynamic in the batch dimension because
                 a batch here is a crowd -- one entry per person in the frame.

**There is deliberately no CoreML path.** It was built and measured before being left out, so this
is a result rather than an omission. ``coremltools`` converts this architecture directly -- unlike
RF-DETR, which needs upstream's own converter to register several ops first -- and the joints agree
to 0.0005 px. It is simply not faster: 22.9 ms against torch-on-MPS's 22.6 for five people on
``small``, and 44.0 against 44.2 on ``base``. A fixed batch shape does not change that (22.9), nor
does fp16 (22.8), and the Neural Engine is three and a half times *worse* (81.1).

That is the opposite of RF-DETR, where CoreML is five times faster than torch, and the likely reason
is that this trunk is a plain ViT: matmuls and layer norms, which Metal already runs at full speed,
where RF-DETR leans on ops MPS handles poorly. Publishing it anyway would make CoreML the ``auto``
choice on Apple silicon -- ``_PREFERENCE`` puts it first there -- so users would download a second
copy of every variant and install ``coremltools`` to run at the same speed.

Should that change, start from these numbers rather than from this note.

There is no fp16 path either. Nothing has measured one against the joints.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
# This file is called vitpose.py, so its own directory would shadow the package it imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import DETECTOR, fixtures, person_boxes, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.vitpose_deploy import SPECS, Predictor  # noqa: E402
from mozo.vendors.vitpose_deploy.image import preprocess  # noqa: E402
from mozo.vendors.vitpose_deploy.postprocess import to_keypoints  # noqa: E402
from mozo.weights import resolve  # noqa: E402
from tools.fetch.vitpose import REVISION  # noqa: E402

INPUT_NAME = "pixel_values"
OUTPUT_NAME = "heatmaps"
OPSET = 17

#: What the export may move a joint by, in the frame's own pixels. Measured rather than assumed:
#: across all four variants the published graphs agree with torch to under a thousandth of a pixel.
#: A tenth of a pixel is a hundred times the worst real case and still an order of magnitude below
#: anything a person could see, so it fails on a genuine break and not on float arithmetic.
#:
#: PixelFlow rounds coordinates to 0.01 downstream, which means anything under 0.005 is invisible
#: by the time a caller sees it. This bound is deliberately tighter than that: an export that had
#: started to drift should fail here rather than be hidden by the rounding.
JOINT_TOLERANCE_PX = 0.1

#: What it may move a confidence by. Heatmap peaks are read straight off the graph, so this is
#: float noise and nothing else.
CONFIDENCE_TOLERANCE = 1e-4


def people(images: list[Path]) -> dict[Path, np.ndarray]:
    """Person boxes per fixture image, dropping the photographs that have nobody in them."""
    found = {image: boxes for image, boxes in person_boxes(images).items() if len(boxes)}
    for image in images:
        if image not in found:
            print(f"  {image.name}: no people, skipping -- an empty heatmap proves nothing")
    if not found:
        raise SystemExit(
            f"{DETECTOR[0]}/{DETECTOR[1]} found nobody in any fixture image, so there is nothing "
            "to verify an export against. Add a photograph with a person in it."
        )
    return found


def _export(model: torch.nn.Module, spec, destination: Path) -> None:
    """Write *model* to *destination* as ONNX.

    The expert is bound to a constant here, which is what makes this exportable at all: the
    mixture-of-experts block indexes a ``ModuleList``, and an index that is a Python integer traces
    to a plain subgraph. Only COCO's expert matches the published head, so nothing is lost -- see
    ``vitpose_deploy/predictor.py``.
    """
    from mozo.vendors.vitpose_deploy.predictor import EXPERT

    class OneExpert(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.model = wrapped

        def forward(self, pixel_values):
            return self.model(pixel_values, EXPERT)

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        OneExpert(model),
        (torch.randn(2, 3, spec.height, spec.width),),
        str(destination),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={name: {0: "people"} for name in (INPUT_NAME, OUTPUT_NAME)},
        opset_version=OPSET,
        dynamo=False,
    )


def _compare(image: Path, want: np.ndarray, got: np.ndarray) -> str:
    """Compare one image's joints between the two runtimes.

    Returns:
        A one-line summary for the report.

    Raises:
        SystemExit: If a joint or a confidence moved beyond tolerance.
    """
    position = float(np.abs(want[..., :2] - got[..., :2]).max())
    confidence = float(np.abs(want[..., 2] - got[..., 2]).max())

    if position > JOINT_TOLERANCE_PX:
        raise SystemExit(f"{image.name}: joints moved {position:.4f} px. Not writing this artifact.")
    if confidence > CONFIDENCE_TOLERANCE:
        raise SystemExit(
            f"{image.name}: confidences moved {confidence:.6f}. Not writing this artifact.")

    return (f"  {image.name:<24} {want.shape[0]:>2} people   "
            f"joints {position:.5f} px   confidences {confidence:.7f}")


def export_variant(variant: str, revision: str | None, boxes: dict[Path, np.ndarray],
                   weights_dir: Path) -> None:
    """Export one variant, verify it against its own torch joints, and place it in the tree."""
    print(f"\n=== vitpose/{variant}")
    checkpoint = resolve("vitpose", variant, revision=revision)
    predictor = Predictor(checkpoint, variant, device="cpu")
    spec = SPECS[variant]

    reference = []
    for image, xyxy in boxes.items():
        batch, centers, scales = preprocess(load_image(str(image)), xyxy, spec.height, spec.width)
        want = to_keypoints(predictor.heatmaps(batch), centers, scales)
        reference.append((image, batch, centers, scales, want))

    destination = weights_dir / "vitpose" / variant / checkpoint.parent.name / "onnx-fp32.onnx"

    # Export to scratch first: a verification failure must not leave a bad artifact where the
    # manifest generator would pick it up.
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / "onnx-fp32.onnx"
        _export(predictor.model, spec, staged)
        print(f"  exported  {spec.height}x{spec.width}, opset {OPSET}, output {OUTPUT_NAME}, "
              "dynamic batch")

        # An artifact is one file. Past protobuf's 2 GB ceiling ``torch.onnx.export`` silently
        # writes the weights beside the graph instead of inside it, leaving a few hundred KB of
        # stub that loads perfectly here -- where its siblings still exist -- and fails wherever it
        # is published to. Checking the directory catches that without reading the protobuf.
        spilled = sorted(p.name for p in staged.parent.iterdir() if p != staged)
        if spilled:
            raise SystemExit(
                f"the graph did not fit in one file: {len(spilled)} weight files were written "
                f"beside it ({', '.join(spilled[:3])}...). ONNX falls back to external data past "
                "protobuf's 2 GB limit, and mozo publishes one file per artifact. Not writing "
                "this artifact."
            )

        runner = OnnxRunner(staged, device="cpu")
        for image, batch, centers, scales, want in reference:
            got = to_keypoints(runner(batch.numpy())[0], centers, scales)
            print(_compare(image, want, got))

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, destination)

    print(f"  onnx-fp32 {destination.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", required=True, revision=REVISION)
    args = parser.parse_args()

    unknown = [v for v in args.variants if v not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(SPECS)}")

    print(f"finding people in the fixtures with {DETECTOR[0]}/{DETECTOR[1]}")
    boxes = people(fixtures())

    for variant in args.variants:
        export_variant(variant, args.revision, boxes, args.weights_dir)

    print(f"\n{len(args.variants)} graphs written. Run tools/generate_manifest.py to publish them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
