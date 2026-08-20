#!/usr/bin/env python3
"""Export RF-DETR variants to ONNX and drop them into the local ``weights/`` tree.

This runs once, on a machine you control, and never ships. Users receive the ``.onnx`` file, not
the code that produced it -- which is the point of exporting at all: the graph carries the
architecture, so the 8,500 lines of ``rfdetr_deploy`` stay off the serving path.

    python tools/export/rfdetr.py small
    python tools/export/rfdetr.py small medium large --revision 2026-08-18

Every export is checked against the torch model it came from before it is written. An export that
disagrees is not published: a silent numerical divergence between two artifacts of the same model
is the one failure this whole scheme exists to prevent.

The check runs on real photographs and compares *detections*, not raw tensors. RF-DETR's two-stage
decoder picks its queries by top-k, and on synthetic noise the scores are near-uniform, so torch
and ONNX legitimately select different queries and 90% of the output disagrees while both remain
correct. On a photograph the scores separate and selection is stable. Raw-tensor tolerances test
the tie-breaking of ``topk``; only the detections test the model.

Three artifacts are produced per variant, all landing in
``weights/rfdetr/<variant>/<revision>/`` where the manifest generator picks them up by stem:

``onnx-fp32``    the graph, exported from the torch model.
``coreml-fp32``  the same model as a CoreML package, via upstream's own exporter, zipped because
                 an ``.mlpackage`` is a directory and an artifact is a file. Five times faster
                 than torch on Apple silicon (10.8 ms against 53.3 ms on nano) at a worst output
                 delta of 0.001.

There is deliberately no fp16 path. It was built, measured against upstream on ten photographs,
and removed: fp16 lost a detection on three of eight variants and moved a score by 0.1119 where
fp32 moves 0.0005; on Apple silicon it was no faster (52 ms against 53), on CPU it was 3-9x
*slower* (2097 ms against 236), and through CoreML it was both wrong and slower (delta 5.06 at
16.5 ms against 0.001 at 10.8). Upstream's own guidance agrees. Should CUDA tensor cores ever
justify revisiting it, start from the measurements rather than from this note.

CoreML goes through ``rfdetr[coreml]`` rather than coremltools directly. Calling coremltools
straight fails on this architecture -- ``meshgrid`` via the traced path, an unsupported ``__and__``
node via ``torch.export`` -- because several ops need registering into
``coremltools._TORCH_OPS_REGISTRY`` first. Upstream already does that, so importing their
converter is what makes the vendored model convertible; there is no reason to rediscover it.

CoreML is exported at fp32 deliberately. At fp16 it is both wrong and slower: worst delta 5.06
against 0.001, at 16.5 ms against 10.8.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
# This file is called rfdetr.py, so its own directory would shadow the package it imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(ROOT))

from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.rfdetr_deploy import Predictor  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: Graph output order is fixed by ``LWDETR.forward_export``: boxes, then logits, then masks.
#: They are named for the keys the post-processor reads, so the artifact carries the mapping and
#: no consumer has to keep a table that can silently disagree with the graph.
OUTPUT_NAMES = ("pred_boxes", "pred_logits", "pred_masks")
INPUT_NAME = "images"

OPSET = 17

#: What the export must reproduce, per fixture image. Counts and labels are exact because a
#: detection appearing or vanishing is a behaviour change however small the tensor delta was.
#:
#: The box bound is measured rather than assumed. Across all eight variants, torch and the
#: published ONNX agree to 0.0002-0.0008 px -- except nano, which is 0.0876. So 0.2 px covers the
#: worst real case twice over and still fails on anything an order of magnitude larger. It was a
#: whole pixel before, which is eleven times the worst disagreement that has ever occurred here
#: and a thousand times the typical one: a bound nobody had measured, wide enough to pass an
#: export that had genuinely broken.
#:
#: nano being a hundred times its siblings is not explained. It is well inside the bound and the
#: detections match exactly, so it is not blocking anything -- but it is the one number here that
#: looks like a fact about the model rather than about float arithmetic.
BOX_TOLERANCE_PX = 0.2
SCORE_TOLERANCE = 0.01

#: Detections below this are noise and would make the count comparison meaningless.
THRESHOLD = 0.5

#: Where the photographs used for verification live.
FIXTURES = ROOT / "tests" / "fixtures" / "images"


def _fixtures() -> list[Path]:
    """Return the photographs to verify against.

    Raises:
        SystemExit: If the fixtures directory is empty -- an export verified against nothing is
            an export that has not been verified.
    """
    images = sorted(p for p in FIXTURES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"no fixture images in {FIXTURES}. Add photographs to verify against.")
    return images


def _detections(predictor: Predictor, raw: tuple, sizes: list[tuple[int, int]]) -> list[dict]:
    """Post-process raw model outputs into detections, the way ``Predictor.predict`` does.

    Written out here rather than reused because the torch and ONNX paths produce their raw
    outputs differently but must share every step after that.
    """
    named = dict(zip(OUTPUT_NAMES, raw))
    results = predictor.postprocess(
        named, target_sizes=torch.tensor(sizes), score_threshold=THRESHOLD
    )
    return [{k: v[r["scores"] > THRESHOLD] for k, v in r.items()} for r in results]


def _export(predictor: Predictor, destination: Path) -> tuple[str, ...]:
    """Write *predictor*'s model to *destination* as ONNX and return its output names."""
    resolution = predictor.spec.resolution
    dummy = torch.randn(1, 3, resolution, resolution)

    with torch.inference_mode():
        arity = len(predictor.model(dummy))
    names = OUTPUT_NAMES[:arity]

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        predictor.model,
        (dummy,),
        str(destination),
        input_names=[INPUT_NAME],
        output_names=list(names),
        dynamic_axes={name: {0: "batch"} for name in (INPUT_NAME, *names)},
        opset_version=OPSET,
        dynamo=False,
    )
    return names


def _compare(image: Path, want: dict, got: dict) -> str:
    """Compare one image's detections between the two runtimes.

    Returns:
        A one-line summary for the report.

    Raises:
        SystemExit: If the count, labels, boxes or scores moved beyond tolerance.
    """
    if len(want["scores"]) != len(got["scores"]):
        raise SystemExit(
            f"{image.name}: torch found {len(want['scores'])} detections, ONNX found "
            f"{len(got['scores'])}. Not writing this artifact."
        )
    if not torch.equal(want["labels"], got["labels"]):
        raise SystemExit(f"{image.name}: ONNX assigned different classes. Not writing this artifact.")

    box_delta = float((want["boxes"] - got["boxes"]).abs().max()) if len(want["scores"]) else 0.0
    score_delta = float((want["scores"] - got["scores"]).abs().max()) if len(want["scores"]) else 0.0

    if box_delta > BOX_TOLERANCE_PX:
        raise SystemExit(f"{image.name}: boxes moved {box_delta:.3f} px. Not writing this artifact.")
    if score_delta > SCORE_TOLERANCE:
        raise SystemExit(f"{image.name}: scores moved {score_delta:.4f}. Not writing this artifact.")

    return (
        f"  {image.name:<24} {len(want['scores']):>3} detections   "
        f"boxes {box_delta:.4f} px   scores {score_delta:.5f}"
    )


def _verify(predictor: Predictor, path: Path, reference: list[tuple]) -> list[str]:
    """Check the exported graph reproduces *reference*, the torch model's own detections.

    Raises:
        SystemExit: If any image's detections moved.
    """
    runner = OnnxRunner(path, device="cpu")
    report = []
    for image, batch, sizes, want in reference:
        raw = runner(batch.numpy())
        got = _detections(predictor, tuple(torch.from_numpy(a) for a in raw), sizes)[0]
        report.append(_compare(image, want, got))
    return report


def _export_coreml(predictor: Predictor, destination: Path, names: tuple[str, ...]) -> None:
    """Export the model as a CoreML package, with its inputs and outputs named like the graph's.

    CoreML takes its names from the traced program, so the input arrives as ``tensors`` (the
    parameter name of ``forward_export``) and the outputs as whatever MIL called them
    (``concat_3``, ``linear_102``). Renaming them to match the ONNX artifact means a consumer
    reads output names off the artifact and never needs a per-runtime table.
    """
    import coremltools as ct
    from rfdetr.export._coreml.converter import export_coreml

    resolution = predictor.spec.resolution
    with tempfile.TemporaryDirectory() as scratch:
        produced = export_coreml(
            predictor.model, torch.randn(1, 3, resolution, resolution), scratch,
            variant_name="model", verbose=False, compute_precision=ct.precision.FLOAT32,
        )
        model = ct.models.MLModel(str(produced))
        spec = model.get_spec()
        ct.utils.rename_feature(spec, spec.description.input[0].name, INPUT_NAME)
        for output, wanted in zip(list(spec.description.output), names):
            ct.utils.rename_feature(spec, output.name, wanted)

        renamed = Path(scratch) / "renamed.mlpackage"
        ct.models.MLModel(spec, weights_dir=model.weights_dir).save(str(renamed))
        shutil.make_archive(str(destination.with_suffix("")), "zip", root_dir=renamed)


def export_variant(variant: str, revision: str | None, weights_dir: Path) -> None:
    """Export one variant, verify it against its own torch detections, and place it in the tree."""
    print(f"\n=== rfdetr/{variant}")
    checkpoint = resolve("rfdetr", variant, revision=revision)
    predictor = Predictor.from_pretrained(f"rfdetr-{variant}", weights=checkpoint, device="cpu")

    # Capture the reference before switching to export mode: ``export()`` replaces ``forward``,
    # so the eager path is unreachable afterwards.
    reference = []
    for image in _fixtures():
        want = predictor.predict(str(image), threshold=THRESHOLD)[0]
        batch, sizes = predictor.preprocess([str(image)])
        reference.append((image, batch, sizes, want))
    predictor.model.export()

    destination = weights_dir / "rfdetr" / variant / checkpoint.parent.name / "onnx-fp32.onnx"

    # Export to scratch first: a verification failure must not leave a bad artifact where the
    # manifest generator would pick it up.
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / "onnx-fp32.onnx"
        names = _export(predictor, staged)
        print(f"  exported  {predictor.spec.resolution}x{predictor.spec.resolution}, "
              f"opset {OPSET}, outputs {', '.join(names)}, dynamic batch")
        for line in _verify(predictor, staged, reference):
            print(line)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(staged, destination)

    revision_dir = destination.parent
    print(f"  onnx-fp32 {destination.stat().st_size / 1e6:.1f} MB")

    coreml = revision_dir / "coreml-fp32.zip"
    _export_coreml(predictor, coreml, names)
    print(f"  coreml-fp32 {coreml.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+", help="variant names, e.g. small seg-medium")
    parser.add_argument("--revision", default=None, help="published revision to export (default: latest)")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights",
                        help="local weights tree to write into (default: ./weights)")
    args = parser.parse_args()

    for variant in args.variants:
        export_variant(variant, args.revision, args.weights_dir)

    print("\nRun tools/generate_manifest.py to pick these up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
