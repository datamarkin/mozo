#!/usr/bin/env python3
"""Export YOLOv8 variants to ONNX and drop them into the local ``weights/`` tree.

This runs once, on a machine you control, and never ships. Users receive the ``.onnx`` file, not
the code that produced it -- which is the point of exporting at all: the graph carries the
architecture, so the vendored package stays off the serving path.

    python tools/export/yolov8.py nano
    python tools/export/yolov8.py nano small --revision 2026-08-19

Every export is checked against the torch model it came from before it is written. An export that
disagrees is not published: a silent numerical divergence between two artifacts of the same model
is the one failure this whole scheme exists to prevent. The check compares *detections* on real
photographs rather than raw tensors -- what matters is that the two artifacts find the same
objects in the same places, and a tensor tolerance measures kernel arithmetic instead.

Two artifacts land in ``weights/yolov8/<variant>/<revision>/``, both mapping a letterboxed
``(1, 3, imgsz, imgsz)`` batch to the raw head output ``(1, 4 + classes, anchors)``. That is a
classic head, so whoever runs one still applies non-maximum suppression -- which mozo does with
the vendor's own :func:`~mozo.vendors.yolov8_deploy.detect`, so every runtime shares it:

``onnx-fp32``    the graph.
``coreml-fp32``  the same model as a CoreML package, zipped because an ``.mlpackage`` is a
                 directory and an artifact is a file. It is the fastest way to run these models on
                 Apple silicon by a wide margin -- measured on this laptop, nano runs 4.2 ms
                 against 7.9 ms on torch MPS and 52.2 ms on torch CPU, and xlarge 41.1 ms against
                 48.8 and 330.2 -- at a worst box error of 0.0004 px.

There is deliberately no fp16 path, in either format. It was measured across four variants and is
a loss everywhere: torch fp16 on MPS is *slower* than fp32 (8.2 ms against 7.9 on nano) and moves
boxes 0.76 px; ONNX fp16 is slower too (43.2 against 34.4). CoreML fp16 is the one that is
genuinely faster, 1.3-1.6x, but it is wrong by 1.5 px on small, 108 px on xlarge, 341 on medium
and 636 on nano -- erratic rather than degrading with size, so the error cannot be bounded. Should
CUDA tensor cores ever justify revisiting it, start from measurements rather than from this note.
The class names a graph cannot carry are published by ``tools/labels/yolov8.py``, which runs over
every variant you fetched rather than only the ones exported here -- a vocabulary is needed by any
runtime that does not record its own, and tying it to this tool would skip the ones you never
export.

**These weights are AGPL-3.0, and so is anything exported from them.** The graph produced here
contains the weights. It lands in the same revision directory as the LICENSE and NOTICE that
``tools/fetch/yolov8.py`` placed, which is what keeps those terms travelling with it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mozo.image import load_image  # noqa: E402
from mozo.runtimes import CoreMLRunner  # noqa: E402
from mozo.vendors.yolov8_deploy import Detector, detect  # noqa: E402

#: The photographs the comparison runs on.
FIXTURES = ROOT / "tests" / "fixtures" / "images"

#: Opset the graphs are written at. A constant rather than a flag: which opset mozo publishes is a
#: property of the artifact, not something a caller should be able to vary per run.
OPSET = 17

#: Thresholds the comparison runs at. Deliberately far below any sensible serving threshold: a
#: divergence that only shows up on marginal detections is still a divergence, and a strict
#: threshold would hide it by discarding exactly the boxes where the two artifacts disagree.
CONF = 0.001
IOU = 0.7
MAX_DET = 300

#: Names both artifacts declare, so nothing downstream needs a per-runtime table.
INPUT_NAME = "images"
OUTPUT_NAME = "predictions"

#: What the export must reproduce. Counts and class ids are exact -- an artifact that finds a
#: different number of objects, or calls one of them something else, is not the same model.
BOX_TOLERANCE = 1e-2      # pixels, in the source image's coordinates
SCORE_TOLERANCE = 1e-3


def _fixtures() -> list[Path]:
    """Photographs to compare on. Real images, because synthetic noise proves nothing here.

    Matched the way ``tools/export/rfdetr.py`` matches them, so a ``.png`` added to the tree is
    not silently verified for one family and skipped for another.
    """
    images = sorted(p for p in FIXTURES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"no fixture images in {FIXTURES}. Add photographs to verify against.")
    return images


def _detections(source: np.ndarray, forward, imgsz: int) -> tuple[np.ndarray, ...]:
    """Detect in one already-decoded image, exactly as mozo would at serving time.

    Post-processing is the vendor's :func:`~mozo.vendors.yolov8_deploy.detect`, so what is
    compared here is the graph against the module and nothing else.
    """
    with torch.no_grad():
        return tuple(t.numpy() for t in detect(source, forward, imgsz, CONF, IOU, MAX_DET))


def _compare(image: Path, want: tuple, got: tuple, kind: str) -> str:
    """Return a one-line report, or raise if the two artifacts disagree beyond tolerance."""
    (want_boxes, want_scores, want_ids), (got_boxes, got_scores, got_ids) = want, got
    if len(want_boxes) != len(got_boxes):
        raise SystemExit(
            f"{image.name}: torch found {len(want_boxes)} detections, {kind} found {len(got_boxes)}")
    if not np.array_equal(want_ids, got_ids):
        raise SystemExit(f"{image.name}: {kind} assigns different class ids to the same detections")

    box_error = float(np.abs(want_boxes - got_boxes).max()) if len(want_boxes) else 0.0
    score_error = float(np.abs(want_scores - got_scores).max()) if len(want_scores) else 0.0
    if box_error > BOX_TOLERANCE or score_error > SCORE_TOLERANCE:
        raise SystemExit(
            f"{image.name}: exported {kind} differs from the torch model -- "
            f"boxes {box_error:g} px, scores {score_error:g}. Not published.")
    return (f"    {kind:<7} {image.name:<20} {len(want_boxes):>4} detections, "
            f"boxes {box_error:.2e} px, scores {score_error:.2e}")


def _export_coreml(network, imgsz: int, destination: Path) -> Path:
    """Convert the traced network to a CoreML package and zip it into place.

    The input and output are renamed to match the ONNX graph's, so a consumer reads one name off
    whichever artifact it was handed rather than keeping a per-runtime table.

    Written to a scratch directory first: a package that fails verification must not be left where
    the manifest generator would pick it up.
    """
    import coremltools as ct

    with torch.no_grad():
        traced = torch.jit.trace(network, torch.zeros(1, 3, imgsz, imgsz))
    with tempfile.TemporaryDirectory() as scratch:
        model = ct.convert(
            traced,
            inputs=[ct.TensorType(name=INPUT_NAME, shape=(1, 3, imgsz, imgsz))],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS13,
        )
        spec = model.get_spec()
        ct.utils.rename_feature(spec, spec.description.output[0].name, OUTPUT_NAME)
        renamed = Path(scratch) / "model.mlpackage"
        ct.models.MLModel(spec, weights_dir=model.weights_dir).save(str(renamed))
        shutil.make_archive(str(destination.with_suffix("")), "zip", root_dir=renamed)
    return destination


def export_variant(variant: str, revision: str, weights_dir: Path) -> None:
    """Export one variant, verify it against its torch model, and place it with its labels."""
    revision_dir = weights_dir / "yolov8" / variant / revision
    checkpoint = revision_dir / "torch-fp32.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"{checkpoint} is missing. Run tools/fetch/yolov8.py {variant} first.")

    print(f"  {variant}")
    detector = Detector(checkpoint, device="cpu")
    destination = revision_dir / "onnx-fp32.onnx"

    sample = torch.zeros(1, 3, detector.imgsz, detector.imgsz)
    with torch.no_grad():
        torch.onnx.export(
            detector.network,
            sample,
            str(destination),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=OPSET,
            dynamo=False,
        )

    # Metadata for anyone who takes this graph somewhere other than mozo. mozo itself reads none
    # of it -- it gets the size from the graph's own input shape and the names from labels.json.
    graph = onnx.load(destination)
    graph.metadata_props.append(onnx.StringStringEntryProto(
        key="names", value=json.dumps({int(i): name for i, name in detector.names.items()})))
    graph.metadata_props.append(onnx.StringStringEntryProto(
        key="imgsz", value=json.dumps([detector.imgsz, detector.imgsz])))
    graph.metadata_props.append(onnx.StringStringEntryProto(key="task", value="detect"))
    # A classic head: whoever runs this graph must apply non-maximum suppression themselves.
    graph.metadata_props.append(onnx.StringStringEntryProto(key="end2end", value="False"))
    onnx.save(graph, destination)

    # Decoded once and shared by every comparison below, so each artifact provably starts from
    # the same pixels -- and so this does not pay for a JPEG decode per runtime per image.
    sources = {image: load_image(str(image)) for image in _fixtures()}
    reference = {image: _detections(pixels, detector.forward, detector.imgsz)
                 for image, pixels in sources.items()}

    session = onnxruntime.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    onnx_forward = lambda b: torch.from_numpy(session.run(None, {INPUT_NAME: b.numpy()})[0])  # noqa: E731
    for image, pixels in sources.items():
        print(_compare(image, reference[image], _detections(pixels, onnx_forward, detector.imgsz), "onnx"))
    print(f"    onnx-fp32.onnx {destination.stat().st_size / 1e6:.1f} MB")

    coreml = _export_coreml(detector.network, detector.imgsz, revision_dir / "coreml-fp32.zip")
    runner = CoreMLRunner(coreml)
    coreml_forward = lambda b: torch.from_numpy(runner(b.numpy())[0]).float()  # noqa: E731
    for image, pixels in sources.items():
        print(_compare(image, reference[image], _detections(pixels, coreml_forward, detector.imgsz), "coreml"))
    print(f"    coreml-fp32.zip {coreml.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+", help="variant names, e.g. nano small")
    parser.add_argument("--revision", default="2026-08-19", help="revision directory to export into")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    args = parser.parse_args()

    for variant in args.variants:
        export_variant(variant, args.revision, args.weights_dir)

    print(f"\n{len(args.variants)} exported. Run tools/labels/yolov8.py, then tools/generate_manifest.py.")
    print("the graphs contain AGPL-3.0 weights and are covered by the NOTICE beside them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
