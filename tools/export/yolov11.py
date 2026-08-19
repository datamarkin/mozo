#!/usr/bin/env python3
"""Export YOLO11 variants to ONNX and drop them into the local ``weights/`` tree.

This runs once, on a machine you control, and never ships. Users receive the ``.onnx`` file, not
the code that produced it -- which is the point of exporting at all: the graph carries the
architecture, so the vendored package stays off the serving path.

    python tools/export/yolov11.py nano
    python tools/export/yolov11.py nano small --revision 2026-08-19

Every export is checked against the torch model it came from before it is written. An export that
disagrees is not published: a silent numerical divergence between two artifacts of the same model
is the one failure this whole scheme exists to prevent. The check compares *detections* on real
photographs rather than raw tensors -- what matters is that the two artifacts find the same
objects in the same places, and a tensor tolerance measures kernel arithmetic instead.

One artifact lands in ``weights/yolov11/<variant>/<revision>/``, mapping a letterboxed
``(1, 3, imgsz, imgsz)`` batch to the raw head output ``(1, 4 + classes, anchors)``. That is a
classic head, so whoever runs it still applies non-maximum suppression -- which mozo does with the
vendor's own :func:`~mozo.vendors.yolov11_deploy.detect`, so every runtime shares it.

**No CoreML, unlike YOLOv8.** Converting the full network produces a package that aborts the
process when it runs::

    MPSGraphExecutable.mm:5070: failed assertion 'Error: MLIR pass manager failed'

That is an abort, not an exception: no ``except`` clause anywhere catches it, and a server that
loaded such an artifact would simply die. It was bisected to layer 10, the ``C2PSA`` attention
block -- a graph cut after layer 9 converts and runs, and the block converts and runs correctly on
its own, so it is a compiler pass failing on the assembled graph rather than an unsupported
operation. Rewriting the attention as a 3-D batched matmul (bit-identical in torch, ``max|d|`` 0.0)
did not help, and neither did ``macOS15`` as the deployment target.

Restricting compute units to CPU and the Neural Engine does produce a working, accurate package --
0.0002 px -- but at 23.5 ms against 10.4 ms for torch on MPS, so there is nothing to gain even
where it is safe. Nothing in ``mozo/runtimes.py`` special-cases this: ``auto`` only ever chooses
among what a variant publishes, and this family publishes no CoreML.

**No fp16 either.** Those measurements were taken on the CoreML path, which this family does not
have, so they belong to the sibling and live in ``tools/export/yolov8.py`` rather than being
paraphrased here where they cannot be reproduced. The short version: fp16 finds every object fp32
finds and puts them in slightly the wrong place, which is not a trade a detector should make.

**These weights are AGPL-3.0, and so is anything exported from them.** The graph produced here
contains the weights. It lands in the same revision directory as the LICENSE and NOTICE that
``tools/fetch/yolov11.py`` placed, which is what keeps those terms travelling with it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mozo.image import load_image  # noqa: E402
from mozo.vendors.yolov11_deploy import Detector, detect  # noqa: E402

#: The photographs the comparison runs on.
FIXTURES = ROOT / "tests" / "fixtures" / "images"

#: Opset the graphs are written at. A constant rather than a flag: which opset mozo publishes is a
#: property of the artifact, not something a caller should be able to vary per run.
OPSET = 17

#: Thresholds the comparison runs at. Well below any sensible serving threshold, because a
#: divergence that only shows up on marginal detections is still a divergence and a strict
#: threshold would hide it by discarding exactly the boxes where the two artifacts disagree --
#: but not arbitrarily low, which is the mistake the sibling family's exporter started from.
#:
#: Below about 0.005 the comparison stops measuring the model. On the fixture photograph, 540 of
#: 8400 anchors have their winning class decided by a top-two margin of 1.6e-07 at a score of
#: essentially zero, and the two artifacts agree on scores only to 4.7e-06 -- so which class
#: "wins" there is decided by float noise, differs run to run, and has nothing to say about
#: whether the graph is faithful. At 0.001 that produced a class-id mismatch on an export whose
#: raw head agrees to 2.2e-03 px. At 0.01 the counts and every class id match exactly, with 59
#: detections still in play, most of them marginal.
CONF = 0.01
IOU = 0.7
MAX_DET = 300

#: Names the graph declares, matching the sibling family's so nothing downstream needs a
#: per-family table to read one artifact's input and output.
INPUT_NAME = "images"
OUTPUT_NAME = "predictions"

#: What the export must reproduce. Counts and class ids are exact -- an artifact that finds a
#: different number of objects, or calls one of them something else, is not the same model.
BOX_TOLERANCE = 1e-2      # pixels, in the source image's coordinates
SCORE_TOLERANCE = 1e-3


def _fixtures() -> list[Path]:
    """Photographs to compare on. Real images, because synthetic noise proves nothing here."""
    images = sorted(p for p in FIXTURES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"no fixture images in {FIXTURES}. Add photographs to verify against.")
    return images


def _detections(source: np.ndarray, forward, imgsz: int) -> tuple[np.ndarray, ...]:
    """Detect in one already-decoded image, exactly as mozo would at serving time.

    Post-processing is the vendor's :func:`~mozo.vendors.yolov11_deploy.detect`, so what is
    compared here is the graph against the module and nothing else.
    """
    with torch.no_grad():
        return tuple(t.numpy() for t in detect(source, forward, imgsz, CONF, IOU, MAX_DET))


def _compare(image: Path, want: tuple, got: tuple, kind: str) -> str:
    """Return a one-line report, or raise if the two artifacts disagree beyond tolerance.

    Detections are paired by position, which is sound only because both sides are full precision
    and their suppression order is therefore identical. It would not be sound for a reduced
    precision artifact: perturbing scores reorders near-tied boxes, and pairing by position then
    subtracts unrelated boxes and reports an error in the hundreds of pixels that is an artefact
    of the pairing rather than of the model. Any future fp16 artifact needs IoU matching here.
    """
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


def export_variant(variant: str, revision: str, weights_dir: Path, sources: dict) -> None:
    """Export one variant and verify it against its torch model.

    *sources* is the decoded fixture set, passed in rather than read here so that exporting five
    variants decodes each photograph once instead of five times.
    """
    revision_dir = weights_dir / "yolov11" / variant / revision
    checkpoint = revision_dir / "torch-fp32.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"{checkpoint} is missing. Run tools/fetch/yolov11.py {variant} first.")

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

    reference = {image: _detections(pixels, detector.forward, detector.imgsz)
                 for image, pixels in sources.items()}

    session = onnxruntime.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    onnx_forward = lambda b: torch.from_numpy(session.run(None, {INPUT_NAME: b.numpy()})[0])  # noqa: E731
    for image, pixels in sources.items():
        print(_compare(image, reference[image], _detections(pixels, onnx_forward, detector.imgsz), "onnx"))
    print(f"    onnx-fp32.onnx {destination.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variants", nargs="+", help="variant names, e.g. nano small")
    parser.add_argument("--revision", default="2026-08-19", help="revision directory to export into")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "weights")
    args = parser.parse_args()

    # Decoded once for the whole run, so every variant and every artifact provably starts from
    # the same pixels and no photograph is decoded twice.
    sources = {image: load_image(str(image)) for image in _fixtures()}
    for variant in args.variants:
        export_variant(variant, args.revision, args.weights_dir, sources)

    print(f"\n{len(args.variants)} exported. Run tools/labels/yolov11.py, then tools/generate_manifest.py.")
    print("the graphs contain AGPL-3.0 weights and are covered by the NOTICE beside them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
