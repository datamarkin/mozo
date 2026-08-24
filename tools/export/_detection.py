#!/usr/bin/env python3
"""Export a detection family's weights to the graph artifacts mozo publishes.

This runs once, on a machine you control, and never ships. Users receive the ``.onnx`` file or
the CoreML package, not the code that produced it -- which is the point of exporting at all: the
graph carries the architecture, so the vendored package stays off the serving path.

Every export is checked against the torch model it came from before it is written. An export that
disagrees is not published: a silent numerical divergence between two artifacts of the same model
is the one failure this whole scheme exists to prevent. The check compares *detections* on real
photographs rather than raw tensors -- what matters is that the two artifacts find the same
objects in the same places, and a tensor tolerance measures kernel arithmetic instead.

Shared rather than copied per family, for the same reason ``tools/verify/_detection.py`` is: this
is a publication gate, and a stale second copy keeps writing artifacts while checking something
older than what it is guarding. That is not hypothetical here. When the three families each had
their own exporter, a fix that decoded the fixture photographs once per run instead of once per
variant was made in one of them and never reached the other two, so two of the three re-decoded
every photograph five times while carrying a comment saying they did not.

The families differ in two things, both booleans. Whether they publish CoreML -- YOLOv8 and YOLO12
do; YOLO11 and YOLO26 do not, both because of the ``C2PSA`` block that makes Apple's Metal graph
compiler abort the process rather than raise. And whether their head is end-to-end: YOLO26's graph
carries its own decode and top-k and returns a detection list, where the others return a raw head
that still needs suppression. The reasoning behind each family's answer stays in that family's own
module, where someone attempting the conversion would look.

The second one is read from the model rather than passed in, because the vendor already validated
it against the checkpoint and a fact with one source cannot disagree with itself.
"""

from __future__ import annotations

import importlib
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
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402

#: Opset the graphs are written at. A constant rather than a flag: which opset mozo publishes is a
#: property of the artifact, not something a caller should be able to vary per run.
OPSET = 17

#: Thresholds the comparison runs at. Well below any sensible serving threshold, because a
#: divergence that only shows up on marginal detections is still a divergence and a strict
#: threshold would hide it by discarding exactly the boxes where the two artifacts disagree --
#: but not arbitrarily low, which 0.001 was.
#:
#: Below about 0.005 the comparison stops measuring the model. On the fixture photograph 540 of
#: 8400 anchors have their winning class decided by a top-two margin of 1.6e-07 at a score of
#: essentially zero, while two faithful artifacts agree on scores only to 4.7e-06 -- so which
#: class "wins" there is float noise, differs run to run, and says nothing about whether the graph
#: is faithful. At 0.001 that failed a YOLO11 export whose raw head agreed to 2.2e-03 px.
#:
#: The threshold is the only condition set here. Overlap and detection count are the vendor's own
#: defaults, because those are what mozo serves under -- the adapter passes a threshold and
#: nothing else -- and restating them would verify the graph under conditions no user runs. It
#: also lets an NMS-free family, whose ``detect`` has no overlap to set, share this unchanged.
#: ``tests/test_vendor_agreement.py`` pins the vendors to the same defaults.
CONF = 0.01

#: Names every artifact declares, so nothing downstream needs a per-runtime or per-family table.
INPUT_NAME = "images"
OUTPUT_NAME = "predictions"

#: What an export must reproduce. Counts and class ids are exact -- an artifact that finds a
#: different number of objects, or calls one of them something else, is not the same model.
BOX_TOLERANCE = 1e-2      # pixels, in the source image's coordinates
SCORE_TOLERANCE = 1e-3


def _detections(vendor, source: np.ndarray, forward, imgsz: int) -> tuple[np.ndarray, ...]:
    """Detect in one already-decoded image, exactly as mozo would at serving time.

    Post-processing is the vendor's own ``detect``, so what is compared is the graph against the
    module and nothing else.
    """
    with torch.no_grad():
        # No family publishes a graph carrying masks, so a graph and the torch module it came
        # from have nothing to disagree about there and this compares boxes, scores and classes.
        # A checkpoint that *does* produce masks is refused rather than compared on three quarters
        # of its answer -- unpacking the value into a discard would have read exactly the same and
        # guarded nothing.
        boxes, scores, class_ids, masks = vendor.detect(source, forward, imgsz, CONF)
        if masks is not None:
            raise NotImplementedError(
                "this checkpoint has a mask branch, and this gate compares boxes, scores and "
                "class ids only. Comparing a graph against the module on three quarters of the "
                "answer would report agreement it has not checked."
            )
        return boxes.numpy(), scores.numpy(), class_ids.numpy()


def _compare(image: Path, want: tuple, got: tuple, kind: str) -> str:
    """Return a one-line report, or raise if the two artifacts disagree beyond tolerance.

    Detections are paired by position, which is sound only because both sides are full precision
    and their ordering -- suppression for a classic head, an in-graph top-k for an end-to-end one
    -- is therefore identical. ``tools/export/yolov26.py`` records what that costs when it is not:
    across all 300 rows of an end-to-end graph, position pairing reads 0.54 px where content
    pairing reads 0.001, because two executors of the same top-k may break ties differently. It would not be sound for a reduced
    precision artifact: perturbing scores reorders near-tied boxes, and pairing by position then
    subtracts unrelated boxes and reports an error in the hundreds of pixels that is an artefact
    of the pairing rather than of the model. That mistake is recorded in ``tools/export/yolov8.py``,
    which is where fp16 was measured. Any future fp16 artifact needs IoU matching here.
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


def compare_by_content(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Pair two ``[x1, y1, x2, y2, score, class]`` sets one-to-one; report the worst box and score gap.

    Rows are matched by class and geometry, never by position. Two implementations that rank
    equally scoring detections differently have not found different things, and comparing them
    index by index would say they had.

    Not on any path today: above ``CONF`` every family's two artifacts rank identically and
    :func:`_compare` pairs by position, which is cheaper and stricter. This is what that becomes
    when the assumption breaks -- a reduced-precision artifact, or a threshold low enough to admit
    the noise rows an end-to-end graph pads its output with. Measured on YOLO26's full 300-row
    output, position pairing reports 0.54 px where this reports 0.001.

    It lives here rather than in the vendor that first needed it: no shipping code path compares
    detection sets, and a comparison the wheel carries but never runs is a tool in the wrong layer.
    """
    if left.shape != right.shape:
        raise ValueError(f"detection sets differ in size: {left.shape} against {right.shape}")
    order = [np.lexsort((rows[:, 1], rows[:, 0], rows[:, 5])) for rows in (left, right)]
    left, right = left[order[0]], right[order[1]]
    if not np.array_equal(left[:, 5], right[:, 5]):
        raise ValueError("detection sets disagree on which classes were found")
    return float(np.abs(left[:, :4] - right[:, :4]).max()), float(np.abs(left[:, 4] - right[:, 4]).max())


def _export_onnx(detector, destination: Path) -> None:
    """Write the graph, then record what a consumer outside mozo would otherwise have to guess."""
    imgsz = detector.imgsz
    # Whether the graph returns a detection list or a raw head. A family whose network records no
    # ``end2end`` has a classic head; YOLO26 records it, and its builder refuses a checkpoint that
    # does not. Stamping a constant here would tell a consumer of an end-to-end graph to suppress
    # an already-suppressed list, and mozo reads none of this metadata, so nothing would notice.
    end2end = bool(getattr(detector.network, "end2end", False))
    with torch.no_grad():
        torch.onnx.export(
            detector.network,
            torch.zeros(1, 3, imgsz, imgsz),
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
        key="imgsz", value=json.dumps([imgsz, imgsz])))
    graph.metadata_props.append(onnx.StringStringEntryProto(key="task", value="detect"))
    graph.metadata_props.append(onnx.StringStringEntryProto(key="end2end", value=str(end2end)))
    onnx.save(graph, destination)


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


def export_variant(family: str, vendor, variant: str, revision: str,
                   weights_dir: Path, sources: dict, coreml: bool) -> None:
    """Export one variant, verify every artifact against its torch model, and place them."""
    revision_dir = weights_dir / family / variant / revision
    checkpoint = revision_dir / "torch-fp32.pth"
    if not checkpoint.is_file():
        raise SystemExit(f"{checkpoint} is missing. Run tools/fetch/{family}.py {variant} first.")

    print(f"  {variant}")
    detector = vendor.Detector(checkpoint, device="cpu")
    destination = revision_dir / "onnx-fp32.onnx"
    _export_onnx(detector, destination)

    reference = {image: _detections(vendor, pixels, detector.forward, detector.imgsz)
                 for image, pixels in sources.items()}

    session = onnxruntime.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    onnx_forward = lambda b: torch.from_numpy(session.run(None, {INPUT_NAME: b.numpy()})[0])  # noqa: E731
    for image, pixels in sources.items():
        print(_compare(image, reference[image],
                       _detections(vendor, pixels, onnx_forward, detector.imgsz), "onnx"))
    print(f"    onnx-fp32.onnx {destination.stat().st_size / 1e6:.1f} MB")

    if not coreml:
        return

    from mozo.runtimes import CoreMLRunner

    package = _export_coreml(detector.network, detector.imgsz, revision_dir / "coreml-fp32.zip")
    runner = CoreMLRunner(package)
    coreml_forward = lambda b: torch.from_numpy(runner(b.numpy())[0]).float()  # noqa: E731
    for image, pixels in sources.items():
        print(_compare(image, reference[image],
                       _detections(vendor, pixels, coreml_forward, detector.imgsz), "coreml"))
    print(f"    coreml-fp32.zip {package.stat().st_size / 1e6:.1f} MB")


def run(family: str, *, coreml: bool, description: str = "") -> int:
    """Export the variants named on the command line for *family*.

    The vendor is found from the family name, which is the same string everywhere -- the manifest
    key, the weights directory and ``mozo.vendors.<family>_deploy``.
    """
    args = variant_parser(description or f"export {family}",
                          ROOT / "weights", required=True).parse_args()
    vendor = importlib.import_module(f"mozo.vendors.{family}_deploy")

    # Decoded once for the whole run, so every variant and every artifact provably starts from the
    # same pixels and no photograph is decoded twice.
    sources = {image: load_image(str(image)) for image in fixtures()}
    for variant in args.variants:
        export_variant(family, vendor, variant, args.revision, args.weights_dir, sources, coreml)

    print(f"\n{len(args.variants)} exported. Run tools/labels/{family}.py, then tools/generate_manifest.py.")
    print("the graphs contain AGPL-3.0 weights and are covered by the NOTICE beside them")
    return 0
