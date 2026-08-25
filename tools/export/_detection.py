#!/usr/bin/env python3
"""Export a detection family's weights to the graph artifacts mozo publishes.

This runs once, on a machine you control, and never ships. Users receive the ``.onnx`` file or
the CoreML package, not the code that produced it -- which is the point of exporting at all: the
graph carries the architecture, so the vendored package stays off the serving path.

Every export is checked against the torch model it came from before it is written. An export that
disagrees is not published: a silent numerical divergence between two artifacts of the same model
is the one failure this whole scheme exists to prevent. The check compares *detections* on real
photographs rather than raw tensors -- what matters is that the two artifacts find the same
objects in the same places, and a tensor tolerance measures kernel arithmetic instead. A
segmentation checkpoint is compared on its masks too, by overlap: it answers with four things and
checking three of them would report an agreement it had not looked for.

Shared rather than copied per family, for the same reason ``tools/verify/_detection.py`` is: this
is a publication gate, and a stale second copy keeps writing artifacts while checking something
older than what it is guarding. That is not hypothetical here. When the three families each had
their own exporter, a fix that decoded the fixture photographs once per run instead of once per
variant was made in one of them and never reached the other two, so two of the three re-decoded
every photograph five times while carrying a comment saying they did not.

The families differ in three things, all booleans. Whether they publish CoreML -- YOLOv8 and YOLO12
do; YOLO11 and YOLO26 do not, both because of the ``C2PSA`` block that makes Apple's Metal graph
compiler abort the process rather than raise. Whether their head is end-to-end: YOLO26's graph
carries its own decode and top-k and returns a detection list, where the others return a raw head
that still needs suppression. And whether the checkpoint has a mask branch, which decides whether
the graph has one output or two. The reasoning behind each family's answer stays in that family's
own module, where someone attempting the conversion would look.

Only the first is passed in. The other two are read off the network -- ``end2end`` and ``nm``,
which both vendors record -- because the vendor already validated them against the checkpoint and
a fact with one source cannot disagree with itself. Sampling the mask branch from a detection
result instead would work today and make the graph's shape depend on when it was asked.
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
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402

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
#: Only a segmentation graph carries the second: the prototypes its rows are coefficients of.
INPUT_NAME = "images"
OUTPUT_NAME = "predictions"
PROTOTYPE_NAME = "prototypes"

#: What an export must reproduce. Counts are exact -- an artifact that finds a different number of
#: objects is not the same model.
BOX_TOLERANCE = 1e-2      # pixels, in the source image's coordinates
SCORE_TOLERANCE = 1e-3

#: Worst per-detection mask overlap an export may have against the torch model it came from.
#:
#: Masks are thresholded logits, so a float difference far below any box tolerance still flips
#: whichever pixels sat on the boundary, and only those. Measured at ``CONF`` on the fixture:
#: YOLO26 flips none at either size, YOLO11 flips five pixels in 164.8 million on ``seg-nano`` and
#: one in 59.0 million on ``seg-xlarge``, for a worst single-mask overlap of 0.999896. This leaves
#: room for that and nothing like room for a mask in the wrong place.
MASK_IOU = 0.999


def _detections(vendor, source: np.ndarray, forward, imgsz: int) -> tuple[np.ndarray, ...]:
    """Detect in one already-decoded image, exactly as mozo would at serving time.

    Post-processing is the vendor's own ``detect``, so what is compared is the graph against the
    module and nothing else.

    Returns boxes, scores, class ids and masks, the last being ``None`` from a checkpoint with no
    mask branch. Carried through rather than dropped, because it is a quarter of the answer and
    :func:`_compare` checks it.
    """
    with torch.no_grad():
        boxes, scores, class_ids, masks = vendor.detect(source, forward, imgsz, CONF)
        return (boxes.numpy(), scores.numpy(), class_ids.numpy(),
                None if masks is None else masks.numpy())


def _rows(detections: tuple) -> np.ndarray:
    """The ``[x1, y1, x2, y2, score, class]`` form :func:`pair_by_content` sorts."""
    boxes, scores, class_ids = detections[:3]
    return np.column_stack([boxes, scores, class_ids])


def _reindex(detections: tuple, order: np.ndarray) -> tuple:
    """Put one image's detections into *order*, masks included.

    One place decides what travels with a detection, so re-pairing cannot reorder three of the
    four things a checkpoint answers with and leave the fourth where it was.
    """
    boxes, scores, class_ids, masks = detections
    return (boxes[order], scores[order], class_ids[order],
            None if masks is None else masks[order])


def _mask_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Per-detection overlap between two ``(n, h, w)`` boolean mask stacks.

    Overlap rather than a count of disagreeing pixels, because a count scales with the
    photograph's area: the same tolerance would be strict on a thumbnail and meaningless on a 4K
    frame. Five pixels in 164 million reads as agreement whichever way it is measured; the same
    five along one small mask's edge does not.

    A mask empty on both sides cannot arise -- ``detect`` drops a detection whose mask came out
    empty -- so the union is positive and the division is safe.

    The union is counted rather than built. ``(left | right).sum(...)`` would allocate a second
    full-resolution stack only to reduce it away, and these are source-resolution masks: 67
    detections on the fixture is 165 million booleans per operand. Inclusion-exclusion gets the
    same integer from two reductions over arrays that already exist.
    """
    intersection = (left & right).sum((-2, -1))
    return intersection / (left.sum((-2, -1)) + right.sum((-2, -1)) - intersection)


def _compare(image: Path, want: tuple, got: tuple, kind: str) -> str:
    """Return a one-line report, or raise if the two artifacts disagree beyond tolerance.

    Detections are paired by position first, which is what two full-precision artifacts of the
    same model normally earn: their ordering -- suppression for a classic head, an in-graph top-k
    for an end-to-end one -- is identical, and position pairing is then both cheaper and stricter
    than matching on content.

    Normally, not always. Two executors of the same top-k are free to break a tie differently, and
    at ``CONF`` an end-to-end graph emits rows close enough together for that to happen: YOLO26's
    ``seg-xlarge`` transposes two detections scoring 0.08235180 and 0.08234713, a gap of 4.7e-06,
    which is the agreement floor two faithful artifacts have anyway. Position pairing then
    subtracts unrelated boxes and reports an error that is an artefact of the pairing rather than
    of the model. So a class-id disagreement is not the failure -- it is the signal to re-pair on
    content and ask again, and only a set that still disagrees is a divergence. The report says
    which pairing it used, so a family that has started needing the fallback shows up in the
    export output rather than being discovered later.

    Neither pairing would be sound for a reduced-precision artifact. Perturbing scores reorders
    near-tied boxes wholesale, and :func:`pair_by_content` sorts on coordinates that have
    themselves moved. That mistake is recorded in ``tools/export/yolov8.py``, which is where fp16
    was measured. Any future fp16 artifact needs IoU matching here.
    """
    if len(want[0]) != len(got[0]):
        raise SystemExit(
            f"{image.name}: torch found {len(want[0])} detections, {kind} found {len(got[0])}")

    pairing = ""
    if not np.array_equal(want[2], got[2]):
        try:
            left, right = pair_by_content(_rows(want), _rows(got))
        except ValueError as error:
            raise SystemExit(f"{image.name}: {kind} {error}") from error
        want, got = _reindex(want, left), _reindex(got, right)
        pairing = ", content-paired"

    (want_boxes, want_scores, _, want_masks) = want
    (got_boxes, got_scores, _, got_masks) = got
    box_error = float(np.abs(want_boxes - got_boxes).max()) if len(want_boxes) else 0.0
    score_error = float(np.abs(want_scores - got_scores).max()) if len(want_scores) else 0.0
    if box_error > BOX_TOLERANCE or score_error > SCORE_TOLERANCE:
        raise SystemExit(
            f"{image.name}: exported {kind} differs from the torch model -- "
            f"boxes {box_error:g} px, scores {score_error:g}. Not published.")

    report = (f"    {kind:<7} {image.name:<20} {len(want_boxes):>4} detections, "
              f"boxes {box_error:.2e} px, scores {score_error:.2e}")
    if want_masks is None:
        return report + pairing

    # ``not worst >= MASK_IOU`` rather than ``worst < MASK_IOU``, so a NaN overlap fails rather
    # than passing on a comparison that is False either way.
    overlap = _mask_iou(want_masks, got_masks)
    worst = float(overlap.min()) if len(overlap) else 1.0
    if not worst >= MASK_IOU:
        raise SystemExit(
            f"{image.name}: exported {kind}'s masks differ from the torch model's -- worst "
            f"overlap {worst:g} IoU, against {MASK_IOU} required. Not published.")
    return f"{report}, masks {worst:.6f} IoU{pairing}"


def pair_by_content(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order two ``[x1, y1, x2, y2, score, class]`` sets so that row *i* of each is one detection.

    Rows are matched by class and geometry, never by position. Two implementations that rank
    equally scoring detections differently have not found different things, and comparing them
    index by index would say they had.

    Returns the two orderings rather than the errors themselves, so that whatever else travels
    with a detection -- its mask -- can be reordered alongside it, and so that one place decides
    what "agrees" means. :func:`_compare` reaches for this only after position pairing has found a
    class-id disagreement; it is the fallback, not the norm.

    Sorting on coordinates the two sides agree on only to within their own error is exactly what
    makes it a fallback. Two same-class boxes closer together than that error could sort
    differently on each side, and the pairing would then be wrong without saying so. At full
    precision the gap between distinct detections is orders of magnitude larger than the gap
    between two renderings of one; for a reduced-precision artifact it would not be.

    Raises:
        ValueError: If the two sets differ in size, or do not hold the same classes.
    """
    if left.shape != right.shape:
        raise ValueError(f"detection sets differ in size: {left.shape} against {right.shape}")
    order = [np.lexsort((rows[:, 1], rows[:, 0], rows[:, 5])) for rows in (left, right)]
    if not np.array_equal(left[order[0], 5], right[order[1], 5]):
        raise ValueError("detection sets disagree on which classes were found")
    return order[0], order[1]


def _export_onnx(detector, destination: Path, segments: bool) -> None:
    """Write the graph, then record what a consumer outside mozo would otherwise have to guess."""
    imgsz = detector.imgsz
    # Whether the graph returns a detection list or a raw head. A family whose network records no
    # ``end2end`` has a classic head; YOLO26 records it, and its builder refuses a checkpoint that
    # does not. Stamping a constant here would tell a consumer of an end-to-end graph to suppress
    # an already-suppressed list, and mozo reads none of this metadata, so nothing would notice.
    end2end = bool(getattr(detector.network, "end2end", False))
    # A mask branch answers with a pair -- the rows, and the prototypes those rows carry
    # coefficients of -- so the graph declares a second output. Naming only the first would leave
    # the second called whatever the tracer happened to number it, which is what a consumer
    # outside mozo would then have to guess. The order is the model's own and is what
    # ``mozo/adapters/_yolo.py`` reads them back in.
    outputs = [OUTPUT_NAME, PROTOTYPE_NAME] if segments else [OUTPUT_NAME]
    with torch.no_grad():
        torch.onnx.export(
            detector.network,
            torch.zeros(1, 3, imgsz, imgsz),
            str(destination),
            input_names=[INPUT_NAME],
            output_names=outputs,
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
    graph.metadata_props.append(onnx.StringStringEntryProto(
        key="task", value="segment" if segments else "detect"))
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

    # Mask coefficients per anchor, zero for a head with no mask branch -- which is the same fact
    # as whether the graph declares one output or two. Read off the network exactly as ``end2end``
    # is, and for the same reason: the vendor validated it against the checkpoint, so it cannot
    # disagree with what the model does. Sampling it from a detection result instead would make
    # the graph's shape depend on when the question was asked.
    segments = bool(getattr(detector.network, "nm", 0))

    destination = revision_dir / "onnx-fp32.onnx"
    _export_onnx(detector, destination, segments)

    reference = {image: _detections(vendor, pixels, detector.forward, detector.imgsz)
                 for image, pixels in sources.items()}

    runner = OnnxRunner(destination, device="cpu")

    def onnx_forward(batch: torch.Tensor):
        """A pair from a segmentation graph, a lone tensor from a detection one.

        The vendor's ``detect`` tells the two apart by type rather than by length, so a
        single-output graph must not be handed back wrapped -- that hands ``detect`` rows with
        their batch axis still attached, which is plausible, wrong and silent. Branching on what
        came back rather than on ``segments`` keeps this reading the graph itself, the same way
        ``mozo/adapters/_yolo.py`` does at serving time.
        """
        outputs = tuple(torch.from_numpy(output) for output in runner(batch.numpy()))
        return outputs if len(outputs) > 1 else outputs[0]

    for image, pixels in sources.items():
        print(_compare(image, reference[image],
                       _detections(vendor, pixels, onnx_forward, detector.imgsz), "onnx"))
    print(f"    onnx-fp32.onnx {destination.stat().st_size / 1e6:.1f} MB")

    if not coreml:
        return
    if segments:
        raise SystemExit(
            f"{family}/{variant} has a mask branch, and the CoreML path below takes one output "
            "and would silently drop the prototypes. No family publishes both today, so the "
            "pairing has never been needed. Wire it here before one does."
        )

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
