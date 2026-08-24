#!/usr/bin/env python3
"""Check the vendored YOLO11 against `ultralytics` itself, stage by stage.

    python tools/verify/yolov11_reference.py                     # nano, the fixture photograph
    python tools/verify/yolov11_reference.py --variant seg-nano
    python tools/verify/yolov11_reference.py photo.jpg other.jpg
    python tools/verify/yolov11_reference.py --falsify all       # prove the gate can fail
    python tools/verify/yolov11_reference.py --write             # record the table

This is the third path. ``tools/verify/yolov11.py`` compares the vendored package against mozo,
which is the everyday gate and needs nothing installed; those two share the letterboxing, the
suppression and the coordinate mapping, so a change to either moves both sides together and
neither can see it. This script compares the vendored package against the implementation the
checkpoints come from, which is the only comparison that can.

It needs `ultralytics` importable, so it does not run in CI and it is not part of the test suite.
Run it when the vendor changes, when the published revision changes, or to reproduce the parity
table in ``mozo/vendors/yolov11_deploy/PROVENANCE.md`` -- which is what it prints.

**The comparison is not exact, and that is structural.** Every other mozo vendor is an extraction:
the same operations in the same order, so ``torch.equal`` is the only acceptable answer. This
package is not an extraction. It reads the checkpoint's own record of itself and rebuilds the
network from it, so the two implementations agree mathematically and not bitwise, and the numbers
below are maximum absolute differences under stated tolerances.

**Both sides are handed the same batch** for the layer comparison, so a preprocessing difference
cannot smear into every row. Preprocessing is compared on its own, each side built the way its own
library builds it -- which matters more here than it looks: upstream's predictor letterboxes to a
stride multiple rather than a square, so comparing through ``YOLO.predict()`` compares two
different pictures and reads as a parity failure.

Three pins make the comparison defined, and none is optional:

- **`ultralytics` 8.4.0**, matching the ``v8.4.0`` assets release ``tools/fetch/_ultralytics.py``
  takes the checkpoints from. These checkpoints record ``8.2.100`` as the version that *wrote*
  them, which is not the same question: what is needed is an implementation that reproduces their
  numbers, and 8.4.0 both loads them across the head refactor in between and is the release the
  bytes are published in.
- **CPU, fused, eval.** Fusion folds batch norm into the convolution before it, which is
  arithmetic reordering; comparing a fused side against an unfused one measures the fusion.
- **The mask post-processing of 8.4.0.** Nothing about mask assembly is stored in a checkpoint,
  and upstream has changed it: 8.3.63 crops with a single vectorised comparison and unpads with
  ``int(pad)``, where 8.4.0 has two crop branches and rounds with an asymmetric nudge. The vendor
  reproduces 8.4.0, so this compares against 8.4.0's, and a run against any other version is
  measuring a convention rather than an extraction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# Local weights unless the caller says otherwise: this checks the tree you just built, and
# reaching for the published bucket would verify the wrong bytes. Set before mozo.weights is
# imported, exactly as tools/verify/_detection.py does.
os.environ.setdefault("MOZO_BASE_URL", f"file://{ROOT / 'weights'}")

from common import FIXTURES  # noqa: E402
from fetch._ultralytics import REFERENCE  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.yolov11_deploy.build import load_network  # noqa: E402
from mozo.vendors.yolov11_deploy.image import letterbox  # noqa: E402
from mozo.vendors.yolov11_deploy.model import detect  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: The photograph the recorded table is measured on.
#:
#: From ``tools/common``, which holds the same directory for every export gate, rather than from
#: ``tests/conftest.py`` -- that would pull pytest into a script whose whole point is to run
#: wherever `ultralytics` happens to be installed, and the environment that has `ultralytics` is
#: usually not the one that has the test stack.
FIXTURE = FIXTURES / "example.jpg"

#: Where the table is recorded, for a reader who cannot run this.
TABLE = Path(__file__).with_name("yolov11_reference.json")

#: The square side both sides run at, which is what the checkpoints were trained at.
IMGSZ = 640

#: Threshold for the head rows: low enough to cover marginal anchors, where two implementations
#: diverge first, rather than only the confident ones everything agrees on.
CONF = 0.001

#: Threshold for the detection rows, and it is deliberately not :data:`CONF`.
#:
#: Everything up to the head is two implementations of the same arithmetic, and is compared over
#: every anchor above ``CONF``. Suppression is not: it is a decision rule, and the two sides
#: implement it differently on purpose. Upstream separates classes by shifting every box by
#: ``7680 * class`` and running one pass; mozo shifts by the boxes' own span, because a fixed
#: 7680 is only wide enough when coordinates are positive and a detection running off the
#: letterbox edge produces negative ones.
#:
#: That difference is measurable and was measured. On ``seg-nano`` a class-56 pair has an overlap
#: of **0.700377**, which mozo suppresses; shifted into upstream's band at 430,080 the float32
#: spacing is 0.031 px and the same overlap computes as **0.699832**, which upstream keeps. One
#: detection, at a score of 0.005, decided by arithmetic precision rather than by either rule.
#:
#: So the detection rows are gated where a caller actually reads them, and the low-threshold
#: count is measured and reported rather than gated -- which keeps the disagreement visible
#: instead of hiding it behind a raised bar. A tolerance would not do: a count is a decision, and
#: a tolerance on a decision is meaningless.
SERVING = 0.25

#: Overlap above which two boxes of one class suppress each other. The vendor's default, which is
#: what ``mozo.adapters._yolo`` serves with.
IOU = 0.7

#: Most detections either side will return.
MAX_DET = 300

#: The one reported-only row, named because two places have to agree about skipping it.
MARGINAL = "detections.count.marginal"

#: What each stage is allowed to move by. Boxes are in pixels; scores and coefficients are
#: dimensionless. ``0`` means the two sides must agree exactly -- counts and class ids are
#: decisions, not measurements, and a tolerance on a decision is meaningless. ``None`` means the
#: number is reported and nothing is held to it.
#:
#: ``head.boxes`` covers the anchors that clear :data:`CONF` and ``head.boxes.all`` covers all
#: 8,400, on the sibling family's reasoning: measured over the whole grid the worst disagreement
#: sits on an anchor whose score is zero, which no caller can receive and neither implementation
#: reads. Gating there holds the family to the float noise of its own dead anchors; gating quietly
#: on the survivors alone would hide that the grid moves at all. So both are measured, one is
#: gated.
#:
#: ``masks.pixels`` is the fraction of mask pixels whose yes-or-no differs. A mask can only flip a
#: pixel whose logit is within the two sides' disagreement of zero, so a handful of boundary pixels
#: is what an independent implementation costs; a mask that is actually wrong moves whole regions
#: and cannot stay under this.
#:
#: ``detections.count.marginal`` is the same count taken at :data:`CONF` instead of
#: :data:`SERVING`. It is reported and not gated, for the reason :data:`SERVING` gives.
TOLERANCE = {
    "preprocess": 1e-6,
    "layer": 2e-3,
    "head.boxes": 1e-2,
    "head.boxes.all": None,
    "head.scores": 1e-3,
    "head.coefficients": 1e-3,
    "protos": 1e-3,
    "detections.boxes": 1e-2,
    "detections.scores": 1e-3,
    "detections.classes": 0,
    "detections.count": 0,
    MARGINAL: None,
    "masks.pixels": 1e-5,
}


def beyond(limit: float | None, value: float | None) -> bool:
    """Whether one measurement fails its stage's gate.

    A shape mismatch always fails, whatever the row is held to: two stages that are not even the
    same size have not been compared at all.
    """
    if value is None:
        return True
    return limit is not None and value > limit


def tolerance_for(stage: str) -> float | None:
    """The tolerance a stage is held to; every ``layer.NN`` shares one."""
    return TOLERANCE["layer" if stage.startswith("layer.") else stage]


def reference_model(checkpoint: Path, device: str = "cpu"):
    """Load *checkpoint* through `ultralytics`, fused, ready to run on *device*.

    The suffix check refuses anything but ``.pt``, and mozo publishes ``torch-fp32.pth``. The file
    is the same bytes either way, so it is linked rather than copied -- beside this tool rather
    than beside the weights, so a bootstrap run never writes into a directory of published
    artifacts. The link's target is resolved first: a symlink's relative target is read against the
    *link's* directory, not the caller's.

    The task is inferred from the checkpoint rather than passed. Both kinds go through here, and a
    task named at the call site is a fact stated twice that can disagree with the file.
    """
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != REFERENCE:
        raise SystemExit(
            f"this compares against ultralytics {REFERENCE} and {ultralytics.__version__} is "
            f"installed. See the module docstring for why the version is not a floor."
        )

    linked = TABLE.parent / f".{checkpoint.parent.parent.name}.pt"
    linked.unlink(missing_ok=True)
    linked.symlink_to(checkpoint.resolve())
    try:
        model = YOLO(str(linked)).model
    finally:
        linked.unlink(missing_ok=True)

    head = model.model[-1]
    if getattr(head, "end2end", False):
        raise SystemExit("the reference loaded this head as end-to-end; this family suppresses.")
    return model.eval().fuse().eval().to(device)


def observe(model, image: np.ndarray, batch: torch.Tensor, kind: str,
            marginal: bool = True) -> dict[str, torch.Tensor]:
    """Run one batch and return every stage it produces, keyed the same on both sides.

    *kind* selects which of the two implementations *model* is. The two run different code and
    reach the same quantities, which is the whole point; what they must not do is disagree about
    what a stage is called, so the naming lives here rather than in either branch.

    The head is hooked along with the layers, so its convolutions are not run a second time to
    recover the rows.
    """
    layers = model.model
    seen: dict[str, torch.Tensor] = {}
    handles = [
        layer.register_forward_hook(
            lambda _m, _i, out, index=index: seen.__setitem__(f"layer.{index:02d}", out)
        )
        for index, layer in enumerate(layers)
    ]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()

    raw = seen.pop(f"layer.{len(layers) - 1:02d}")
    # Both heads answer with the same two quantities in different wrappers: the vendor returns the
    # rows alone, or the rows paired with the prototypes; the reference wraps that again in the
    # per-level convolutions its loss would have consumed.
    if kind == "reference":
        raw = raw[0]
    rows, protos = raw if isinstance(raw, tuple) else (raw, None)

    # Both sides carry the checkpoint's own class names, so the split point of the head rows is
    # read rather than assumed on either.
    classes = len(model.names)
    seen["head.boxes.all"] = rows[:, :4]
    seen["head.scores"] = rows[:, 4:4 + classes]
    if protos is not None:
        seen["head.coefficients"] = rows[:, 4 + classes:]
        seen["protos"] = protos

    # The head rows a caller can actually reach. Selected by *our* scores on both sides, so the two
    # are compared on the same anchors rather than each on the ones it happens to rank highest -- a
    # comparison over two different populations measures the selection, not the boxes.
    live = seen["head.scores"][0].amax(0) > CONF
    seen["head.boxes"] = seen["head.boxes.all"][0][:, live]

    # Bound here rather than passed six arguments at each call site: the threshold is the only
    # thing that varies between the two runs below, and each side then declares only what it reads.
    post = ((lambda conf: _vendor_detections(image, rows, protos, conf)) if kind == "vendor"
            else (lambda conf: _reference_detections(image, batch, rows, protos, classes, conf)))

    boxes, scores, class_ids, masks = post(SERVING)
    seen["detections.boxes"] = boxes
    seen["detections.scores"] = scores
    seen["detections.classes"] = class_ids.to(torch.int64)
    seen["detections.count"] = torch.tensor(len(scores))
    if masks is not None:
        seen["masks.pixels"] = masks
    if marginal:
        # The same list at the marginal threshold, where the two suppression rules decide ties
        # differently. Only its length is kept: the rows themselves are what ``SERVING`` explains
        # cannot be paired. Skipped while falsifying, where nothing reads it -- the row carries no
        # tolerance, so it can never appear in ``failures``, and computing it there costs a second
        # full assembly of up to ``MAX_DET`` source-resolution masks per perturbation.
        seen[MARGINAL] = torch.tensor(len(post(CONF)[1]))
    return seen


def _vendor_detections(image: np.ndarray, rows: torch.Tensor, protos: torch.Tensor | None,
                       conf: float):
    """The vendor's own post-processing, run on the head output already in hand.

    Through the shipped :func:`~mozo.vendors.yolov11_deploy.model.detect` rather than a copy of
    its steps, with the forward pass handed back the tensor this function was given. That seam
    exists so a graph runtime can be plugged into it, and using it here is what keeps the gate
    measuring what actually ships instead of a second implementation that agrees with itself.
    """
    answer = rows if protos is None else (rows, protos)
    return detect(image, lambda _batch: answer, IMGSZ, conf, IOU, MAX_DET)


def _reference_detections(image: np.ndarray, batch: torch.Tensor, rows: torch.Tensor,
                          protos: torch.Tensor | None, classes: int, conf: float):
    """Upstream's own post-processing, reproduced from ``SegmentationPredictor.postprocess``.

    The ``retina_masks`` branch, which is the one that returns boxes and masks in the same
    coordinate system; see ``mozo/vendors/yolov11_deploy/mask.py`` for why mozo follows it.

    Re-sorted by score before comparing. Both sides already emerge in score order, so this moves
    nothing -- it is here because a pairing that depends on two independent sorts agreeing about
    ties is a comparison that can silently become row-against-row of different objects.

    *rows* is cloned first. Upstream's suppression writes its corner boxes back over the tensor it
    was handed -- ``prediction[..., :4] = xywh2xyxy(prediction[..., :4])`` -- so passing the head
    output straight in converts the head stages this gate is still holding views of, from
    centre-form to corner-form, after they were recorded. It read as a 634 px disagreement on the
    head boxes of a model whose detections agree to 6e-04 px.
    """
    from ultralytics.utils import ops
    from ultralytics.utils.nms import non_max_suppression

    kept = non_max_suppression(rows.clone(), conf, IOU, nc=classes, max_det=MAX_DET)[0]
    boxes = ops.scale_boxes(batch.shape[2:], kept[:, :4].clone(), image.shape[:2])
    masks = None
    if protos is not None:
        masks = ops.process_mask_native(protos[0], kept[:, 6:], boxes, image.shape[:2]).bool()
        # Upstream drops any detection whose mask came out empty, in ``Results``; the vendor
        # reproduces it, so the two are compared after the same rows have gone.
        alive = masks.amax((-2, -1)) > 0
        boxes, kept, masks = boxes[alive], kept[alive], masks[alive]
    order = kept[:, 4].argsort(descending=True, stable=True)
    return (boxes[order], kept[order, 4], kept[order, 5],
            None if masks is None else masks[order])


def preprocessed(image: np.ndarray, imgsz: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Letterbox one image both ways: the vendor's, and the reference's own.

    The reference decodes to BGR and flips while building the batch; mozo decodes once, to RGB, and
    the vendor takes the array. So each side is handed what its own library hands it, and what is
    compared is the tensor that reaches the network.
    """
    from ultralytics.data.augment import LetterBox

    ours, _, _, _ = letterbox(image, imgsz)

    padded = LetterBox((imgsz, imgsz), auto=False, scaleup=True)(image=image[:, :, ::-1])
    chw = np.ascontiguousarray(padded.transpose(2, 0, 1)[::-1], dtype=np.float32)
    return ours, torch.from_numpy(chw).div_(255.0)[None]


def difference(stage: str, ours: torch.Tensor, theirs: torch.Tensor) -> float | None:
    """How far apart two stages are, or ``None`` if they are not even the same shape.

    Boolean masks are measured as the fraction of pixels that disagree, because a maximum absolute
    difference over yes-or-no answers is 1 or 0 and says only *whether* something moved.
    """
    if ours.shape != theirs.shape:
        return None
    if stage == "masks.pixels":
        return float((ours != theirs).sum()) / ours.numel()
    return float((ours.float() - theirs.float()).abs().max())


def expectations(reference, images: list[Path]) -> dict[str, dict]:
    """What the reference produces for each image: the batch it was given, and its stages.

    Computed once and reused across every falsification, because no perturbation touches it --
    ``falsify`` only ever mutates the vendor, and the batch both sides are handed depends on the
    image alone.
    """
    prepared = {}
    for path in images:
        image = load_image(str(path))
        ours, theirs = preprocessed(image, IMGSZ)
        prepared[path.name] = {
            "image": image,
            "batch": ours,
            "preprocess": difference("preprocess", ours, theirs),
            "stages": observe(reference, image, ours, "reference"),
        }
    return prepared


def compare(vendor, prepared: dict[str, dict], perturb: str | None = None) -> dict[str, dict]:
    """Measure *vendor* against the prepared reference stages, and return every disagreement.

    Takes both sides already built rather than building them, so a caller running several
    perturbations pays for the reference once. *vendor* is mutated by ``falsify`` and must be
    freshly built per perturbation; the reference never is.
    """
    if perturb:
        falsify(vendor, perturb)

    # The marginal count carries no tolerance, so it can never appear in ``failures`` and a
    # falsification run has no use for it -- while computing it costs a second suppression and a
    # second full assembly of up to ``MAX_DET`` source-resolution masks, per perturbation. It is
    # dropped from *both* sides together, so the shape check below does not read its absence on
    # one of them as a disagreement.
    marginal = perturb is None

    measured: dict[str, dict[str, float | None]] = {}
    for name, ready in prepared.items():
        measured.setdefault("preprocess", {})[name] = ready["preprocess"]
        # The same batch to both, so the layer rows measure the layers and nothing else.
        seen = observe(vendor, ready["image"], ready["batch"], "vendor", marginal=marginal)
        expected = ready["stages"] if marginal else {
            stage: value for stage, value in ready["stages"].items() if stage != MARGINAL}
        # A stage one side produced and the other did not is not a small disagreement, and a
        # comparison that quietly skipped it would read as agreement. Recorded before the loop
        # below, which consumes ``seen`` as it goes.
        for stage in sorted(set(seen) ^ set(expected)):
            measured.setdefault(stage, {})[name] = None
        for stage in sorted(set(seen) & set(expected)):
            measured.setdefault(stage, {})[name] = difference(stage, seen.pop(stage), expected[stage])
    return measured


def published(variant: str) -> Path:
    """The published checkpoint for *variant*, as the manifest names it.

    Through :func:`mozo.weights.resolve` rather than by sorting revision directory names: a
    directory listing can select a revision the manifest does not publish, so the reference table
    would be measured against bytes the everyday gate never checks, and nothing would say so.
    """
    try:
        return Path(resolve("yolov11", variant, "torch-fp32"))
    except Exception as failure:
        raise SystemExit(f"no weights for yolov11/{variant}: {failure} "
                         "-- run tools/fetch/yolov11.py") from failure


#: Deliberate breakages, and the stage each should first be caught at. A gate that has never
#: failed has not been shown to work, and a gate that fails everywhere at once has not been shown
#: to localise -- which is most of what a per-stage table is for.
FALSIFICATIONS = {
    "batchnorm-eps": "layers, then everything after them -- never the preprocessing",
    "anchor-offset": "the boxes, and the masks they crop -- never a score or a prototype",
    "proto-upsample": "the prototypes and the masks, and nothing else at all",
    "coefficient-roll": "the coefficients and the masks -- every box and score left alone",
    "regrid": "nothing -- the control, a change that provably cannot move a number",
}


def falsify(vendor, which: str) -> None:
    """Perturb one constant, so the table can be watched to fail where that constant reaches."""
    head = vendor.model[-1]
    if which == "batchnorm-eps":
        # Fusion has already folded the epsilon in, so this offsets every fused bias -- which is
        # what an unfused model built with a different epsilon would have done from the start.
        for module in vendor.modules():
            if isinstance(module, torch.nn.Conv2d) and module.bias is not None:
                module.bias.data.add_(1e-4)
    elif which == "anchor-offset":
        head.spec["strides"] = tuple(stride * 1.001 for stride in head.spec["strides"])
    elif which == "proto-upsample":
        # The substitution upstream's own comment names beside the layer: an interpolation where a
        # learned transposed convolution belongs.
        head.proto.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")
    elif which == "coefficient-roll":
        # Each anchor's coefficients handed to the next anchor along, which is what gathering the
        # coefficients with a second selection would do: every mask on a neighbouring object, and
        # nothing raised anywhere.
        for branch in head.cv4:
            branch.register_forward_hook(lambda _m, _i, out: out.roll(1, dims=-1))
    elif which == "regrid":
        # The control. Clearing the cache forces every anchor centre to be built again from
        # scratch, which exercises the code and must produce identical numbers.
        from mozo.vendors.yolov11_deploy.flow import anchor_grid
        anchor_grid.cache_clear()


def failures(measured: dict[str, dict]) -> list[tuple[str, str]]:
    """Every ``(stage, image)`` that disagreed, by the one rule in :func:`beyond`."""
    return [(stage, name) for stage in sorted(measured)
            for name, value in measured[stage].items()
            if beyond(tolerance_for(stage), value)]


def report(measured: dict[str, dict], images: list[Path]) -> int:
    """Print the parity table and return an exit code."""
    names = [path.name for path in images]
    width = max(len(stage) for stage in measured)
    print(f"\n{'Check':<{width}}  {'Tolerance':>10}  " + "  ".join(f"{n:>12}" for n in names))
    print("-" * (width + 12 + 14 * len(names)))

    for stage in sorted(measured):
        limit = tolerance_for(stage)
        cells = []
        for name in names:
            value = measured[stage].get(name)
            if value is None:
                cells.append(f"{'SHAPE':>12}")
            elif limit == 0:
                cells.append(f"{'equal' if value == 0 else 'DIFFERS':>12}")
            else:
                cells.append(f"{value:>12.2e}")
        shown = "--" if limit is None else "exact" if limit == 0 else f"{limit:.0e}"
        print(f"{stage:<{width}}  {shown:>10}  " + "  ".join(cells))

    failed = failures(measured)
    if failed:
        print(f"\n{len(failed)} comparison(s) outside tolerance:")
        for stage, name in failed:
            print(f"  {stage} on {name}")
        return 1
    print(f"\nall {sum(len(r) for r in measured.values())} comparisons within tolerance")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="*", type=Path, help="images to compare on")
    parser.add_argument("--variant", default="nano")
    parser.add_argument("--falsify", metavar="NAME", choices=[*FALSIFICATIONS, "all"],
                        help=f"perturb a constant on the vendor: {sorted(FALSIFICATIONS)}, or 'all'")
    parser.add_argument("--write", action="store_true",
                        help="record the table, for a reader who cannot run this")
    arguments = parser.parse_args()

    images = arguments.images or [FIXTURE]
    missing = [p for p in images if not p.exists()]
    if missing:
        raise SystemExit(f"no such image: {missing}")

    checkpoint = published(arguments.variant)
    reference = reference_model(checkpoint)
    prepared = expectations(reference, images)

    def vendor():
        """A clean vendor, because ``falsify`` mutates the one it is given."""
        return load_network(str(checkpoint), fuse=True).eval()

    if arguments.falsify == "all":
        print("Falsifying the gate. Each row should fail where its constant reaches, and only "
              "there.\n")
        for which, expectation in FALSIFICATIONS.items():
            measured = compare(vendor(), prepared, perturb=which)
            moved = sorted({stage for stage, _ in failures(measured)})
            print(f"{which}\n  expected: {expectation}\n  moved:    {moved or 'nothing'}\n")
        return 0

    import ultralytics

    print(f"variant     yolov11/{arguments.variant}")
    print(f"reference   ultralytics {ultralytics.__version__}")
    print(f"torch       {torch.__version__}, CPU, fused")
    print(f"images      {', '.join(p.name for p in images)}")

    measured = compare(vendor(), prepared, perturb=arguments.falsify)
    code = report(measured, images)

    if arguments.write and code == 0:
        # Merged rather than overwritten, so recording one variant does not silently drop the
        # others measured on earlier runs -- a table missing most of the family still looks
        # complete.
        recorded = json.loads(TABLE.read_text()) if TABLE.exists() else {}
        recorded.update({
            "reference": f"ultralytics {ultralytics.__version__}",
            "torch": torch.__version__,
            "device": "cpu",
            "fused": True,
            "conf": CONF,
            "iou": IOU,
            "imgsz": IMGSZ,
        })
        recorded.setdefault("variants", {})[arguments.variant] = {
            "images": [p.name for p in images],
            "stages": {stage: {"tolerance": tolerance_for(stage), "per_image": per_image}
                       for stage, per_image in measured.items()},
        }
        TABLE.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        print(f"wrote yolov11/{arguments.variant} to {TABLE.relative_to(ROOT)} "
              f"({len(recorded['variants'])} variant(s) recorded)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
