#!/usr/bin/env python3
"""Check the vendored YOLO26 against `ultralytics` itself, stage by stage.

    python tools/verify/yolov26_reference.py                    # nano, the fixture photograph
    python tools/verify/yolov26_reference.py --variant small
    python tools/verify/yolov26_reference.py photo.jpg other.jpg
    python tools/verify/yolov26_reference.py --falsify all      # prove the gate can fail
    python tools/verify/yolov26_reference.py --write            # record the table

This is the third path. ``tools/verify/yolov26.py`` compares the vendored package against mozo,
which is the everyday gate and needs nothing installed; those two share the letterboxing and the
coordinate mapping, so a change to either moves both sides together and neither can see it. This
script compares the vendored package against the implementation the checkpoints come from, which
is the only comparison that can.

It needs `ultralytics` importable, so it does not run in CI and it is not part of the test suite.
Run it when the vendor changes, when the published revision changes, or to reproduce the parity
table in ``mozo/vendors/yolov26_deploy/PROVENANCE.md`` -- which is what it prints.

**The comparison is not exact, and that is structural.** Every other mozo vendor is an extraction:
the same operations in the same order, so ``torch.equal`` is the only acceptable answer. This
package is not an extraction. It reads the checkpoint's own record of itself and rebuilds the
network from it, so the two implementations agree mathematically and not bitwise, and the numbers
below are maximum absolute differences under stated tolerances.

**Both sides are handed the same batch** for the layer comparison, so a preprocessing difference
cannot smear into all 23 rows. Preprocessing is compared on its own, each side built the way its
own library builds it.

Two pins make the comparison defined, and neither is optional:

- **`ultralytics` 8.4.0**, matching the ``v8.4.0`` assets release ``tools/fetch/_ultralytics.py``
  takes the checkpoints from. Earlier releases are not merely older: 8.3.222 -- the version the
  checkpoints record as their own writer -- ships no YOLO26 configuration at all, and loads this
  head with ``end2end`` false, which runs the one-to-many ``cv2``/``cv3`` branch and NMS instead
  of the NMS-free ``one2one`` path. It produces detections, and they are a different model's.
- **CPU, fused, eval.** Fusion folds batch norm into the convolution before it, which is
  arithmetic reordering; comparing a fused side against an unfused one measures the fusion.
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

from fetch._ultralytics import REFERENCE  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.vendors.yolov26_deploy.image import letterbox  # noqa: E402
from mozo.vendors.yolov26_deploy.network import build_detector  # noqa: E402
from mozo.weights import resolve  # noqa: E402

#: The photograph the recorded table is measured on.
#:
#: Spelled out rather than imported from ``tests/conftest.py``, which is where the sibling benches
#: get it. Importing it pulls pytest into a script whose whole point is to run wherever
#: `ultralytics` happens to be installed -- and the environment that has `ultralytics` is usually
#: not the environment that has the test stack. One duplicated path against a gate that will not
#: start is the right trade; ``test_stated_counts``-style drift does not apply to a constant that
#: fails loudly the moment it is wrong.
FIXTURE = ROOT / "tests" / "fixtures" / "images" / "example.jpg"

#: Where the table is recorded, for a reader who cannot run this.
TABLE = Path(__file__).with_name("yolov26_reference.json")

#: Serving threshold for the detection rows. Low enough to cover marginal detections, where two
#: implementations diverge first, rather than only the confident ones everything agrees on.
CONF = 0.001

#: What each stage is allowed to move by. Boxes are in pixels of the letterboxed input; scores are
#: probabilities. ``0`` means the two sides must agree exactly -- counts and class ids are
#: decisions, not measurements, and a tolerance on a decision is meaningless. ``None`` means the
#: number is reported and nothing is held to it.
#:
#: ``head.boxes`` covers the anchors that clear :data:`CONF`, and ``head.boxes.all`` covers all
#: 8,400 of them. The split is not a convenience. Measured over everything, the worst disagreement
#: on ``large`` is 1.24e-02 px at anchor 6540 -- whose score is 0.000000, so no caller can ever
#: receive it, and neither implementation reads it: both take a top-k first. Over the anchors that
#: actually clear the threshold the same image reads 2.44e-04 px, and above 0.01 it reads
#: 6.10e-05. Gating on the whole grid would hold the family to the float noise of its own dead
#: anchors, which is a number that can fail without anything being wrong. Gating quietly on the
#: survivors alone would hide that the grid moves at all, so both are measured and one is gated.
TOLERANCE = {
    "preprocess": 1e-6,
    "layer": 2e-3,
    "head.boxes": 1e-2,
    "head.boxes.all": None,
    "head.scores": 1e-3,
    "detections.boxes": 1e-2,
    "detections.scores": 1e-3,
    "detections.classes": 0,
    "detections.count": 0,
}


def beyond(limit: float | None, value: float | None) -> bool:
    """Whether one measurement fails its stage's gate.

    The single definition of "this disagreed", because there were two and they disagreed with each
    other: the table counted a shape mismatch on a reported-only row as a failure and the
    falsification summary did not, so the same event was a failure in one place and invisible in
    the other. A shape mismatch always fails -- two stages that are not even the same size have
    not been compared at all, whatever the row is held to.
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
    artifacts.

    ``tools/bench/yolov26.py`` imports this rather than building its own, following
    ``tools/bench/easyocr.py``: the thing being timed has to be the thing that was verified. Its
    own copy had drifted already -- no version pin, no ``end2end`` assertion, and an ``imgsz``
    hardcoded to 640 -- so under 8.3.222 it would have timed the one-to-many head against the
    vendor's full decode and printed the ratio as a speed-up.
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
    # Resolved first: a symlink's relative target is read against the *link's* directory, not the
    # caller's, so a relative checkpoint path silently points at nothing under tools/verify/.
    linked.symlink_to(checkpoint.resolve())
    try:
        model = YOLO(str(linked), task="detect").model
    finally:
        linked.unlink(missing_ok=True)

    head = model.model[-1]
    if not head.end2end:
        raise SystemExit(
            "the reference loaded this head with end2end false, so it would run the one-to-many "
            "branch and NMS. That is not what this family is; see the module docstring."
        )
    return model.eval().fuse().eval().to(device)


def observe(model, batch: torch.Tensor, kind: str) -> dict[str, torch.Tensor]:
    """Run one batch and return every stage it produces, keyed the same on both sides.

    *kind* selects which of the two implementations *model* is. The two run different code and
    reach the same quantities, which is the whole point; what they must not do is disagree about
    what a stage is called, so the naming lives here rather than in either branch.
    """
    head = model.model[-1]
    seen: dict[str, torch.Tensor] = {}
    #: The head is hooked along with the 23 layers, and that is what stops its convolutions being
    #: run twice. They are a seventh of the forward pass on nano and are needed for the head rows,
    #: so recovering them by calling the head again cost a measurable fraction of every comparison
    #: -- doubled, because both sides pay it, and multiplied again by every falsification pass.
    handles = [
        layer.register_forward_hook(
            lambda _m, _i, out, index=index: seen.__setitem__(f"layer.{index:02d}", out)
        )
        for index, layer in enumerate(model.model)
    ]
    try:
        with torch.no_grad():
            rows = model(batch)
    finally:
        for handle in handles:
            handle.remove()

    raw = seen.pop(f"layer.{len(model.model) - 1:02d}")
    if kind == "vendor":
        feats = [seen[f"layer.{i:02d}"] for i in model.sources[-1]]
        shapes = [(int(f.shape[2]), int(f.shape[3])) for f in feats]
        with torch.no_grad():
            # ``_decode`` returns mask coefficients as well; this gate compares detection
            # heads, whose coefficient tensor is zero-width, so it is unpacked and dropped.
            boxes, logits, _coefficients = model._decode(raw, shapes)
        seen["head.boxes.all"] = boxes
        seen["head.scores"] = logits.sigmoid()
    else:
        # ``Detect.forward`` returns ``(y, preds)`` off the export path, and ``preds["one2one"]``
        # is what its own ``_inference`` consumes -- so the decode is re-derived from the
        # convolutions the hook already caught, rather than from a second ``forward_head``.
        rows, preds = raw
        with torch.no_grad():
            decoded = head._inference(preds["one2one"])
        boxes, scores = decoded.split((4, head.nc), 1)
        seen["head.boxes.all"] = boxes.transpose(1, 2)
        seen["head.scores"] = scores.transpose(1, 2)

    # The head rows a caller can actually reach. Selected by *our* scores on both sides, so the
    # two are compared on the same anchors rather than each on the ones it happens to rank highest
    # -- a comparison over two different populations measures the selection, not the boxes.
    live = seen["head.scores"][0].amax(1) > CONF
    seen["head.boxes"] = seen["head.boxes.all"][0][live]

    kept = rows[0][rows[0][:, 4] > CONF]
    seen["detections.boxes"] = kept[:, :4]
    seen["detections.scores"] = kept[:, 4]
    seen["detections.classes"] = kept[:, 5].to(torch.int64)
    seen["detections.count"] = torch.tensor(len(kept))
    return seen


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


def difference(ours: torch.Tensor, theirs: torch.Tensor) -> float | None:
    """How far apart two stages are, or ``None`` if they are not even the same shape."""
    if ours.shape != theirs.shape:
        return None
    return float((ours.float() - theirs.float()).abs().max())


def expectations(reference, images: list[Path], imgsz: int) -> dict[str, dict]:
    """What the reference produces for each image: the batch it was given, and its stages.

    Computed once and reused across every falsification, because no perturbation touches it --
    ``falsify`` only ever mutates the vendor, and the batch both sides are handed depends on the
    image and ``imgsz`` alone. Recomputing it per perturbation re-ran a full reference forward and
    reloaded the checkpoint five times over to arrive at the same numbers.
    """
    prepared = {}
    for path in images:
        image = load_image(str(path))
        ours, theirs = preprocessed(image, imgsz)
        prepared[path.name] = {
            "batch": ours,
            "preprocess": difference(ours, theirs),
            "stages": observe(reference, ours, "reference"),
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

    measured: dict[str, dict[str, float | None]] = {}
    for name, ready in prepared.items():
        measured.setdefault("preprocess", {})[name] = ready["preprocess"]
        # The same batch to both, so the 23 layer rows measure the layers and nothing else.
        seen = observe(vendor, ready["batch"], "vendor")
        expected = ready["stages"]
        # A stage one side produced and the other did not is not a small disagreement, and a
        # comparison that quietly skipped it would read as agreement. Recorded before the loop
        # below, which consumes ``seen`` as it goes.
        for stage in sorted(set(seen) ^ set(expected)):
            measured.setdefault(stage, {})[name] = None
        for stage in sorted(set(seen) & set(expected)):
            measured.setdefault(stage, {})[name] = difference(seen.pop(stage), expected[stage])
    return measured


def published(variant: str) -> Path:
    """The published checkpoint for *variant*, as the manifest names it.

    Through :func:`mozo.weights.resolve` rather than by sorting revision directory names, which is
    what the sibling bench and every other tool here do. A directory listing can select a revision
    the manifest does not publish -- so the reference table would be measured against bytes the
    everyday gate never checks, and nothing would say so.
    """
    try:
        return Path(resolve("yolov26", variant, "torch-fp32"))
    except Exception as failure:
        raise SystemExit(f"no weights for yolov26/{variant}: {failure} "
                         "-- run tools/fetch/yolov26.py") from failure


#: Deliberate breakages, and the stage each should first be caught at. A gate that has never
#: failed has not been shown to work, and a gate that fails everywhere at once has not been shown
#: to localise -- which is most of what a per-stage table is for.
FALSIFICATIONS = {
    # Not every layer: the offset is additive and small, so early rows whose activations are
    # large next to it stay inside 2e-3 and the drift only crosses it from layer 8 on. What the
    # row is actually checking is the boundary -- a change inside the network must never move the
    # preprocessing, and if it does, the two sides are not being handed the same batch.
    "batchnorm-eps": "layers, then the head and the detections -- never the preprocessing",
    "anchor-offset": "the boxes, leaving every layer and every score alone",
    "stride-swap": "the boxes, leaving every layer and every score alone",
    "topk-budget": "the detection count, and the rows that follow from it",
    "sigmoid-order": "nothing -- the control, a change that provably cannot move a number",
}


def falsify(vendor, which: str) -> None:
    """Perturb one constant, so the table can be watched to fail where that constant reaches."""
    if which == "batchnorm-eps":
        # Fusion has already folded the epsilon in, so this re-folds every convolution with a
        # different one -- which is what an unfused model would have done from the start.
        for module in vendor.modules():
            if isinstance(module, torch.nn.Conv2d) and module.bias is not None:
                module.bias.data.add_(1e-4)
    elif which == "anchor-offset":
        vendor._anchor_cache.clear()
        original = vendor._anchor_grid

        def shifted(shapes, dtype, device):
            anchors, scales = original(shapes, dtype, device)
            return anchors - 0.5, scales

        vendor._anchor_grid = shifted
    elif which == "stride-swap":
        vendor._anchor_cache.clear()
        vendor.strides = [vendor.strides[0]] * len(vendor.strides)
    elif which == "topk-budget":
        vendor.max_det = vendor.max_det // 2
    elif which == "sigmoid-order":
        pass  # the control: a change that provably cannot move a number


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

    def vendor():
        """A clean vendor, because ``falsify`` mutates the one it is given."""
        built = build_detector(str(checkpoint), fuse=True)
        return built.eval()

    prepared = expectations(reference, images, vendor().imgsz)

    if arguments.falsify == "all":
        print("Falsifying the gate. Each row should fail where its constant reaches, and only "
              "there.\n")
        for which, expectation in FALSIFICATIONS.items():
            measured = compare(vendor(), prepared, perturb=which)
            moved = sorted({stage for stage, _ in failures(measured)})
            print(f"{which}\n  expected: {expectation}\n  moved:    {moved or 'nothing'}\n")
        return 0

    import ultralytics

    print(f"variant     yolov26/{arguments.variant}")
    print(f"reference   ultralytics {ultralytics.__version__}")
    print(f"torch       {torch.__version__}, CPU, fused")
    print(f"images      {', '.join(p.name for p in images)}")

    measured = compare(vendor(), prepared, perturb=arguments.falsify)
    code = report(measured, images)

    if arguments.write and code == 0:
        # Merged rather than overwritten, so recording one variant does not silently drop the
        # four that were measured on earlier runs -- a table missing four fifths of the family
        # still looks complete.
        recorded = json.loads(TABLE.read_text()) if TABLE.exists() else {}
        recorded.update({
            "reference": f"ultralytics {ultralytics.__version__}",
            "torch": torch.__version__,
            "device": "cpu",
            "fused": True,
            "conf": CONF,
        })
        recorded.setdefault("variants", {})[arguments.variant] = {
            "images": [p.name for p in images],
            "stages": {stage: {"tolerance": tolerance_for(stage), "per_image": per_image}
                       for stage, per_image in measured.items()},
        }
        TABLE.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        print(f"wrote yolov26/{arguments.variant} to {TABLE.relative_to(ROOT)} "
              f"({len(recorded['variants'])} variant(s) recorded)")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
