"""Shared fixtures.

The weight system is deliberately model-agnostic: the generator hashes whatever files it finds,
and the resolver fetches whatever the manifest names. So it can be tested end to end against a
synthetic zoo whose artifacts are a few bytes each, with no checkpoints, no network, and no
model code. That is the whole point of these fixtures -- the real 3 GB tree proves the models,
not the plumbing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "tools" / "generate_manifest.py"

#: The one real photograph the model tests run on. Path rather than array, because half the
#: callers want the bytes and half want the decoded image.
FIXTURE = ROOT / "tests" / "fixtures" / "images" / "example.jpg"

#: Images with text on them, for the families that read rather than classify. Kept apart from
#: ``images/`` because every other family's gate and bench iterate that directory whole, and a
#: page of rendered text is not a photograph to compare detections on.
TEXT_FIXTURES = ROOT / "tests" / "fixtures" / "text"


def published(family: str, variant: str) -> list[str]:
    """The artifact keys *variant* publishes, or ``[]`` if it publishes nothing at all.

    Answered from the manifest, which ships in the wheel -- so no network and no cache. This is
    "is it published", not "are the bytes here": obtaining the bytes can still fail, which is
    why fixtures that build a predictor also catch :class:`WeightsError`.
    """
    from mozo.weights import WeightsError, artifacts

    try:
        return artifacts(family, variant)
    except WeightsError:
        return []


def require_weights(family: str, variant: str, runtime: str = "torch-fp32") -> None:
    """Skip unless *variant* publishes *runtime*."""
    if runtime not in published(family, variant):
        pytest.skip(f"{family}/{variant} does not publish {runtime}")


#: One step of PixelFlow's coordinate rounding, which is the most a boundary can move an edge by
#: after two faithful runtimes have both been through it.
#:
#: Not a tolerance on its own. What a runtime pairing may move by is this *plus* whatever that
#: family's exporter already guarantees at full precision, and those differ -- ``tools/export``
#: holds a YOLO graph to 1e-2 px and an RF-DETR graph to 1.0 px, because a transformer's
#: selection is a different numerical animal from a convolutional head's. Each suite derives its
#: own; only this step is shared, because only this step comes from PixelFlow.
COORD_STEP = 0.01


def as_pixelflow_reports(boxes, scores, class_ids):
    """Put a model's raw numbers through the same door mozo's results go through.

    PixelFlow rounds coordinates and confidences to its own precision, so a raw vendor output and
    a mozo result can only be compared after both have been rounded the same way. This hands the
    raw arrays to ``pf.detections.from_arrays`` -- exactly what
    :meth:`~mozo.adapters._yolo.YOLOPredictor.predict` does -- and reads the numbers back, so the
    rule applies itself and is never restated here.

    That indirection is the point. This helper used to hardcode the rounding, which meant mozo
    held a private copy of a policy PixelFlow owns: when PixelFlow stopped truncating boxes to
    whole pixels, every copy was silently wrong and five files had to be found and patched. A
    truncation is not a decimals value either, so importing a precision constant would not have
    caught it -- only going through the real thing does.

    Lives here rather than in a family's test because it is a fact about mozo's result boundary,
    not about any one model. ``tools/verify/*.py`` imports it too.
    """
    import numpy as np
    import pixelflow as pf

    rows = pf.detections.from_arrays(boxes=boxes, scores=scores, class_ids=class_ids).to_dict()
    return (np.array([row["bbox"] for row in rows], dtype=np.float64).reshape(-1, 4),
            np.array([row["confidence"] for row in rows], dtype=np.float64))


def as_pixelflow_classifications(scores, labels):
    """The same door, for the families that answer with a score per label instead of a box.

    ``pf.from_scores`` rounds confidences exactly as ``from_arrays`` does, and for exactly the same
    reason a raw vendor score cannot be compared to a mozo one without going through it. Written
    as a second helper rather than a branch in the first because the two take different arrays and
    return different things; what they share is refusing to restate PixelFlow's precision.

    Returns the whole rows rather than the confidences alone, so ``top_k``'s ordering and its
    label-to-id mapping come from PixelFlow too. Handing back scores only would leave every caller
    re-deriving the rank with its own ``sorted(..., key=-score)``, which is the same private copy
    of a PixelFlow rule that this helper exists to prevent -- one level along.
    """
    import pixelflow as pf

    return pf.from_scores(scores, labels=list(labels)).top_k(len(labels)).to_dict()


@pytest.fixture(scope="session")
def image():
    """The fixture photograph, decoded to mozo's contract (RGB)."""
    from mozo.image import load_image

    return load_image(str(FIXTURE))


@pytest.fixture(scope="session")
def payload() -> bytes:
    """The fixture photograph as encoded bytes, i.e. what an HTTP request body carries."""
    return FIXTURE.read_bytes()

#: What a synthetic zoo contains: two variants of one family, one revision apart, with a
#: different artifact set each so selection and absence are both exercised -- plus one variant
#: whose graph runtime is split across parts, because SAM 2 publishes an encoder and a decoder
#: rather than one file and the resolver has to rejoin them without knowing what they are.
_ZOO = {
    "toy/alpha": {
        "2026-01-01": {"torch-fp32.pth": b"alpha-torch-v1", "LICENSE": b"Apache-2.0"},
        "2026-02-01": {
            "torch-fp32.pth": b"alpha-torch-v2",
            "onnx-fp32.onnx": b"alpha-onnx",
            "labels.json": json.dumps([{"id": 1, "name": "cat"}, {"id": 5, "name": "dog"}]).encode(),
            "LICENSE": b"Apache-2.0",
        },
    },
    "toy/beta": {
        "2026-01-01": {"torch-fp32.pth": b"beta-torch", "LICENSE": b"MIT"},
    },
    "toy/split": {
        "2026-01-01": {
            # A NOTICE as well as a LICENCE, because the families that publish split runtimes
            # are the ones whose terms ask for attribution to travel with the copy -- and every
            # part has to bring both.
            "NOTICE": b"attribution",
            "labels.json": json.dumps([{"id": 0, "name": "thing"}]).encode(),
            "torch-fp32.pth": b"split-torch",
            "onnx-fp32-encoder.onnx": b"split-onnx-encoder",
            "onnx-fp32-decoder.onnx": b"split-onnx-decoder",
            "coreml-fp16-encoder.zip": b"split-coreml-encoder",
            "coreml-fp16-decoder.zip": b"split-coreml-decoder",
            "coreml-fp16-prompt.zip": b"split-coreml-prompt",
            "LICENSE": b"Apache-2.0",
        },
    },
}


@pytest.fixture
def run_generator(zoo: Path):
    """Run the manifest generator over the synthetic zoo, as a subprocess so the real CLI runs."""
    def run(out: Path, *, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(GENERATOR), "--weights-dir", str(zoo), "--out", str(out)],
            check=check, capture_output=True, text=True,
        )
    return run


@pytest.fixture
def zoo(tmp_path: Path) -> Path:
    """Build a synthetic ``weights/`` tree on disk and return its path."""
    root = tmp_path / "weights"
    for model_id, revisions in _ZOO.items():
        family, variant = model_id.split("/")
        for revision, files in revisions.items():
            directory = root / family / variant / revision
            directory.mkdir(parents=True)
            for name, payload in files.items():
                (directory / name).write_bytes(payload)
    return root


@pytest.fixture
def manifest_file(run_generator, tmp_path: Path) -> Path:
    """The manifest the generator writes for the synthetic zoo."""
    destination = tmp_path / "manifest.json"
    run_generator(destination)
    return destination


@pytest.fixture
def weights(monkeypatch, zoo: Path, manifest_file: Path, tmp_path: Path):
    """Point :mod:`mozo.weights` at the synthetic zoo, served from the local filesystem."""
    from mozo import weights as module

    monkeypatch.setattr(module, "_MANIFEST_PATH", manifest_file)
    monkeypatch.setattr(module, "_manifest", None)
    monkeypatch.setenv("MOZO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MOZO_BASE_URL", zoo.as_uri())
    monkeypatch.delenv("MOZO_OFFLINE", raising=False)
    yield module
    module._manifest = None
