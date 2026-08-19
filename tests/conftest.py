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


@pytest.fixture(scope="session")
def image():
    """The fixture photograph, decoded to mozo's contract (RGB)."""
    from mozo.utils import load_image

    return load_image(str(FIXTURE))


@pytest.fixture(scope="session")
def payload() -> bytes:
    """The fixture photograph as encoded bytes, i.e. what an HTTP request body carries."""
    return FIXTURE.read_bytes()

#: What a synthetic zoo contains: two variants of one family, one revision apart, with a
#: different artifact set each so selection and absence are both exercised.
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
