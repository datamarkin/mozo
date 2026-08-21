"""SAM 2: what is true of this family and not of the others.

The contract every promptable family shares -- ranking, naming, batching, the server's prompt
parsing -- is in ``test_promptable.py``, checked against all of them. What is left here is SAM 2's
own: four variants where the others have one, and several published runtimes where the others
publish only torch.
"""

from __future__ import annotations

import pytest

from mozo.registry import get_model_info
from mozo.vendors.sam2_deploy.config import SPECS

FAMILY = "sam2"


def test_registry_agrees_with_the_adapter():
    """The variant list is written twice -- here and in the adapter -- so that answering "what
    exists" needs no torch import. This is what holds the two copies in step."""
    from mozo.adapters.sam2 import Sam2Predictor

    entry = get_model_info(FAMILY)
    assert entry["adapter_class"] == "Sam2Predictor"
    assert entry["module"] == "mozo.adapters.sam2"
    assert entry["task_type"] == "promptable_segmentation"
    assert set(entry["variants"]) == set(Sam2Predictor.VARIANTS)


def test_the_adapter_publishes_every_variant_the_vendor_can_build():
    """The vendor knows four geometries; a fifth published with no geometry, or a geometry with
    nothing published, is a mismatch nothing else would report."""
    from mozo.adapters.sam2 import Sam2Predictor

    assert set(Sam2Predictor.VARIANTS) == set(SPECS)


def test_a_graph_runtime_is_refused_rather_than_quietly_replaced():
    """SAM 2 publishes ONNX and CoreML, and no promptable adapter can execute either yet -- such
    a model exports as several graphs. Answering the request with torch would look like it had
    been honoured, so it raises."""
    from mozo.adapters.sam2 import Sam2Predictor
    from mozo.runtimes import RuntimeError_

    with pytest.raises(RuntimeError_, match="can only execute torch"):
        Sam2Predictor("tiny", device="cpu", runtime="onnx-fp32")


def test_auto_never_chooses_a_runtime_the_adapter_would_refuse():
    """The point of declaring the capability to ``select_runtime`` rather than checking it
    afterwards. ``_PREFERENCE`` leads with CoreML on Apple silicon and SAM 2 publishes
    ``coreml-fp16``; the day someone measures it and adds it to that table, ``auto`` on a Mac
    would otherwise start choosing an artifact this adapter refuses -- breaking the *default* path."""
    from mozo.adapters.sam2 import Sam2Predictor
    from mozo.runtimes import select_runtime
    from mozo.weights import artifacts

    for variant in Sam2Predictor.VARIANTS:
        published = artifacts(FAMILY, variant)
        for device in ("cpu", "cuda", "mps"):
            chosen = select_runtime(device, published, "auto", executes=Sam2Predictor.EXECUTES)
            assert chosen.startswith("torch"), f"{variant} on {device} chose {chosen}"
