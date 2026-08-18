"""Tests for runtime selection.

What these protect is a judgement call: ``auto`` prefers torch on every device because that is
what measurement showed, and a future edit that promotes ONNX on a hunch should have to change
a test that says so.
"""

from __future__ import annotations

import pytest

from mozo.runtimes import RuntimeError_, providers_for, select_runtime

BOTH = ["onnx-fp32", "torch-fp32"]


class TestSelectRuntime:
    @pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1", "mps"])
    def test_auto_prefers_torch_on_every_device(self, device):
        assert select_runtime(device, BOTH) == "torch-fp32"

    def test_auto_takes_what_exists_when_torch_is_absent(self):
        assert select_runtime("cpu", ["onnx-fp32"]) == "onnx-fp32"

    def test_auto_falls_through_to_anything_published(self):
        assert select_runtime("cpu", ["coreml-fp16"]) == "coreml-fp16"

    def test_explicit_request_is_honoured(self):
        assert select_runtime("cpu", BOTH, requested="onnx-fp32") == "onnx-fp32"

    def test_explicit_request_for_something_unpublished_fails_loudly(self):
        with pytest.raises(RuntimeError_, match="onnx-fp16"):
            select_runtime("cpu", BOTH, requested="onnx-fp16")

    def test_nothing_published_is_an_error(self):
        with pytest.raises(RuntimeError_, match="nothing runnable"):
            select_runtime("cpu", [])

    def test_auto_skips_a_runtime_whose_library_is_missing(self, monkeypatch):
        """`auto` must not hand back an artifact this machine cannot execute."""
        import mozo.runtimes as module
        monkeypatch.setitem(module._REQUIRES, "coreml", "a_module_that_does_not_exist")
        assert select_runtime("mps", ["coreml-fp32", "torch-fp32"]) == "torch-fp32"

    def test_asking_by_name_for_a_missing_library_still_returns_the_key(self, monkeypatch):
        """An explicit request is honoured up to the point the runner explains what is missing."""
        import mozo.runtimes as module
        monkeypatch.setitem(module._REQUIRES, "coreml", "a_module_that_does_not_exist")
        assert select_runtime("mps", ["coreml-fp32", "torch-fp32"], requested="coreml-fp32") == "coreml-fp32"


class TestProviders:
    def test_cuda_asks_for_the_gpu_first_and_keeps_cpu_as_a_floor(self):
        assert providers_for("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_mps_does_not_ask_for_coreml(self):
        """The CoreML provider placed 608 of 1415 RF-DETR nodes and then failed at run time."""
        assert providers_for("mps") == ["CPUExecutionProvider"]

    def test_cpu(self):
        assert providers_for("cpu") == ["CPUExecutionProvider"]
