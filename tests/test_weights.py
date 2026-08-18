"""Tests for manifest generation and weight resolution.

These run against the synthetic zoo in ``conftest.py``, not against real checkpoints. The
resolver has no idea what a model is -- it fetches a named file and checks its hash -- so a
14-byte artifact exercises every path a 386 MB one does, in milliseconds and without a network.
"""

from __future__ import annotations

import hashlib
import json

import pytest


class TestManifest:
    def test_derives_everything_from_the_tree(self, manifest_file):
        manifest = json.loads(manifest_file.read_text())
        assert manifest["schema"] == 1
        assert sorted(manifest["models"]) == ["toy/alpha", "toy/beta"]

        alpha = manifest["models"]["toy/alpha"]
        assert sorted(alpha["revisions"]) == ["2026-01-01", "2026-02-01"]
        assert alpha["latest"] == "2026-02-01"

    def test_artifact_key_is_the_file_stem(self, manifest_file):
        artifacts = json.loads(manifest_file.read_text())["models"]["toy/alpha"]["revisions"]["2026-02-01"]["artifacts"]
        assert sorted(artifacts) == ["LICENSE", "labels", "onnx-fp32", "torch-fp32"]

    def test_records_real_size_and_hash(self, manifest_file, zoo):
        record = json.loads(manifest_file.read_text())["models"]["toy/beta"]["revisions"]["2026-01-01"]["artifacts"]["torch-fp32"]
        source = zoo / record["path"]
        assert record["size"] == source.stat().st_size
        assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()

    def test_carries_no_hand_typed_field(self, manifest_file):
        """Everything in a revision is read off the disk. Nothing is asserted by a human."""
        revision = json.loads(manifest_file.read_text())["models"]["toy/beta"]["revisions"]["2026-01-01"]
        assert set(revision) == {"artifacts"}

    def test_is_idempotent(self, run_generator, tmp_path):
        """A stale manifest is the realistic failure, so regeneration has to be a no-op."""
        outputs = []
        for name in ("first.json", "second.json"):
            out = tmp_path / name
            run_generator(out)
            outputs.append(out.read_bytes())
        assert outputs[0] == outputs[1]

    def test_refuses_a_revision_without_a_licence(self, run_generator, zoo, tmp_path):
        """Publishing weights without their terms is the one thing generation will not do."""
        (zoo / "toy" / "beta" / "2026-01-01" / "LICENSE").unlink()
        result = run_generator(tmp_path / "m.json", check=False)
        assert result.returncode == 1
        assert "LICENSE" in result.stderr

    def test_ignores_dotfiles(self, run_generator, zoo, tmp_path):
        """.DS_Store appears unbidden on macOS and must not become an artifact."""
        (zoo / "toy" / "beta" / "2026-01-01" / ".DS_Store").write_bytes(b"junk")
        out = tmp_path / "m.json"
        run_generator(out)
        artifacts = json.loads(out.read_text())["models"]["toy/beta"]["revisions"]["2026-01-01"]["artifacts"]
        assert sorted(artifacts) == ["LICENSE", "torch-fp32"]


class TestResolve:
    def test_downloads_verifies_and_caches(self, weights):
        path = weights.resolve("toy", "alpha")
        assert path.read_bytes() == b"alpha-torch-v2"
        assert path.parent.name == "2026-02-01"

    def test_licence_travels_with_the_weights(self, weights):
        path = weights.resolve("toy", "alpha")
        assert (path.parent / "LICENSE").read_bytes() == b"Apache-2.0"

    def test_revision_is_in_the_path_so_pinning_survives_a_cache_hit(self, weights):
        latest = weights.resolve("toy", "alpha")
        pinned = weights.resolve("toy", "alpha", revision="2026-01-01")
        assert pinned.read_bytes() == b"alpha-torch-v1"
        assert latest.read_bytes() == b"alpha-torch-v2"
        assert pinned != latest

    def test_second_call_is_a_cache_hit(self, weights, monkeypatch):
        first = weights.resolve("toy", "alpha")
        monkeypatch.setenv("MOZO_BASE_URL", "file:///nonexistent")
        assert weights.resolve("toy", "alpha") == first

    def test_selects_by_artifact_key(self, weights):
        assert weights.resolve("toy", "alpha", "onnx-fp32").read_bytes() == b"alpha-onnx"

    def test_unpublished_model_says_bring_your_own(self, weights):
        with pytest.raises(weights.WeightsError, match="checkpoint path"):
            weights.resolve("toy", "gamma")

    def test_unknown_revision_lists_what_exists(self, weights):
        with pytest.raises(weights.WeightsError, match="2026-01-01, 2026-02-01"):
            weights.resolve("toy", "alpha", revision="1999-01-01")

    def test_unknown_artifact_lists_what_exists(self, weights):
        with pytest.raises(weights.WeightsError, match="torch-fp32"):
            weights.resolve("toy", "beta", "onnx-fp32")

    def test_corrupt_download_is_rejected_and_leaves_nothing(self, weights, zoo):
        (zoo / "toy" / "beta" / "2026-01-01" / "torch-fp32.pth").write_bytes(b"tampered!!")
        with pytest.raises(weights.WeightsError, match="does not match the manifest"):
            weights.resolve("toy", "beta")
        cached = weights.cache_dir() / "toy" / "beta" / "2026-01-01"
        assert not list(cached.glob("torch-fp32*"))

    def test_offline_names_the_path_url_and_hash(self, weights, monkeypatch):
        monkeypatch.setenv("MOZO_OFFLINE", "1")
        with pytest.raises(weights.WeightsError) as error:
            weights.resolve("toy", "beta")
        message = str(error.value)
        assert "torch-fp32.pth" in message and "sha256" in message

    def test_offline_still_serves_what_is_cached(self, weights, monkeypatch):
        path = weights.resolve("toy", "beta")
        monkeypatch.setenv("MOZO_OFFLINE", "1")
        assert weights.resolve("toy", "beta") == path


class TestArtifacts:
    def test_lists_what_a_revision_publishes(self, weights):
        assert weights.artifacts("toy", "alpha") == ["labels", "onnx-fp32", "torch-fp32"]

    def test_omits_the_licence(self, weights):
        """LICENSE ships with everything rather than being an artifact you choose."""
        assert "LICENSE" not in weights.artifacts("toy", "alpha")

    def test_reads_the_revision_asked_for(self, weights):
        assert weights.artifacts("toy", "alpha", revision="2026-01-01") == ["torch-fp32"]
