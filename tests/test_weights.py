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
        assert sorted(manifest["models"]) == ["toy/alpha", "toy/beta", "toy/split"]

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


class TestManifestRejectsHalfARuntime:
    """A part on its own is indistinguishable from a runtime, and would be offered as one."""

    def test_a_lone_part_is_refused(self, zoo, run_generator, tmp_path):
        (zoo / "toy" / "beta" / "2026-01-01" / "onnx-fp32-encoder.onnx").write_bytes(b"half")
        result = run_generator(tmp_path / "out.json", check=False)
        assert result.returncode != 0
        assert "names a part of 'onnx-fp32' but is the only one" in result.stderr + result.stdout

    def test_a_complete_split_runtime_is_accepted(self, manifest_file):
        """The fixture's own split variant is the positive case, so the check cannot be
        vacuously satisfied by refusing everything."""
        import json

        artifacts = json.loads(manifest_file.read_text())["models"]["toy/split"]["revisions"]
        keys = artifacts["2026-01-01"]["artifacts"]
        assert {"onnx-fp32-encoder", "onnx-fp32-decoder"} <= set(keys)


class TestSplitRuntimes:
    """A runtime is not always one file.

    SAM 2 exports its encoder and its decoder as separate graphs on purpose: the expensive half
    depends only on the image, and welding them together would forfeit the reuse that makes a
    second click cheap. So an artifact key is ``<framework>-<precision>`` with an optional part
    on the end, and everything that chooses or fetches a runtime has to cope with both shapes.
    """

    def test_parts_collapse_into_one_runtime_name(self, weights):
        """A caller choosing a runtime should see ``onnx-fp32``, not two halves to rejoin."""
        from mozo.runtimes import runnable
        from mozo.weights import artifacts

        published = artifacts("toy", "split")
        assert "onnx-fp32-encoder" in published, "the manifest still lists the parts"
        assert runnable(published) == ["coreml-fp16", "onnx-fp32", "torch-fp32"]

    def test_a_split_runtime_resolves_to_all_of_its_parts(self, weights):
        from mozo.weights import parts

        got = parts("toy", "split", "onnx-fp32")
        assert sorted(got) == ["decoder", "encoder"]
        assert got["encoder"].read_bytes() == b"split-onnx-encoder"
        assert got["decoder"].read_bytes() == b"split-onnx-decoder"

    def test_three_parts_are_no_harder_than_two(self, weights):
        """CoreML splits the prompt encoder out as well, so two is not the only answer."""
        from mozo.weights import parts

        assert sorted(parts("toy", "split", "coreml-fp16")) == ["decoder", "encoder", "prompt"]

    def test_a_whole_runtime_is_one_unnamed_part(self, weights):
        """The single-file case goes through the same door rather than a special one."""
        from mozo.weights import parts

        got = parts("toy", "split", "torch-fp32")
        assert list(got) == [""]
        assert got[""].read_bytes() == b"split-torch"

    def test_parts_verifies_every_piece(self, weights, zoo):
        """Each part is fetched through resolve, so a corrupt half is caught like a whole one."""
        from mozo.weights import WeightsError, parts

        (zoo / "toy" / "split" / "2026-01-01" / "onnx-fp32-decoder.onnx").write_bytes(b"tampered")
        with pytest.raises(WeightsError, match="does not match the manifest"):
            parts("toy", "split", "onnx-fp32")

    def test_an_unpublished_runtime_says_what_there_is(self, weights):
        from mozo.weights import WeightsError, parts

        with pytest.raises(WeightsError, match="coreml-fp16, onnx-fp32, torch-fp32"):
            parts("toy", "split", "tensorrt-fp16")

    def test_a_part_key_is_not_a_runtime_name(self, weights):
        """``onnx-fp32-encoder`` names a file, not something you can choose to run as."""
        from mozo.weights import WeightsError, parts

        with pytest.raises(WeightsError, match="publishes no"):
            parts("toy", "split", "onnx-fp32-encoder")

    def test_selection_never_offers_half_a_runtime(self, weights):
        """``auto`` must not hand back an encoder as though it were something runnable."""
        from mozo.runtimes import select_runtime
        from mozo.weights import artifacts

        assert select_runtime("cpu", artifacts("toy", "split")) == "torch-fp32"

    def test_single_file_families_are_untouched(self, weights):
        """The change must be invisible to every family that publishes whole runtimes."""
        from mozo.runtimes import runnable, select_runtime
        from mozo.weights import artifacts, parts

        published = artifacts("toy", "alpha")
        assert runnable(published) == ["onnx-fp32", "torch-fp32"]
        assert select_runtime("cpu", published) == "torch-fp32"
        assert parts("toy", "alpha", "onnx-fp32")[""].read_bytes() == b"alpha-onnx"

    def test_data_artifacts_are_not_offered_as_runtimes(self, weights):
        """``labels`` is neither a companion nor something a model can be executed as. Offering
        it would contradict the list ``runnable`` gives a caller to choose from."""
        from mozo.runtimes import runnable
        from mozo.weights import WeightsError, artifacts, parts

        published = artifacts("toy", "split")
        assert "labels" in published, "the fixture publishes one, so the filter is exercised"
        assert "labels" not in runnable(published)

        with pytest.raises(WeightsError, match="publishes no 'labels'"):
            parts("toy", "split", "labels")

    def test_every_part_brings_the_licence_and_the_notice(self, weights):
        from mozo.weights import parts

        for part in parts("toy", "split", "coreml-fp16").values():
            assert (part.parent / "LICENSE").read_bytes() == b"Apache-2.0"
            assert (part.parent / "NOTICE").read_bytes() == b"attribution"

    def test_one_run_names_every_file_that_has_to_be_placed(self, weights, monkeypatch):
        """Offline, a caller placing files by hand should learn about all three at once rather
        than discovering the next one each time they rerun."""
        from mozo.weights import WeightsError, parts

        monkeypatch.setenv("MOZO_OFFLINE", "1")
        with pytest.raises(WeightsError) as raised:
            parts("toy", "split", "coreml-fp16")
        message = str(raised.value)
        for part in ("coreml-fp16-encoder", "coreml-fp16-decoder", "coreml-fp16-prompt"):
            assert part in message, f"{part} was not named"
        assert "coreml-fp16 is 3 files and 3 are not cached" in message

    def test_resolving_a_split_runtime_by_name_says_what_to_do(self, weights):
        """``select_runtime`` hands back ``onnx-fp32`` and every adapter passes that to
        ``resolve``. For a split runtime there is no such file, and the error has to point
        somewhere rather than contradict the name the caller was just given."""
        from mozo.runtimes import select_runtime
        from mozo.weights import WeightsError, artifacts, resolve

        runtime = select_runtime("cpu", artifacts("toy", "split"), requested="onnx-fp32")
        with pytest.raises(WeightsError, match="as 2 files, not one"):
            resolve("toy", "split", runtime)

    def test_auto_can_return_a_split_runtime_when_it_is_the_only_one(self, weights):
        """It has to -- there is nothing else to run. What that costs is that the name ``auto``
        hands back is not a file, which is why ``resolve`` explains itself and ``parts`` exists.
        A caller that uses ``parts`` is unaffected; the pairing is tested above."""
        from mozo.runtimes import select_runtime

        assert select_runtime("cpu", ["onnx-fp32-encoder", "onnx-fp32-decoder"]) == "onnx-fp32"

