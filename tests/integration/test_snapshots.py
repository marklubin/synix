"""Integration tests for immutable Synix snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synix import FlatFile, Pipeline, SearchIndex, Source
from synix.build.artifacts import ArtifactStore
from synix.build.object_store import ObjectStore
from synix.build.refs import RefStore
from synix.build.runner import run
from synix.build.snapshots import _pipeline_fingerprint, list_runs
from synix.build.validators import RequiredField
from synix.transforms import CoreSynthesis, EpisodeSummary, MonthlyRollup


def _build_pipeline(build_dir: Path, source_dir: Path, *, synix_dir: Path | None = None) -> Pipeline:
    pipeline = Pipeline("snapshot-pipeline")
    pipeline.build_dir = str(build_dir)
    pipeline.source_dir = str(source_dir)
    pipeline.synix_dir = str(synix_dir) if synix_dir is not None else str(build_dir.parent / ".synix")
    pipeline.llm_config = {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    transcripts = Source("transcripts")
    episodes = EpisodeSummary("episodes", depends_on=[transcripts])
    monthly = MonthlyRollup("monthly", depends_on=[episodes])
    core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

    pipeline.add(transcripts, episodes, monthly, core)
    pipeline.add(SearchIndex("memory-index", sources=[episodes, monthly, core], search=["fulltext"]))
    pipeline.add(FlatFile("context-doc", sources=[core], output_path=str(build_dir / "context.md")))
    return pipeline


class TestSnapshots:
    def test_pipeline_fingerprint_ignores_machine_local_paths_and_secrets(self, tmp_path):
        """Fingerprint should reflect logical build config, not local directories or API keys."""
        pipeline_a = Pipeline(
            "fingerprint-test",
            source_dir=str(tmp_path / "sources-a"),
            build_dir=str(tmp_path / "build-a"),
            synix_dir=str(tmp_path / ".synix-a"),
            llm_config={
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.3,
                "api_key": "secret-a",
            },
        )
        pipeline_b = Pipeline(
            "fingerprint-test",
            source_dir=str(tmp_path / "sources-b"),
            build_dir=str(tmp_path / "build-b"),
            synix_dir=str(tmp_path / ".synix-b"),
            llm_config={
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.3,
                "api_key": "secret-b",
            },
        )

        transcripts_a = Source("transcripts")
        episodes_a = EpisodeSummary("episodes", depends_on=[transcripts_a])
        pipeline_a.add(transcripts_a, episodes_a)
        pipeline_a.add(SearchIndex("memory-index", sources=[episodes_a], search=["fulltext"]))

        transcripts_b = Source("transcripts")
        episodes_b = EpisodeSummary("episodes", depends_on=[transcripts_b])
        pipeline_b.add(transcripts_b, episodes_b)
        pipeline_b.add(SearchIndex("memory-index", sources=[episodes_b], search=["fulltext"]))

        assert _pipeline_fingerprint(pipeline_a) == _pipeline_fingerprint(pipeline_b)

    def test_build_commits_manifest_and_snapshot(self, tmp_path, source_dir_with_fixtures, mock_llm):
        """A successful build records immutable objects and moves HEAD."""
        build_dir = tmp_path / "build"
        pipeline = _build_pipeline(build_dir, source_dir_with_fixtures)

        result = run(pipeline, source_dir=str(source_dir_with_fixtures))

        assert result.snapshot_oid is not None
        assert result.manifest_oid is not None
        assert result.run_ref is not None
        assert result.head_ref == "refs/heads/main"
        assert result.synix_dir == str(tmp_path / ".synix")

        object_store = ObjectStore(tmp_path / ".synix")
        ref_store = RefStore(tmp_path / ".synix")

        snapshot = object_store.get_json(result.snapshot_oid)
        manifest = object_store.get_json(result.manifest_oid)

        assert snapshot["type"] == "snapshot"
        assert snapshot["manifest_oid"] == result.manifest_oid
        assert snapshot["run_id"]
        assert manifest["type"] == "manifest"
        assert manifest["pipeline_name"] == "snapshot-pipeline"
        assert len(manifest["artifacts"]) > 0
        assert set(manifest["projections"]) == {"memory-index", "context-doc"}

        assert ref_store.read_head_target() == "refs/heads/main"
        assert ref_store.read_ref("HEAD") == result.snapshot_oid
        assert ref_store.read_ref(result.run_ref) == result.snapshot_oid

    def test_projection_input_closure_uses_projection_sources(self, tmp_path, source_dir_with_fixtures, mock_llm):
        """Projection input_oids should only include artifacts from the configured source layers."""
        build_dir = tmp_path / "build"
        pipeline = _build_pipeline(build_dir, source_dir_with_fixtures)

        result = run(pipeline, source_dir=str(source_dir_with_fixtures))
        assert result.manifest_oid is not None

        object_store = ObjectStore(tmp_path / ".synix")
        manifest = object_store.get_json(result.manifest_oid)
        projection_oid = manifest["projections"]["memory-index"]
        projection = object_store.get_json(projection_oid)

        store = ArtifactStore(build_dir)
        expected_count = (
            len(store.list_artifacts("episodes"))
            + len(store.list_artifacts("monthly"))
            + len(store.list_artifacts("core"))
        )

        assert len(projection["input_oids"]) == expected_count

    def test_successive_builds_preserve_old_run_ref(self, tmp_path, source_dir_with_fixtures, mock_llm):
        """Each successful build gets a new snapshot while older run refs remain resolvable."""
        build_dir = tmp_path / "build"
        pipeline = _build_pipeline(build_dir, source_dir_with_fixtures)

        first = run(pipeline, source_dir=str(source_dir_with_fixtures))
        second = run(pipeline, source_dir=str(source_dir_with_fixtures))

        assert first.snapshot_oid is not None
        assert second.snapshot_oid is not None
        assert first.snapshot_oid != second.snapshot_oid
        assert first.run_ref != second.run_ref

        object_store = ObjectStore(tmp_path / ".synix")
        ref_store = RefStore(tmp_path / ".synix")
        first_snapshot = object_store.get_json(first.snapshot_oid)
        second_snapshot = object_store.get_json(second.snapshot_oid)

        assert first_snapshot["manifest_oid"] == first.manifest_oid
        assert second_snapshot["manifest_oid"] == second.manifest_oid
        assert second_snapshot["parent_snapshot_oids"] == [first.snapshot_oid]
        assert ref_store.read_ref("HEAD") == second.snapshot_oid
        assert ref_store.read_ref(first.run_ref) == first.snapshot_oid
        assert ref_store.read_ref(second.run_ref) == second.snapshot_oid

        runs = list_runs(build_dir)
        assert {run_info["ref"] for run_info in runs} == {first.run_ref, second.run_ref}

    def test_source_change_creates_new_manifest(self, tmp_path, source_dir_with_fixtures, mock_llm):
        """A changed source export produces a new manifest while keeping old snapshots intact."""
        build_dir = tmp_path / "build"
        pipeline = _build_pipeline(build_dir, source_dir_with_fixtures)

        first = run(pipeline, source_dir=str(source_dir_with_fixtures))

        claude_path = source_dir_with_fixtures / "claude_export.json"
        data = json.loads(claude_path.read_text())
        data["conversations"].append(
            {
                "uuid": "conv-new-snapshot-test",
                "title": "Snapshot test conversation",
                "created_at": "2024-03-25T10:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "msg-new-1",
                        "sender": "human",
                        "text": "Tell me about snapshotting.",
                        "created_at": "2024-03-25T10:00:00Z",
                    },
                    {
                        "uuid": "msg-new-2",
                        "sender": "assistant",
                        "text": "Snapshotting captures immutable build state.",
                        "created_at": "2024-03-25T10:01:00Z",
                    },
                ],
            }
        )
        claude_path.write_text(json.dumps(data))

        second = run(pipeline, source_dir=str(source_dir_with_fixtures))

        assert first.snapshot_oid is not None
        assert second.snapshot_oid is not None
        assert first.manifest_oid is not None
        assert second.manifest_oid is not None
        assert first.manifest_oid != second.manifest_oid

        object_store = ObjectStore(tmp_path / ".synix")
        first_manifest = object_store.get_json(first.manifest_oid)
        second_manifest = object_store.get_json(second.manifest_oid)

        assert len(second_manifest["artifacts"]) > len(first_manifest["artifacts"])

    def test_validation_failure_does_not_advance_snapshot_refs(self, tmp_path, source_dir_with_fixtures, mock_llm):
        """A failing validation result should not record or advance snapshot refs."""
        build_dir = tmp_path / "build"
        synix_dir = tmp_path / ".synix"
        pipeline = _build_pipeline(build_dir, source_dir_with_fixtures, synix_dir=synix_dir)

        core = next(layer for layer in pipeline.layers if layer.name == "core")
        pipeline.add_validator(RequiredField(field="missing_required_field", layers=[core]))

        result = run(pipeline, source_dir=str(source_dir_with_fixtures), validate=True)

        assert result.validation is not None
        assert not result.validation.passed
        assert result.snapshot_oid is None
        assert result.run_ref is None
        assert result.manifest_oid is None
        assert list_runs(build_dir, synix_dir=synix_dir) == []

    def test_flatfile_projection_must_live_under_build_dir(self, tmp_path, source_dir_with_fixtures, mock_llm):
        """Snapshotting rejects flat-file projections that point outside the build root."""
        build_dir = tmp_path / "build"
        pipeline = _build_pipeline(build_dir, source_dir_with_fixtures)
        pipeline.projections = [
            proj for proj in pipeline.projections if not isinstance(proj, FlatFile)
        ] + [FlatFile("external-doc", sources=[next(layer for layer in pipeline.layers if layer.name == "core")], output_path=str(tmp_path / "outside.md"))]

        with pytest.raises(ValueError, match="must write inside the build directory"):
            run(pipeline, source_dir=str(source_dir_with_fixtures))
