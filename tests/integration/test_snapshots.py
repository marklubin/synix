"""Integration tests for immutable Synix snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synix import FlatFile, Pipeline, SearchIndex, Source
from synix.build.artifacts import ArtifactStore
from synix.build.object_store import SCHEMA_VERSION, ObjectStore
from synix.build.refs import RefStore
from synix.build.runner import run
from synix.build.snapshots import _pipeline_fingerprint, commit_build_snapshot, list_runs, start_build_transaction
from synix.build.validators import RequiredField
from synix.core.models import Artifact
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
    def test_commit_uses_transaction_state_not_mutable_build_files(self, tmp_path):
        """Snapshot commit should not reread manifest/provenance from the mutable build directory."""
        build_dir = tmp_path / "build"
        pipeline = Pipeline(
            "transaction-only",
            build_dir=str(build_dir),
            synix_dir=str(tmp_path / ".synix"),
        )
        transcripts = Source("transcripts")
        pipeline.add(transcripts)

        txn = start_build_transaction(pipeline, build_dir, run_id="20260306T120000000000Z")
        txn.record_artifact(
            Artifact(label="ep-1", artifact_type="episode", content="Captured before build-dir mutation."),
            layer_name="transcripts",
            layer_level=0,
            parent_labels=[],
        )

        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "manifest.json").write_text("{not valid json", encoding="utf-8")
        (build_dir / "provenance.json").write_text("{not valid json", encoding="utf-8")

        snapshot_info = commit_build_snapshot(txn)
        object_store = ObjectStore(tmp_path / ".synix")
        manifest = object_store.get_json(snapshot_info["manifest_oid"])

        assert manifest["artifacts"] == {"ep-1": txn.artifact_oids["ep-1"]}
        stored_artifact = object_store.get_json(manifest["artifacts"]["ep-1"])
        assert stored_artifact["label"] == "ep-1"

    def test_commit_rejects_when_head_advances_during_build(self, tmp_path):
        """Snapshot commit should fail closed if another writer advances HEAD first."""
        build_dir = tmp_path / "build"
        pipeline = Pipeline(
            "concurrent-build",
            build_dir=str(build_dir),
            synix_dir=str(tmp_path / ".synix"),
        )
        transcripts = Source("transcripts")
        pipeline.add(transcripts)

        txn = start_build_transaction(pipeline, build_dir, run_id="20260306T120000000001Z")
        txn.record_artifact(
            Artifact(label="ep-1", artifact_type="episode", content="Concurrent head test."),
            layer_name="transcripts",
            layer_level=0,
            parent_labels=[],
        )

        ref_store = RefStore(tmp_path / ".synix")
        ref_store.write_ref(txn.head_ref, "1" * 64)

        with pytest.raises(RuntimeError, match="HEAD advanced during build"):
            commit_build_snapshot(txn)

    def test_commit_rejects_artifact_id_content_mismatch(self, tmp_path):
        """Snapshot commit should not make corrupted cached artifacts canonical."""
        build_dir = tmp_path / "build"
        pipeline = Pipeline(
            "artifact-integrity",
            build_dir=str(build_dir),
            synix_dir=str(tmp_path / ".synix"),
        )
        transcripts = Source("transcripts")
        pipeline.add(transcripts)

        txn = start_build_transaction(pipeline, build_dir, run_id="20260306T120000000002Z")

        with pytest.raises(ValueError, match="does not match its content hash"):
            txn.record_artifact(
                Artifact(
                    label="ep-1",
                    artifact_type="episode",
                    content="Actual content.",
                    artifact_id="sha256:" + "0" * 64,
                ),
                layer_name="transcripts",
                layer_level=0,
                parent_labels=[],
            )

    def test_empty_content_artifacts_are_hashed_and_snapshotted(self, tmp_path):
        """Empty-string content is still valid content and must hash consistently."""
        build_dir = tmp_path / "build"
        pipeline = Pipeline(
            "empty-content",
            build_dir=str(build_dir),
            synix_dir=str(tmp_path / ".synix"),
        )
        transcripts = Source("transcripts")
        pipeline.add(transcripts)

        txn = start_build_transaction(pipeline, build_dir, run_id="20260306T120000000003Z")
        artifact = Artifact(label="empty", artifact_type="note", content="")
        txn.record_artifact(
            artifact,
            layer_name="transcripts",
            layer_level=0,
            parent_labels=[],
        )

        snapshot_info = commit_build_snapshot(txn)
        manifest = ObjectStore(tmp_path / ".synix").get_json(snapshot_info["manifest_oid"])
        stored_artifact = ObjectStore(tmp_path / ".synix").get_json(manifest["artifacts"]["empty"])

        assert artifact.artifact_id == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert stored_artifact["artifact_id"] == artifact.artifact_id
        assert ObjectStore(tmp_path / ".synix").get_bytes(stored_artifact["content_oid"]) == b""

    def test_runs_list_recovers_pending_ref_updates(self, tmp_path):
        """Interrupted ref updates should be replayed before run history is listed."""
        build_dir = tmp_path / "build"
        synix_dir = tmp_path / ".synix"
        object_store = ObjectStore(synix_dir)

        manifest_oid = object_store.put_json(
            {
                "type": "manifest",
                "schema_version": SCHEMA_VERSION,
                "pipeline_name": "recovery-test",
                "pipeline_fingerprint": "sha256:test",
                "artifacts": {},
                "projections": {},
            }
        )
        snapshot_oid = object_store.put_json(
            {
                "type": "snapshot",
                "schema_version": SCHEMA_VERSION,
                "manifest_oid": manifest_oid,
                "parent_snapshot_oids": [],
                "created_at": "2026-03-06T08:20:07Z",
                "pipeline_name": "recovery-test",
                "run_id": "20260306T082007123456Z",
            }
        )

        journal_dir = synix_dir / "ref_journal"
        journal_dir.mkdir(parents=True, exist_ok=True)
        (journal_dir / "pending.json").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "type": "ref_update",
                    "updates": {
                        "refs/runs/20260306T082007123456Z": snapshot_oid,
                        "refs/heads/main": snapshot_oid,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        runs = list_runs(build_dir, synix_dir=synix_dir)
        ref_store = RefStore(synix_dir)

        assert ref_store.read_ref("refs/heads/main") == snapshot_oid
        assert ref_store.read_ref("refs/runs/20260306T082007123456Z") == snapshot_oid
        assert [run_info["ref"] for run_info in runs] == ["refs/runs/20260306T082007123456Z"]
        assert not any(journal_dir.iterdir())

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
        assert manifest["projections"] == {}
        assert (build_dir / "search.db").exists()
        assert (build_dir / "context.md").exists()

        build_store = ArtifactStore(build_dir)
        first_label, first_artifact_oid = next(iter(manifest["artifacts"].items()))
        first_artifact = object_store.get_json(first_artifact_oid)
        assert object_store.get_bytes(first_artifact["content_oid"]).decode("utf-8") == build_store.load_artifact(first_label).content

        assert ref_store.read_head_target() == "refs/heads/main"
        assert ref_store.read_ref("HEAD") == result.snapshot_oid
        assert ref_store.read_ref(result.run_ref) == result.snapshot_oid

    def test_snapshot_scope_is_artifacts_only_for_now(self, tmp_path, source_dir_with_fixtures, mock_llm):
        """Canonical snapshots currently capture artifacts only; projections stay in the compatibility surface."""
        build_dir = tmp_path / "build"
        pipeline = _build_pipeline(build_dir, source_dir_with_fixtures)

        result = run(pipeline, source_dir=str(source_dir_with_fixtures))
        assert result.manifest_oid is not None

        object_store = ObjectStore(tmp_path / ".synix")
        manifest = object_store.get_json(result.manifest_oid)

        assert manifest["projections"] == {}
        assert (build_dir / "search.db").exists()
        assert (build_dir / "context.md").exists()

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
        assert first.manifest_oid == second.manifest_oid
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

    def test_flatfile_projection_outside_build_dir_is_allowed_but_not_snapshotted(
        self,
        tmp_path,
        source_dir_with_fixtures,
        mock_llm,
    ):
        """Projection outputs can still target arbitrary paths until release adapters own projection state."""
        build_dir = tmp_path / "build"
        pipeline = _build_pipeline(build_dir, source_dir_with_fixtures)
        outside_path = tmp_path / "outside.md"
        pipeline.projections = [
            proj for proj in pipeline.projections if not isinstance(proj, FlatFile)
        ] + [
            FlatFile(
                "external-doc",
                sources=[next(layer for layer in pipeline.layers if layer.name == "core")],
                output_path=str(outside_path),
            )
        ]

        result = run(pipeline, source_dir=str(source_dir_with_fixtures))
        assert result.manifest_oid is not None
        manifest = ObjectStore(tmp_path / ".synix").get_json(result.manifest_oid)

        assert manifest["projections"] == {}
        assert outside_path.exists()
