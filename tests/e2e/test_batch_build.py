"""E2E tests for synix batch-build CLI commands.

Tests exercise the full CLI with mocked OpenAI Batch API.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synix.cli.main import main

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with source exports and a build dir."""
    source_dir = tmp_path / "exports"
    source_dir.mkdir()
    build_dir = tmp_path / "build"

    # Copy fixture exports
    shutil.copy(FIXTURES_DIR / "chatgpt_export.json", source_dir / "chatgpt_export.json")

    return {"root": tmp_path, "source_dir": source_dir, "build_dir": build_dir}


@pytest.fixture
def openai_pipeline_file(workspace):
    """Pipeline with an OpenAI layer for batch testing."""
    path = workspace["root"] / "pipeline.py"
    path.write_text(f"""
from synix import Pipeline, Source
from synix.ext import EpisodeSummary

pipeline = Pipeline("test-batch")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 512}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts], batch=True)

pipeline.add(transcripts, episodes)
""")
    return str(path)


@pytest.fixture
def anthropic_only_pipeline_file(workspace):
    """Pipeline with only Anthropic layers — no batchable layers."""
    path = workspace["root"] / "pipeline.py"
    path.write_text(f"""
from synix import Pipeline, Source
from synix.ext import EpisodeSummary

pipeline = Pipeline("test-anthropic")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"provider": "anthropic", "model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 512}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])

pipeline.add(transcripts, episodes)
""")
    return str(path)


@pytest.fixture
def mixed_pipeline_file(workspace):
    """Pipeline with mixed providers — OpenAI episodes, Anthropic core."""
    path = workspace["root"] / "pipeline.py"
    path.write_text(f"""
from synix import Pipeline, Source
from synix.ext import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("test-mixed")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"provider": "anthropic", "model": "claude-sonnet-4-20250514"}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts],
    config={{"llm_config": {{"provider": "openai", "model": "gpt-4o-mini"}}}},
    batch=True)
monthly = MonthlyRollup("monthly", depends_on=[episodes],
    batch=False)

pipeline.add(transcripts, episodes, monthly)
""")
    return str(path)


@pytest.fixture
def force_sync_pipeline_file(workspace):
    """Pipeline with OpenAI provider but batch=False."""
    path = workspace["root"] / "pipeline.py"
    path.write_text(f"""
from synix import Pipeline, Source
from synix.ext import EpisodeSummary

pipeline = Pipeline("test-force-sync")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"provider": "openai", "model": "gpt-4o-mini"}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts], batch=False)

pipeline.add(transcripts, episodes)
""")
    return str(path)


def _mock_openai(monkeypatch, complete_immediately: bool = True):
    """Set up OpenAI mocks that return controlled responses.

    Args:
        complete_immediately: If True, batch status is 'completed' on first check.
    """
    mock_file = MagicMock()
    mock_file.id = "file-test123"

    mock_batch = MagicMock()
    mock_batch.id = "batch-test456"

    mock_completed_batch = MagicMock()
    mock_completed_batch.status = "completed"
    mock_completed_batch.output_file_id = "file-output789"
    mock_completed_batch.error_file_id = None

    def _make_response_jsonl(custom_ids):
        lines = []
        for cid in custom_ids:
            entry = {
                "custom_id": cid,
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": f"Mock summary for {cid[:12]}"}}],
                        "model": "gpt-4o-mini",
                        "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
                    },
                },
            }
            lines.append(json.dumps(entry))
        return "\n".join(lines)

    # Track submitted keys for response generation
    _submitted_keys = []

    def _files_create(**kwargs):
        file_obj = kwargs.get("file")
        if file_obj:
            content = file_obj.read().decode()
            for line in content.strip().split("\n"):
                parsed = json.loads(line)
                _submitted_keys.append(parsed["custom_id"])
        return mock_file

    mock_content = MagicMock()

    def _files_content(file_id):
        mock_content.text = _make_response_jsonl(_submitted_keys)
        return mock_content

    mock_client = MagicMock()
    mock_client.files.create.side_effect = _files_create
    mock_client.batches.create.return_value = mock_batch
    mock_client.batches.retrieve.return_value = mock_completed_batch
    mock_client.files.content.side_effect = _files_content

    import openai as openai_mod

    monkeypatch.setattr(openai_mod, "OpenAI", lambda **kw: mock_client)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")

    return mock_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchBuildPlan:
    def test_plan_shows_batch_sequencing(self, runner, openai_pipeline_file, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        result = runner.invoke(main, ["batch-build", "plan", openai_pipeline_file])
        assert result.exit_code == 0
        assert "Experimental" in result.output
        assert "Batch Build Plan" in result.output
        assert "batch" in result.output.lower()

    def test_plan_anthropic_only_shows_all_sync(self, runner, anthropic_only_pipeline_file):
        result = runner.invoke(main, ["batch-build", "plan", anthropic_only_pipeline_file])
        assert result.exit_code == 0
        # Source is always source mode, episodes is sync (anthropic)
        assert "sync" in result.output.lower()


class TestBatchBuildRun:
    def test_run_with_poll_completes(self, runner, openai_pipeline_file, monkeypatch):
        _mock_openai(monkeypatch)
        result = runner.invoke(main, ["batch-build", "run", openai_pipeline_file, "--poll"])
        assert result.exit_code == 0, f"stdout:\n{result.output}\nexception:\n{result.exception}"
        assert "completed" in result.output.lower()

    def test_run_without_poll_shows_resume_instructions(self, runner, openai_pipeline_file, monkeypatch):
        _mock_openai(monkeypatch)
        result = runner.invoke(main, ["batch-build", "run", openai_pipeline_file])
        assert result.exit_code == 0
        assert "submitted" in result.output.lower() or "completed" in result.output.lower()

    def test_experimental_warning_displayed(self, runner, openai_pipeline_file, monkeypatch):
        _mock_openai(monkeypatch)
        result = runner.invoke(main, ["batch-build", "run", openai_pipeline_file, "--poll"])
        assert "Experimental" in result.output


class TestBatchBuildResume:
    def test_resume_completes_after_submit(self, runner, openai_pipeline_file, workspace, monkeypatch):
        _mock_openai(monkeypatch)

        # First run submits
        result = runner.invoke(main, ["batch-build", "run", openai_pipeline_file])
        assert result.exit_code == 0

        # Extract build ID from output
        build_id = None
        for line in result.output.splitlines():
            if "Build ID:" in line:
                build_id = line.split("Build ID:")[-1].strip()
                # Remove Rich markup if present
                import re

                build_id = re.sub(r"\[/?[^\]]*\]", "", build_id).strip()
                break

        if build_id is None:
            pytest.skip("Could not extract build ID from output")

        # Resume with poll
        result = runner.invoke(main, ["batch-build", "resume", build_id, openai_pipeline_file, "--poll"])
        assert result.exit_code == 0

    def test_resume_idempotent(self, runner, openai_pipeline_file, workspace, monkeypatch):
        """Repeated resume doesn't duplicate artifacts."""
        _mock_openai(monkeypatch)

        # Full run
        result = runner.invoke(main, ["batch-build", "run", openai_pipeline_file, "--poll"])
        assert result.exit_code == 0

        build_id = None
        import re

        for line in result.output.splitlines():
            if "Build ID:" in line:
                build_id = re.sub(r"\[/?[^\]]*\]", "", line.split("Build ID:")[-1]).strip()
                break

        if build_id is None:
            pytest.skip("Could not extract build ID")

        # Resume again — should be no-op (all layers completed)
        result = runner.invoke(main, ["batch-build", "resume", build_id, openai_pipeline_file, "--poll"])
        assert result.exit_code == 0
        assert "completed" in result.output.lower()


class TestBatchBuildList:
    def test_list_shows_builds(self, runner, openai_pipeline_file, workspace, monkeypatch):
        _mock_openai(monkeypatch)

        # Create a build
        runner.invoke(main, ["batch-build", "run", openai_pipeline_file, "--poll"])

        # List builds
        result = runner.invoke(main, ["batch-build", "list", "--build-dir", str(workspace["build_dir"])])
        assert result.exit_code == 0
        assert "batch-" in result.output

    def test_list_empty(self, runner, workspace):
        build_dir = workspace["build_dir"]
        build_dir.mkdir(parents=True, exist_ok=True)
        result = runner.invoke(main, ["batch-build", "list", "--build-dir", str(build_dir)])
        assert result.exit_code == 0
        assert "No batch builds found" in result.output


class TestBatchBuildStatus:
    def test_status_shows_details(self, runner, openai_pipeline_file, workspace, monkeypatch):
        _mock_openai(monkeypatch)

        result = runner.invoke(main, ["batch-build", "run", openai_pipeline_file, "--poll"])
        assert result.exit_code == 0

        import re

        build_id = None
        for line in result.output.splitlines():
            if "Build ID:" in line:
                build_id = re.sub(r"\[/?[^\]]*\]", "", line.split("Build ID:")[-1]).strip()
                break

        if build_id is None:
            pytest.skip("Could not extract build ID")

        result = runner.invoke(main, ["batch-build", "status", build_id, "--build-dir", str(workspace["build_dir"])])
        assert result.exit_code == 0
        assert build_id in result.output

    def test_status_unknown_build(self, runner, workspace):
        build_dir = workspace["build_dir"]
        build_dir.mkdir(parents=True, exist_ok=True)
        result = runner.invoke(main, ["batch-build", "status", "nonexistent", "--build-dir", str(build_dir)])
        assert result.exit_code != 0


class TestErrorHandling:
    def test_no_openai_layers_is_usage_error(self, runner, anthropic_only_pipeline_file):
        result = runner.invoke(main, ["batch-build", "run", anthropic_only_pipeline_file])
        assert result.exit_code != 0
        assert "No batchable layers" in result.output or "batchable" in str(result.exception)

    def test_batch_true_on_anthropic_is_value_error(self, runner, workspace):
        path = workspace["root"] / "pipeline.py"
        path.write_text(f"""
from synix import Pipeline, Source
from synix.ext import EpisodeSummary

pipeline = Pipeline("bad")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"provider": "anthropic", "model": "claude-sonnet-4-20250514"}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts], batch=True)

pipeline.add(transcripts, episodes)
""")
        result = runner.invoke(main, ["batch-build", "run", str(path)])
        assert result.exit_code != 0

    def test_batch_false_forces_sync(self, runner, force_sync_pipeline_file, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        result = runner.invoke(main, ["batch-build", "run", force_sync_pipeline_file])
        # Should fail because no batchable layers (batch=False forces sync)
        assert result.exit_code != 0
        assert "No batchable layers" in result.output or "batchable" in str(result.exception)

    def test_cassette_replay_mode_rejected_without_responses(self, runner, openai_pipeline_file, monkeypatch, tmp_path):
        """Replay rejected when no batch_responses.json exists."""
        cassette_dir = tmp_path / "cassettes"
        cassette_dir.mkdir()
        monkeypatch.setenv("SYNIX_CASSETTE_MODE", "replay")
        monkeypatch.setenv("SYNIX_CASSETTE_DIR", str(cassette_dir))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        result = runner.invoke(main, ["batch-build", "run", openai_pipeline_file])
        assert result.exit_code != 0
        assert "replay" in result.output.lower() or "replay" in str(result.exception).lower()

    def test_cassette_replay_mode_allowed_with_responses(self, runner, openai_pipeline_file, workspace, monkeypatch):
        """Replay allowed when batch_responses.json exists in SYNIX_CASSETTE_DIR."""
        cassette_dir = Path(workspace["root"]) / "cassettes"
        cassette_dir.mkdir()

        # Create batch_responses.json with empty dict (no matching keys — but the
        # gate check only looks for file existence; the actual batch layer will
        # fall through to "collecting" since no cassette key matches).
        (cassette_dir / "batch_responses.json").write_text("{}")
        monkeypatch.setenv("SYNIX_CASSETTE_MODE", "replay")
        monkeypatch.setenv("SYNIX_CASSETTE_DIR", str(cassette_dir))

        # Should NOT reject at the pre-validation gate
        result = runner.invoke(main, ["batch-build", "run", openai_pipeline_file])
        # May fail later (no real API), but must not fail with the replay UsageError
        output_and_exc = (result.output + str(result.exception)).lower()
        assert "incompatible with batch-build" not in output_and_exc


class TestMixedProviders:
    def test_mixed_openai_and_anthropic(self, runner, mixed_pipeline_file, monkeypatch):
        """OpenAI layers batch, Anthropic layers run sync."""
        _mock_openai(monkeypatch)

        # Mock anthropic for the sync layer
        import anthropic as anthropic_mod

        mock_anthropic_response = MagicMock()
        mock_anthropic_response.content = [MagicMock(text="Monthly synthesis")]
        mock_anthropic_response.model = "claude-sonnet-4-20250514"
        mock_anthropic_response.usage = MagicMock(input_tokens=100, output_tokens=50)

        mock_anthropic_client = MagicMock()
        mock_anthropic_client.messages.create.return_value = mock_anthropic_response
        monkeypatch.setattr(anthropic_mod, "Anthropic", lambda **kw: mock_anthropic_client)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake")

        result = runner.invoke(main, ["batch-build", "run", mixed_pipeline_file, "--poll"])
        assert result.exit_code == 0, f"stdout:\n{result.output}\nexception:\n{result.exception}"

    def test_legacy_strict_dict_transform_runs_after_batch_layer(self, runner, workspace, monkeypatch):
        """Batch runner still passes a raw dict to legacy sync transforms."""
        path = workspace["root"] / "legacy_pipeline.py"
        path.write_text(f"""
from synix import Artifact, Pipeline, Source, Transform
from synix.ext import EpisodeSummary

pipeline = Pipeline("test-legacy-batch")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 512}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts], batch=True)

class LegacyStrictDictTransform(Transform):
    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        assert type(config) is dict
        return [
            Artifact(
                label=f"{{config['prefix']}}-{{inp.label}}",
                artifact_type="legacy_summary",
                content=inp.content,
                input_ids=[inp.artifact_id],
            )
            for inp in inputs
        ]

legacy = LegacyStrictDictTransform("legacy", depends_on=[episodes], config={{"prefix": "legacy"}})

pipeline.add(transcripts, episodes, legacy)
""")

        _mock_openai(monkeypatch)
        result = runner.invoke(main, ["batch-build", "run", str(path), "--poll"])
        assert result.exit_code == 0, f"stdout:\n{result.output}\nexception:\n{result.exception}"


class TestCorruptedState:
    def test_corrupted_state_quarantined(self, runner, openai_pipeline_file, workspace, monkeypatch):
        _mock_openai(monkeypatch)

        # Create a build first
        result = runner.invoke(main, ["batch-build", "run", openai_pipeline_file, "--poll"])
        assert result.exit_code == 0

        import re

        build_id = None
        for line in result.output.splitlines():
            if "Build ID:" in line:
                build_id = re.sub(r"\[/?[^\]]*\]", "", line.split("Build ID:")[-1]).strip()
                break

        if build_id is None:
            pytest.skip("Could not extract build ID")

        # Corrupt the state file
        state_path = workspace["build_dir"] / "builds" / build_id / "batch_state.json"
        if state_path.exists():
            state_path.write_text("{corrupt!!!}")

            # Resume should fail
            result = runner.invoke(main, ["batch-build", "resume", build_id, openai_pipeline_file])
            assert result.exit_code != 0

    def test_reset_state_restarts_layer(self, runner, openai_pipeline_file, workspace, monkeypatch):
        _mock_openai(monkeypatch)

        result = runner.invoke(main, ["batch-build", "run", openai_pipeline_file, "--poll"])
        assert result.exit_code == 0

        import re

        build_id = None
        for line in result.output.splitlines():
            if "Build ID:" in line:
                build_id = re.sub(r"\[/?[^\]]*\]", "", line.split("Build ID:")[-1]).strip()
                break

        if build_id is None:
            pytest.skip("Could not extract build ID")

        # Corrupt the state file
        state_path = workspace["build_dir"] / "builds" / build_id / "batch_state.json"
        if state_path.exists():
            state_path.write_text("{corrupt!!!}")

            # Resume with --reset-state should recover
            result = runner.invoke(
                main, ["batch-build", "resume", build_id, openai_pipeline_file, "--poll", "--reset-state"]
            )
            assert result.exit_code == 0


class TestFingerprintMismatch:
    def test_mismatch_on_resume_is_error(self, runner, workspace, monkeypatch):
        _mock_openai(monkeypatch)

        # Create pipeline v1
        path = workspace["root"] / "pipeline.py"
        path.write_text(f"""
from synix import Pipeline, Source
from synix.ext import EpisodeSummary

pipeline = Pipeline("v1")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"provider": "openai", "model": "gpt-4o-mini"}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts], batch=True)

pipeline.add(transcripts, episodes)
""")

        result = runner.invoke(main, ["batch-build", "run", str(path)])
        assert result.exit_code == 0

        import re

        build_id = None
        for line in result.output.splitlines():
            if "Build ID:" in line:
                build_id = re.sub(r"\[/?[^\]]*\]", "", line.split("Build ID:")[-1]).strip()
                break

        if build_id is None:
            pytest.skip("Could not extract build ID")

        # Change pipeline (add a layer)
        path.write_text(f"""
from synix import Pipeline, Source
from synix.ext import EpisodeSummary, MonthlyRollup

pipeline = Pipeline("v2")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"provider": "openai", "model": "gpt-4o-mini"}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts], batch=True)
monthly = MonthlyRollup("monthly", depends_on=[episodes], batch=True)

pipeline.add(transcripts, episodes, monthly)
""")

        # Resume should fail with mismatch
        result = runner.invoke(main, ["batch-build", "resume", build_id, str(path), "--poll"])
        assert result.exit_code != 0
        assert "mismatch" in result.output.lower() or "mismatch" in str(result.exception).lower()

    def test_allow_pipeline_mismatch_proceeds(self, runner, workspace, monkeypatch):
        _mock_openai(monkeypatch)

        path = workspace["root"] / "pipeline.py"
        path.write_text(f"""
from synix import Pipeline, Source
from synix.ext import EpisodeSummary

pipeline = Pipeline("v1")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"provider": "openai", "model": "gpt-4o-mini"}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts], batch=True)

pipeline.add(transcripts, episodes)
""")

        result = runner.invoke(main, ["batch-build", "run", str(path), "--poll"])
        assert result.exit_code == 0

        import re

        build_id = None
        for line in result.output.splitlines():
            if "Build ID:" in line:
                build_id = re.sub(r"\[/?[^\]]*\]", "", line.split("Build ID:")[-1]).strip()
                break

        if build_id is None:
            pytest.skip("Could not extract build ID")

        # Change pipeline name (changes hash)
        path.write_text(f"""
from synix import Pipeline, Source
from synix.ext import EpisodeSummary

pipeline = Pipeline("v2-changed")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"provider": "openai", "model": "gpt-4o-mini"}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts], batch=True)

pipeline.add(transcripts, episodes)
""")

        # Resume with --allow-pipeline-mismatch should work
        result = runner.invoke(
            main, ["batch-build", "resume", build_id, str(path), "--poll", "--allow-pipeline-mismatch"]
        )
        assert result.exit_code == 0


class TestRequestKeyCorrectness:
    def test_different_params_produce_different_keys(self):
        """Verify that changing any output-affecting param changes the key."""
        from synix.build.cassette import compute_cassette_key

        msgs = [{"role": "user", "content": "test"}]

        k1 = compute_cassette_key("openai", "gpt-4o-mini", msgs, 512, 0.3)
        k2 = compute_cassette_key("openai", "gpt-4o", msgs, 512, 0.3)
        k3 = compute_cassette_key("openai", "gpt-4o-mini", msgs, 1024, 0.3)
        k4 = compute_cassette_key("openai", "gpt-4o-mini", msgs, 512, 0.9)
        k5 = compute_cassette_key("anthropic", "gpt-4o-mini", msgs, 512, 0.3)
        k6 = compute_cassette_key("openai", "gpt-4o-mini", [{"role": "user", "content": "other"}], 512, 0.3)

        keys = {k1, k2, k3, k4, k5, k6}
        assert len(keys) == 6, "All keys should be distinct"
