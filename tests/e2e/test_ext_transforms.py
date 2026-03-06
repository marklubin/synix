"""E2E tests for generic platform transforms.

Exercises the generic synthesis transforms through the full pipeline runner
with mocked LLM.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synix.cli import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with source bios and a build dir."""
    bios_dir = tmp_path / "sources" / "bios"
    bios_dir.mkdir(parents=True)

    (bios_dir / "alice.md").write_text("# Alice\n\nBackend engineer with 8 years experience in distributed systems.\n")
    (bios_dir / "bob.md").write_text("# Bob\n\nProduct designer with UX expertise and accessibility focus.\n")
    (bios_dir / "carol.md").write_text("# Carol\n\nData scientist specializing in climate modeling and Python.\n")

    brief_dir = tmp_path / "sources" / "brief"
    brief_dir.mkdir(parents=True)
    (brief_dir / "project_brief.md").write_text("# Climate Dashboard\n\nBuild a real-time climate sensor dashboard.\n")

    return {"root": tmp_path, "bios_dir": bios_dir, "brief_dir": brief_dir}


@pytest.fixture
def ext_pipeline_file(workspace):
    """Write a pipeline using ext transforms."""
    path = workspace["root"] / "pipeline.py"
    path.write_text(f"""
from synix import Pipeline, SearchIndex, Source
from synix.transforms import MapSynthesis, ReduceSynthesis, FoldSynthesis

pipeline = Pipeline("ext-test")
pipeline.source_dir = "{workspace["bios_dir"]}"
pipeline.build_dir = "{workspace["root"] / "build"}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

bios = Source("bios")
project_brief = Source("project_brief", dir="{workspace["brief_dir"]}")

work_styles = MapSynthesis(
    "work_styles", depends_on=[bios],
    prompt="Infer work style from this background:\\n\\n{{artifact}}",
    artifact_type="work_style",
)

team_dynamics = ReduceSynthesis(
    "team_dynamics", depends_on=[work_styles],
    prompt="Analyze team dynamics:\\n\\n{{artifacts}}",
    label="team-dynamics",
    artifact_type="team_dynamics",
)

final_report = FoldSynthesis(
    "final_report", depends_on=[team_dynamics, project_brief],
    prompt="Update report:\\n\\nCurrent: {{accumulated}}\\n\\nNew: {{artifact}}",
    initial="No report yet.",
    label="final-report",
    artifact_type="final_report",
)

pipeline.add(bios, project_brief, work_styles, team_dynamics, final_report)
pipeline.add(SearchIndex("search", sources=[work_styles, team_dynamics, final_report], search=["fulltext"]))
""")
    return path


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    """Mock Anthropic API for all E2E tests."""

    def mock_create(**kwargs):
        resp = MagicMock()
        resp.content = [MagicMock(text="Mock LLM response for ext transform testing.")]
        resp.model = "claude-sonnet-4-20250514"
        resp.usage = MagicMock(input_tokens=100, output_tokens=50)
        return resp

    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtPipelineBuild:
    """E2E tests for ext transforms through the pipeline runner."""

    def test_build_produces_artifacts(self, runner, workspace, ext_pipeline_file):
        """Full build with Map, Reduce, Fold produces expected artifacts."""
        result = runner.invoke(main, ["run", str(ext_pipeline_file)])
        assert result.exit_code == 0, f"Build failed: {result.output}"
        assert "Build Summary" in result.output

        # Check manifest
        manifest_path = workspace["root"] / "build" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())

        labels = set(manifest.keys())
        # 3 bios (source) + 1 brief (source) + 3 work_styles (map) + 1 team-dynamics (reduce) + 1 final-report (fold)
        assert "team-dynamics" in labels
        assert "final-report" in labels
        # work_style labels follow the pattern work_styles-t-text-{name}
        ws_labels = [l for l in labels if l.startswith("work_styles-")]
        assert len(ws_labels) == 3

    def test_rebuild_is_cached(self, runner, workspace, ext_pipeline_file):
        """Second build uses cache — no LLM calls."""
        result1 = runner.invoke(main, ["run", str(ext_pipeline_file)])
        assert result1.exit_code == 0

        result2 = runner.invoke(main, ["run", str(ext_pipeline_file)])
        assert result2.exit_code == 0
        assert "cached" in result2.output.lower()

    def test_plan_shows_layers(self, runner, ext_pipeline_file):
        """Plan command shows ext transform layers."""
        result = runner.invoke(main, ["plan", str(ext_pipeline_file)])
        assert result.exit_code == 0
        assert "work_styles" in result.output
        assert "team_dynamics" in result.output
        assert "final_report" in result.output

    def test_search_after_build(self, runner, workspace, ext_pipeline_file):
        """Search works after building with ext transforms."""
        result1 = runner.invoke(main, ["run", str(ext_pipeline_file)])
        assert result1.exit_code == 0

        build_dir = str(workspace["root"] / "build")
        result2 = runner.invoke(main, ["search", "mock", "--build-dir", build_dir])
        assert result2.exit_code == 0

    def test_list_shows_artifacts(self, runner, workspace, ext_pipeline_file):
        """List command shows artifacts from ext transforms."""
        result1 = runner.invoke(main, ["run", str(ext_pipeline_file)])
        assert result1.exit_code == 0

        build_dir = str(workspace["root"] / "build")
        result2 = runner.invoke(main, ["list", "--build-dir", build_dir])
        assert result2.exit_code == 0
        assert "team-dynamics" in result2.output
        assert "final-report" in result2.output


class TestGroupSynthesisE2E:
    """E2E test for GroupSynthesis specifically."""

    def test_group_by_metadata(self, runner, workspace):
        """GroupSynthesis groups artifacts and produces one output per group."""
        path = workspace["root"] / "group_pipeline.py"
        bios_dir = workspace["bios_dir"]

        # Create source files with team metadata embedded in filenames
        team_dir = workspace["root"] / "sources" / "teams"
        team_dir.mkdir(parents=True)
        for name, team in [("alice", "engineering"), ("bob", "design"), ("carol", "engineering")]:
            (team_dir / f"{name}.md").write_text(f"# {name}\n\nTeam: {team}\nSkills: various\n")

        path.write_text(f"""
from synix import Pipeline, Source
from synix.transforms import GroupSynthesis
from synix.core.models import Artifact, Transform
from synix.build.llm_transforms import _get_llm_client, _logged_complete

# Custom transform that adds team metadata from content
class TeamTagger(Transform):
    def execute(self, inputs, config):
        results = []
        for a in inputs:
            team = "unknown"
            for line in a.content.splitlines():
                if line.startswith("Team: "):
                    team = line[6:].strip()
            results.append(Artifact(
                label=a.label,
                artifact_type="tagged",
                content=a.content,
                input_ids=[a.artifact_id],
                metadata={{"team": team}},
            ))
        return results

pipeline = Pipeline("group-test")
pipeline.source_dir = "{team_dir}"
pipeline.build_dir = "{workspace["root"] / "build"}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

sources = Source("sources")
tagged = TeamTagger("tagged", depends_on=[sources])
by_team = GroupSynthesis(
    "team-summaries",
    depends_on=[tagged],
    group_by="team",
    prompt="Summarize team '{{group_key}}':\\n\\n{{artifacts}}",
    artifact_type="team_summary",
)

pipeline.add(sources, tagged, by_team)
""")

        result = runner.invoke(main, ["run", str(path)])
        assert result.exit_code == 0, f"Build failed: {result.output}"

        manifest_path = workspace["root"] / "build" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        labels = set(manifest.keys())

        # Should have team-design and team-engineering groups
        assert "team-design" in labels
        assert "team-engineering" in labels
