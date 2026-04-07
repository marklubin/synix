"""End-to-end test: pipeline with agent-backed transforms.

Verifies the full path: source files -> agent-backed transforms -> build ->
artifacts with correct provenance (prompt_id from transform, agent_fingerprint
from agent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import synix
from synix import Pipeline, Source
from synix.agents import Group
from synix.core.models import Artifact
from synix.sdk import BuildResult
from synix.transforms import MapSynthesis, ReduceSynthesis

# ---------------------------------------------------------------------------
# Deterministic test agent
# ---------------------------------------------------------------------------


class DeterministicAgent:
    """Test agent that returns predictable content based on input."""

    def __init__(self, prefix: str = "AGENT:", fingerprint: str = "test-agent-v1"):
        self._prefix = prefix
        self._fingerprint = fingerprint

    @property
    def agent_id(self) -> str:
        return f"deterministic-{self._fingerprint}"

    def fingerprint_value(self) -> str:
        return self._fingerprint

    def map(self, artifact: Artifact) -> str:
        return f"{self._prefix} processed {len(artifact.content)} chars"

    def reduce(self, artifacts: list[Artifact]) -> str:
        total_chars = sum(len(a.content) for a in artifacts)
        return f"{self._prefix} reduced {len(artifacts)} artifacts ({total_chars} chars)"

    def group(self, artifacts: list[Artifact]) -> list[Group]:
        return [Group(key="all", artifacts=artifacts, content=f"{self._prefix} grouped")]

    def fold(self, accumulated: str, artifact: Artifact, step: int, total: int) -> str:
        return f"{self._prefix} fold step {step}/{total}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_workspace(tmp_path: Path):
    """Create a workspace with source data for agent pipeline testing."""
    ws_dir = tmp_path / "agent-test"
    project = synix.init(str(ws_dir))

    # Create source files
    sources = ws_dir / "sources"
    sources.mkdir(exist_ok=True)
    (sources / "doc-a.md").write_text("First document about machine learning fundamentals.")
    (sources / "doc-b.md").write_text("Second document about distributed systems design.")
    (sources / "doc-c.md").write_text("Third document about agent memory architecture.")

    return project, ws_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentPipeline:
    """Full pipeline build with agent-backed transforms."""

    def test_build_with_agent_transforms(self, agent_workspace):
        """Source -> MapSynthesis(agent) -> ReduceSynthesis(agent) -> build."""
        project, ws_dir = agent_workspace
        agent = DeterministicAgent(prefix="SUMMARY:", fingerprint="summarizer-v1")

        pipeline = Pipeline("agent-test", source_dir=str(ws_dir / "sources"))

        notes = Source("notes")

        summaries = MapSynthesis(
            "summaries",
            depends_on=[notes],
            prompt="Summarize this document:\\n\\n{artifact}",
            agent=agent,
            artifact_type="summary",
        )

        combined = ReduceSynthesis(
            "combined",
            depends_on=[summaries],
            prompt="Combine these summaries into one report:\\n\\n{artifacts}",
            agent=agent,
            label="final-report",
            artifact_type="report",
        )

        pipeline.add(notes, summaries, combined)

        project.set_pipeline(pipeline)
        result = project.build()

        assert isinstance(result, BuildResult)
        assert result.built > 0, f"Expected artifacts built, got {result}"

        # Verify artifacts via release
        project.release_to("local")
        release = project.release("local")

        all_arts = list(release.artifacts())
        assert len(all_arts) > 0

        # Check summary artifacts have agent_fingerprint
        summary_arts = [a for a in all_arts if a.artifact_type == "summary"]
        assert len(summary_arts) == 3, (
            f"Expected 3 summary artifacts (one per source doc), got {len(summary_arts)}: "
            f"{[a.label for a in summary_arts]}"
        )

        for art in summary_arts:
            assert art.content.startswith("SUMMARY:"), f"Unexpected content: {art.content!r}"
            assert art.agent_fingerprint == "summarizer-v1"
            # prompt_id is the transform's prompt hash, not agent fingerprint
            assert art.prompt_id is not None
            assert art.prompt_id != "summarizer-v1"
            # model_config should be None (agent handled execution)
            assert art.model_config is None

        # Check report artifact
        report_arts = [a for a in all_arts if a.artifact_type == "report"]
        assert len(report_arts) == 1
        assert report_arts[0].content.startswith("SUMMARY:")
        assert report_arts[0].agent_fingerprint == "summarizer-v1"

    def test_cache_invalidation_on_agent_change(self, agent_workspace):
        """Changing agent fingerprint invalidates cache."""
        project, ws_dir = agent_workspace

        agent_v1 = DeterministicAgent(fingerprint="agent-v1")

        pipeline = Pipeline("cache-test", source_dir=str(ws_dir / "sources"))
        notes = Source("notes")
        summaries = MapSynthesis(
            "summaries",
            depends_on=[notes],
            prompt="Summarize:\\n{artifact}",
            agent=agent_v1,
            artifact_type="summary",
        )
        pipeline.add(notes, summaries)

        # First build
        project.set_pipeline(pipeline)
        result1 = project.build()
        assert result1.built >= 3  # 3 source docs -> 3 summaries (+ sources)

        # Second build with same agent -- should be fully cached
        result2 = project.build()
        assert result2.cached > 0, "Transforms should be cached on identical rebuild"

        # Third build with different agent fingerprint -- summaries should rebuild
        agent_v2 = DeterministicAgent(fingerprint="agent-v2")
        pipeline2 = Pipeline("cache-test", source_dir=str(ws_dir / "sources"))
        notes2 = Source("notes")
        summaries2 = MapSynthesis(
            "summaries",
            depends_on=[notes2],
            prompt="Summarize:\\n{artifact}",
            agent=agent_v2,
            artifact_type="summary",
        )
        pipeline2.add(notes2, summaries2)
        project.set_pipeline(pipeline2)
        result3 = project.build()
        # Agent fingerprint changed -> summaries should rebuild
        assert result3.built >= 3

    def test_provenance_separation(self, agent_workspace):
        """prompt_id and agent_fingerprint are distinct provenance dimensions."""
        project, ws_dir = agent_workspace

        agent = DeterministicAgent(fingerprint="my-agent-fp")

        pipeline = Pipeline("prov-test", source_dir=str(ws_dir / "sources"))
        notes = Source("notes")
        summaries = MapSynthesis(
            "summaries",
            depends_on=[notes],
            prompt="Please summarize:\\n{artifact}",
            agent=agent,
            artifact_type="summary",
        )
        pipeline.add(notes, summaries)

        project.set_pipeline(pipeline)
        project.build()
        project.release_to("local")
        release = project.release("local")

        for art in release.artifacts(layer="summaries"):
            # prompt_id is the transform's prompt hash
            assert art.prompt_id is not None
            # agent_fingerprint is the agent's fingerprint
            assert art.agent_fingerprint == "my-agent-fp"
            # They are different values
            assert art.prompt_id != art.agent_fingerprint

    def test_no_agent_backward_compat(self, agent_workspace, mock_llm):
        """Transforms without agent= work exactly as before."""
        project, ws_dir = agent_workspace

        pipeline = Pipeline(
            "compat-test",
            source_dir=str(ws_dir / "sources"),
            llm_config={
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.3,
                "max_tokens": 1024,
            },
        )
        notes = Source("notes")
        summaries = MapSynthesis(
            "summaries",
            depends_on=[notes],
            prompt="Summarize:\\n{artifact}",
            artifact_type="summary",
            # NO agent= parameter
        )
        pipeline.add(notes, summaries)

        project.set_pipeline(pipeline)
        result = project.build()
        assert result.built > 0

        project.release_to("local")
        release = project.release("local")

        for art in release.artifacts(layer="summaries"):
            assert art.agent_fingerprint is None  # no agent used
            assert art.prompt_id is not None  # prompt_id still set
            assert art.model_config is not None  # model_config set (built-in LLM path)
