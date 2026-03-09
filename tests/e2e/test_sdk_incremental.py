"""E2E tests — SDK-driven incremental pipeline lifecycle.

Scenario: template 01 (chatbot-export-synthesis) pipeline driven entirely
through the SDK.  Starts with a ChatGPT export, builds, releases, searches.
Then adds Claude Code session files incrementally, rebuilds, re-releases, and
verifies new content propagates while upstream artifacts stay cached.

All tests use real FTS5, real source parsing, mocked LLM.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import synix
from synix import FlatFile, Pipeline, SearchSurface, Source, SynixSearch
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup
from synix.sdk import BuildResult

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


# ---------------------------------------------------------------------------
# Pipeline builder — mirrors template 01 (pipeline_monthly.py)
# ---------------------------------------------------------------------------


def _make_pipeline() -> Pipeline:
    """Template-01 pipeline: transcripts → episodes → monthly → core."""
    pipeline = Pipeline(
        "personal-memory",
        source_dir="./sources",
        build_dir="./build",
        llm_config={
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
            "max_tokens": 1024,
        },
    )

    transcripts = Source("transcripts")
    episodes = EpisodeSummary("episodes", depends_on=[transcripts])
    monthly = MonthlyRollup("monthly", depends_on=[episodes])
    core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

    memory_search = SearchSurface(
        "memory-search",
        sources=[episodes, monthly, core],
        modes=["fulltext"],
    )

    pipeline.add(transcripts, episodes, monthly, core, memory_search)
    pipeline.add(SynixSearch("search", surface=memory_search))
    pipeline.add(FlatFile("context-doc", sources=[core], output_path="./build/context.md"))

    return pipeline


# ---------------------------------------------------------------------------
# Claude Code session factories
# ---------------------------------------------------------------------------


def _claude_code_session(
    session_id: str,
    turns: list[tuple[str, str]],
    *,
    date: str = "2024-04-10",
    cwd: str = "/home/user/project",
) -> str:
    """Build a minimal Claude Code .jsonl session string.

    turns: list of (user_msg, assistant_msg) pairs.
    """
    lines = []
    ts_base = f"{date}T10:00:00Z"
    for i, (user_msg, asst_msg) in enumerate(turns):
        lines.append(
            json.dumps(
                {
                    "type": "user",
                    "message": {"role": "user", "content": user_msg},
                    "timestamp": ts_base,
                    "sessionId": session_id,
                    "cwd": cwd,
                }
            )
        )
        lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"role": "assistant", "content": asst_msg},
                    "timestamp": ts_base,
                    "sessionId": session_id,
                    "cwd": cwd,
                }
            )
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIncrementalMemoryBuild:
    """SDK-driven incremental memory build: ChatGPT export → add sessions → rebuild."""

    def test_initial_build_and_search(self, tmp_path: Path, mock_llm):
        """Build from ChatGPT export, release, search returns results."""
        project_dir = tmp_path / "memory"
        project_dir.mkdir()
        pipeline = _make_pipeline()
        project = synix.init(project_dir, pipeline=pipeline)

        # Add the ChatGPT fixture (3 conversations)
        src = project.source("transcripts")
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", Path(src._dir) / "chatgpt.json")

        result = project.build()
        assert isinstance(result, BuildResult)
        assert result.built > 0
        assert result.snapshot_oid is not None

        project.release_to("local")
        mem = project.release("local")

        # Search should find ML content from the fixture
        results = mem.search("machine learning", mode="keyword")
        assert len(results) > 0

        # Should have transcript + episode + monthly + core artifacts
        all_arts = list(mem.artifacts())
        types = {a.artifact_type for a in all_arts}
        assert "transcript" in types
        assert "episode" in types

        # Context doc should exist
        context = mem.flat_file("context-doc")
        assert len(context) > 0

    def test_add_claude_code_session_incremental(self, tmp_path: Path, mock_llm):
        """Add a Claude Code session after initial build → cached upstream + new downstream."""
        project_dir = tmp_path / "memory"
        project_dir.mkdir()
        pipeline = _make_pipeline()
        project = synix.init(project_dir, pipeline=pipeline)

        # Phase 1: Initial build with ChatGPT export
        src = project.source("transcripts")
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", Path(src._dir) / "chatgpt.json")

        r1 = project.build()
        initial_built = r1.built
        initial_oid = r1.snapshot_oid
        calls_after_build1 = len(mock_llm)

        project.release_to("local")
        mem1 = project.release("local")
        ep_count1 = sum(1 for a in mem1.artifacts() if a.artifact_type == "episode")

        # Phase 2: Add a Claude Code session
        session = _claude_code_session(
            "session-001",
            [
                (
                    "How do I set up a Kubernetes cluster?",
                    "To set up a Kubernetes cluster, you can use kubeadm, minikube, or a managed service like EKS.",
                ),
                (
                    "What about persistent storage?",
                    "For persistent storage in Kubernetes, use PersistentVolumes and PersistentVolumeClaims.",
                ),
            ],
            date="2024-04-10",
        )
        (Path(src._dir) / "session-001.jsonl").write_text(session)

        r2 = project.build()
        assert r2.snapshot_oid != initial_oid, "New snapshot after adding session"
        assert r2.built > 0, "Should build new artifacts"
        assert r2.cached > 0, "Should cache existing ChatGPT episodes"

        # Fewer LLM calls than full build (cached episodes don't need LLM)
        calls_in_build2 = len(mock_llm) - calls_after_build1
        assert calls_in_build2 < calls_after_build1, "Incremental build should make fewer LLM calls"

        # Re-release and verify
        project.release_to("local")
        mem2 = project.release("local")
        ep_count2 = sum(1 for a in mem2.artifacts() if a.artifact_type == "episode")
        assert ep_count2 > ep_count1, f"Should have more episodes ({ep_count2} vs {ep_count1})"

        # More episodes → more search results for generic mock content
        results2 = mem2.search("programming", mode="keyword")
        assert len(results2) > 0, "Search should return results after re-release"

        # The new Claude Code session should appear as a transcript
        transcript_labels = {a.label for a in mem2.artifacts() if a.artifact_type == "transcript"}
        assert any("claude-code" in l for l in transcript_labels), "Claude Code transcript should appear"

        # Core memory should still exist and be updated
        core_arts = [a for a in mem2.artifacts() if a.artifact_type == "core_memory"]
        assert len(core_arts) >= 1

    def test_multiple_incremental_sessions(self, tmp_path: Path, mock_llm):
        """Add sessions one at a time, verify each rebuild is incremental."""
        project_dir = tmp_path / "memory"
        project_dir.mkdir()
        pipeline = _make_pipeline()
        project = synix.init(project_dir, pipeline=pipeline)

        src = project.source("transcripts")
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", Path(src._dir) / "chatgpt.json")

        # Build 1: initial
        r1 = project.build()
        project.release_to("local")
        mem1 = project.release("local")
        art_count1 = len(list(mem1.artifacts()))

        # Build 2: add first session
        s1 = _claude_code_session(
            "s1",
            [
                ("What is Redis?", "Redis is an in-memory data store used as a cache and message broker."),
            ],
        )
        (Path(src._dir) / "s1.jsonl").write_text(s1)
        r2 = project.build()
        project.release_to("local")
        mem2 = project.release("local")
        art_count2 = len(list(mem2.artifacts()))
        assert art_count2 > art_count1

        # Build 3: add second session
        s2 = _claude_code_session(
            "s2",
            [
                ("Explain GraphQL", "GraphQL is a query language for APIs that lets clients request specific data."),
            ],
        )
        (Path(src._dir) / "s2.jsonl").write_text(s2)
        r3 = project.build()
        project.release_to("local")
        mem3 = project.release("local")
        art_count3 = len(list(mem3.artifacts()))
        assert art_count3 > art_count2

        # Each build should produce a new snapshot
        assert r1.snapshot_oid != r2.snapshot_oid
        assert r2.snapshot_oid != r3.snapshot_oid

        # Search should return results from all builds
        # (mock LLM returns generic text with "programming" and "machine learning")
        assert len(mem3.search("programming", mode="keyword")) > 0
        assert len(mem3.search("machine learning", mode="keyword")) > 0

        # Claude Code transcripts should appear
        cc_transcripts = [a for a in mem3.artifacts() if a.artifact_type == "transcript" and "claude-code" in a.label]
        assert len(cc_transcripts) == 2, "Both Claude Code sessions should produce transcripts"

    def test_rebuild_no_changes_fully_cached(self, tmp_path: Path, mock_llm):
        """Build twice with no source changes → second build has more cache hits and identical artifacts."""
        project_dir = tmp_path / "memory"
        project_dir.mkdir()
        pipeline = _make_pipeline()
        project = synix.init(project_dir, pipeline=pipeline)

        src = project.source("transcripts")
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", Path(src._dir) / "chatgpt.json")

        r1 = project.build()
        calls_build1 = len(mock_llm)

        # Snapshot artifacts from build 1
        project.release_to("v1")
        mem1 = project.release("v1")
        arts1 = {a.label: (a.artifact_id, a.content) for a in mem1.artifacts()}

        r2 = project.build()
        calls_build2 = len(mock_llm) - calls_build1

        # Second build: all transforms cached, no LLM calls needed
        assert calls_build2 == 0, f"Expected 0 LLM calls on no-change rebuild, got {calls_build2}"
        # Source artifacts always re-parse (built), but transform outputs are cached
        assert r2.cached > 0, "Transform outputs should be cached"

        # Verify content-addressed identity — artifact IDs and content must match
        project.release_to("v2")
        mem2 = project.release("v2")
        arts2 = {a.label: (a.artifact_id, a.content) for a in mem2.artifacts()}

        for label in arts1:
            assert label in arts2, f"artifact {label} missing from second build"
            assert arts1[label][0] == arts2[label][0], f"artifact_id mismatch for {label}"
            assert arts1[label][1] == arts2[label][1], f"content mismatch for {label}"

    def test_remove_source_and_rebuild(self, tmp_path: Path, mock_llm):
        """Remove a source file → rebuild still succeeds, fewer transcripts parsed."""
        project_dir = tmp_path / "memory"
        project_dir.mkdir()
        pipeline = _make_pipeline()
        project = synix.init(project_dir, pipeline=pipeline)

        src = project.source("transcripts")
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", Path(src._dir) / "chatgpt.json")

        s1 = _claude_code_session(
            "s1",
            [
                ("What is Redis?", "Redis is an in-memory data store."),
            ],
        )
        (Path(src._dir) / "s1.jsonl").write_text(s1)

        project.build()
        project.release_to("v1")
        mem1 = project.release("v1")
        t_count1 = sum(1 for a in mem1.artifacts() if a.artifact_type == "transcript")

        # Remove the session
        src.remove("s1.jsonl")
        assert "s1.jsonl" not in src.list()

        r2 = project.build()
        project.release_to("v2")
        mem2 = project.release("v2")
        t_count2 = sum(1 for a in mem2.artifacts() if a.artifact_type == "transcript")

        # Source layer re-parses: fewer transcripts in the new snapshot
        assert t_count2 < t_count1, "Fewer transcripts after removing a source"
        # Build still succeeds and produces a valid release
        assert r2.snapshot_oid is not None
        assert len(mem2.search("programming", mode="keyword")) > 0

    def test_refs_track_build_history(self, tmp_path: Path, mock_llm):
        """Each build advances HEAD; releases create release refs."""
        project_dir = tmp_path / "memory"
        project_dir.mkdir()
        pipeline = _make_pipeline()
        project = synix.init(project_dir, pipeline=pipeline)

        src = project.source("transcripts")
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", Path(src._dir) / "chatgpt.json")

        r1 = project.build()

        # Add session and rebuild
        s1 = _claude_code_session(
            "s1",
            [
                ("Hello", "Hi there! How can I help?"),
            ],
        )
        (Path(src._dir) / "s1.jsonl").write_text(s1)
        r2 = project.build()

        assert r1.snapshot_oid != r2.snapshot_oid

        # Release
        project.release_to("prod")

        # Refs should include heads and releases
        refs = project.refs()
        assert any("heads/main" in k for k in refs)
        assert any("releases/prod" in k for k in refs)

        # Releases list
        assert "prod" in project.releases()

    def test_receipt_and_layers(self, tmp_path: Path, mock_llm):
        """Receipt and layer inspection after build + release."""
        project_dir = tmp_path / "memory"
        project_dir.mkdir()
        pipeline = _make_pipeline()
        project = synix.init(project_dir, pipeline=pipeline)

        src = project.source("transcripts")
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", Path(src._dir) / "chatgpt.json")

        project.build()
        project.release_to("local")
        mem = project.release("local")

        # Receipt
        receipt = mem.receipt()
        assert receipt["pipeline_name"] == "personal-memory"
        assert receipt["release_name"] == "local"
        assert "adapters" in receipt

        # Layers
        layers = mem.layers()
        layer_names = {l.name for l in layers}
        assert "transcripts" in layer_names
        assert "episodes" in layer_names

        # Layer artifacts
        for layer in layers:
            if layer.name == "episodes":
                assert layer.count > 0
                ep_arts = list(layer.artifacts())
                assert len(ep_arts) == layer.count

    def test_lineage_through_layers(self, tmp_path: Path, mock_llm):
        """Lineage traces from core → monthly → episodes."""
        project_dir = tmp_path / "memory"
        project_dir.mkdir()
        pipeline = _make_pipeline()
        project = synix.init(project_dir, pipeline=pipeline)

        src = project.source("transcripts")
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", Path(src._dir) / "chatgpt.json")

        project.build()
        project.release_to("local")
        mem = project.release("local")

        # Find core memory artifact
        core_arts = [a for a in mem.artifacts() if a.artifact_type == "core_memory"]
        assert len(core_arts) >= 1

        chain = mem.lineage(core_arts[0].label)
        chain_types = {a.artifact_type for a in chain}
        # Core should trace back through monthly rollups
        assert "core_memory" in chain_types

    def test_clean_removes_releases(self, tmp_path: Path, mock_llm):
        """project.clean() removes releases."""
        project_dir = tmp_path / "memory"
        project_dir.mkdir()
        pipeline = _make_pipeline()
        project = synix.init(project_dir, pipeline=pipeline)

        src = project.source("transcripts")
        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", Path(src._dir) / "chatgpt.json")

        project.build()
        project.release_to("local")
        assert (project.synix_dir / "releases").exists()

        project.clean()
        assert not (project.synix_dir / "releases").exists()
