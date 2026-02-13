# Test Plan

## Philosophy

Tests are not optional. **Every module gets tests as it's built.** No module is "done" until its tests pass. Critical because:
1. Agents write code — tests verify correctness without manual inspection
2. Incremental builds depend on hash comparison being exactly right
3. The demo must work flawlessly on recording day

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures: temp dirs, sample artifacts, mock LLM
├── fixtures/
│   ├── chatgpt_export.json
│   ├── claude_export.json
│   └── sample_pipeline.py
├── unit/
│   ├── test_artifact_store.py
│   ├── test_provenance.py
│   ├── test_dag.py
│   ├── test_config.py
│   ├── test_parsers.py
│   ├── test_transforms.py
│   ├── test_search_index.py
│   ├── test_flat_file.py
│   └── test_cli.py
├── integration/
│   ├── test_pipeline_run.py
│   ├── test_incremental_rebuild.py
│   ├── test_config_change.py
│   └── test_projections.py
└── e2e/
    ├── test_demo_flow.py
    └── test_real_data.py
```

## conftest.py Fixtures

```python
@pytest.fixture
def tmp_build_dir(tmp_path):
    """Clean build directory for each test."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    return build_dir

@pytest.fixture
def sample_artifacts():
    """Pre-built artifacts for testing downstream modules."""
    return [
        Artifact(label="t-001", artifact_type="transcript", content="...", ...),
        Artifact(label="t-002", artifact_type="transcript", content="...", ...),
        Artifact(label="ep-001", artifact_type="episode", content="...", ...),
    ]

@pytest.fixture
def mock_llm(monkeypatch):
    """Mock Anthropic API — returns deterministic responses based on prompt."""
    def mock_create(*args, **kwargs):
        prompt = kwargs.get("messages", [{}])[0].get("content", "")
        if "episode summary" in prompt.lower():
            return MockResponse("This is a summary of the conversation about...")
        elif "monthly rollup" in prompt.lower():
            return MockResponse("In January 2025, the main themes were...")
        elif "core memory" in prompt.lower():
            return MockResponse("## Identity\nMark is a software engineer...")
        return MockResponse("Mock response")
    monkeypatch.setattr("anthropic.Anthropic.messages.create", mock_create)

@pytest.fixture
def sample_pipeline(tmp_build_dir):
    """A complete test pipeline with all layers and projections."""
    # Returns a Pipeline object ready to run with mock data
```

## Unit Tests

**test_artifact_store.py**
- `test_save_and_load_roundtrip` — save artifact, load by ID, content matches
- `test_load_nonexistent_returns_none`
- `test_list_by_layer` — save 5 artifacts across 3 layers, list each correctly
- `test_artifact_id_computed` — artifact_id is SHA256 of content
- `test_manifest_persistence` — save artifacts, reload store, manifest intact
- `test_overwrite_artifact` — save same ID twice, latest wins

**test_provenance.py**
- `test_record_and_retrieve` — record provenance, get_parents returns correct IDs
- `test_chain_walking` — 3-level chain, get_chain returns full path
- `test_chain_multiple_parents` — monthly rollup with 5 episode inputs
- `test_persistence` — reload from disk, data intact

**test_dag.py**
- `test_topological_sort_simple` — 4 layers, correct order
- `test_topological_sort_diamond` — diamond dependency, no duplicate
- `test_cycle_detection` — circular dependency raises error
- `test_rebuild_detection_all_new` — empty build dir, everything needs rebuild
- `test_rebuild_detection_all_cached` — matching hashes, nothing needs rebuild
- `test_rebuild_detection_partial` — change prompt at level 2, levels 2+3 rebuild
- `test_rebuild_cascades` — changing level 1 forces rebuild of 2 and 3

**test_config.py**
- `test_load_pipeline_module` — import pipeline.py, get Pipeline object
- `test_validate_acyclic` / `test_validate_cyclic_rejected`
- `test_validate_missing_dependency` — raises error
- `test_validate_single_root` — exactly one level-0 layer required

**test_parsers.py**
- `test_chatgpt_parse_basic` / `test_chatgpt_metadata` / `test_chatgpt_message_ordering`
- `test_chatgpt_empty_conversation` — gracefully skip
- `test_claude_parse_basic` / `test_claude_metadata`
- `test_mixed_sources` — no collisions

**test_transforms.py** (with mock LLM)
- `test_episode_summary_prompt_construction` — prompt includes transcript
- `test_episode_summary_output_artifact` — valid Artifact with correct type
- `test_monthly_rollup_groups_by_month` — 10 episodes across 3 months, 3 calls
- `test_topical_rollup_clusters`
- `test_core_synthesis_respects_budget`
- `test_prompt_template_loading` / `test_prompt_id_versioning`

**test_search_index.py**
- `test_materialize_and_query` — index 10 artifacts, query returns results
- `test_layer_filtering` / `test_provenance_always_included`
- `test_ranking` / `test_empty_query` / `test_rebuild_replaces_index`

**test_flat_file.py**
- `test_materialize_creates_file` / `test_content_matches_core` / `test_markdown_formatting`

**test_cli.py**
- `test_*_command_exists` — all commands have `--help`
- `test_run_missing_pipeline_errors` / `test_search_no_index_errors`

## Integration Tests

**test_pipeline_run.py** (mock LLM, real everything else)
- `test_full_pipeline_mock_llm` — all layers built, all projections materialized
- `test_artifact_count_matches_expectations`
- `test_all_artifacts_have_provenance`
- `test_search_returns_results_after_run` / `test_context_doc_exists_after_run`

**test_incremental_rebuild.py** (the critical one)
- `test_second_run_all_cached` — run twice, second run builds 0
- `test_new_source_partial_rebuild` — add one conversation, only its chain rebuilds
- `test_prompt_change_cascades` — change episode prompt, episodes + rollups + core rebuild
- `test_cache_metrics_accurate`

**test_config_change.py** (the demo moment)
- `test_swap_monthly_to_topical` — transcripts/episodes cached, topics/core rebuilt
- `test_search_results_differ` / `test_context_doc_differs`

**test_projections.py**
- `test_search_index_reflects_all_layers`
- `test_provenance_chain_depth` — core traces back to transcript
- `test_flat_file_is_ready_to_use`

## E2E Tests

**test_demo_flow.py** — the exact demo sequence, automated:
```python
def test_demo_sequence(real_exports_dir, tmp_build_dir):
    result1 = run("pipeline.py", real_exports_dir, tmp_build_dir)
    assert result1.built > 0 and result1.cached == 0

    results = search("anthropic", tmp_build_dir)
    assert len(results) > 0 and all(r.provenance_chain for r in results)
    assert (tmp_build_dir / "context.md").exists()

    result2 = run("pipeline.py", real_exports_dir, tmp_build_dir)
    assert result2.built == 0 and result2.cached > 0

    result3 = run("pipeline_topical.py", real_exports_dir, tmp_build_dir)
    assert result3.cached > 0 and result3.built > 0

    results2 = search("anthropic", tmp_build_dir)
    assert results2 != results
```

**test_real_data.py** (slow, manual pre-demo validation)

## Test Rules

1. Write tests BEFORE or ALONGSIDE the module, never after
2. All tests must pass before moving to the next phase (`uv run pytest`)
3. Mock the LLM for unit and integration tests — only E2E hits real API
4. Use `tmp_path` for all filesystem tests — no shared state
5. Test failure cases, not just happy path
6. Integration tests (incremental rebuild) are the most important
