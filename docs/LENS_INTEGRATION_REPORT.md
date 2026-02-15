# Synix Integration Report — LENS Datagen Pipeline

**Date**: 2025-02-15
**Pipeline**: lens-datagen (5 layers, 4 validators, 1 search projection)
**Synix version**: 0.10.0 (installed via `uvx`)
**Model**: gpt-4o-mini (OpenAI)

## Build Summary

| Metric | Value |
|--------|-------|
| Runtime | 1518s (~25 min) |
| LLM calls | 75 |
| Tokens | 120,932 |
| Est. cost | $0.73 |
| Artifacts built | 37 |
| Artifacts cached | 1 (spec from prior run) |
| Concurrency | `-j 4` |

Output: 30 signal episodes, 2 distractor episodes, 4 questions, 1 key fact audit, FTS5+semantic search index.

---

## Issue 1: Manifest Format Collision (Critical)

**Symptom**: `TypeError: string indices must be integers, not 'str'` immediately on build start.

**Root cause**: `ArtifactStore` uses `manifest.json` in the build directory to track cached artifacts. Our pre-existing pipeline had already written a `manifest.json` to the same directory with a completely different schema — flat keys like `scope_id`, `spec_version` (strings), not the `{label: {path, artifact_id, layer, level}}` dicts synix expects. When synix iterated the manifest entries and called `entry["layer"]`, it was indexing into a string.

**Location**: `synix/build/artifacts.py:93` — `if entry["layer"] == layer:` assumes every manifest value is a dict.

**Fix applied**: Backed up old `generated/` directory and started clean. Also discovered the reverse problem: our release step was writing its own `manifest.json` into the build directory, clobbering synix's manifest and breaking caching for subsequent runs. Fixed by renaming release output to `release_manifest.json`.

**Recommendation**: Either:
- Validate manifest entries on load (skip non-dict entries gracefully)
- Use a namespaced filename like `.synix_manifest.json` to avoid collisions with user files in the build directory
- Add a clear error message: "manifest.json has unexpected format — was the build directory used by another tool?"

---

## Issue 2: `max_tokens` vs `max_completion_tokens` (OpenAI API)

**Symptom**: `400 Bad Request — Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.`

**Root cause**: `LLMClient._complete_openai()` (`llm_client.py:158-163`) unconditionally passes `max_tokens` to the OpenAI chat completions API. Newer OpenAI models (o1, o3, gpt-5.x) require `max_completion_tokens` instead. The `max_tokens` parameter was deprecated for these model families starting with o1 in late 2024.

**Location**: `synix/build/llm_client.py:158`

```python
response = self._client.chat.completions.create(
    model=self.config.model,
    max_tokens=max_tokens,  # fails for o1, o3, gpt-5.x models
    temperature=temperature,
    messages=messages,
)
```

**Fix applied**: Changed our pipeline default model from `gpt-5.2` to `gpt-4o-mini` which still accepts `max_tokens`.

**Recommendation**: Detect the model family and use the appropriate parameter:

```python
# Models that require max_completion_tokens instead of max_tokens
_NEW_PARAM_MODELS = {"o1", "o3", "gpt-5"}

def _needs_new_token_param(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _NEW_PARAM_MODELS)

# In _complete_openai:
token_param = "max_completion_tokens" if _needs_new_token_param(self.config.model) else "max_tokens"
kwargs = {token_param: max_tokens, ...}
```

Or use the openai SDK's built-in handling if available in newer versions.

---

## Issue 3: No Traceback in Build Failures

**Symptom**: Build errors show only the exception message, not the full stack trace.

```
  signal_episodes (level 1)  0.0s
Pipeline failed: string indices must be integers, not 'str'
```

No file path, no line number, no traceback. With a message like "string indices must be integers" this is extremely difficult to debug without knowing *where* the error occurred.

**Workaround**: Had to write a standalone debug script that imported synix internals and wrapped `runner.run()` in a try/except to get the actual traceback:

```
File "synix/build/runner.py", line 148, in run
    if layer.level > 0 and _layer_fully_cached(layer, inputs, store, transform_fp):
File "synix/build/artifacts.py", line 93, in list_artifacts
    if entry["layer"] == layer:
TypeError: string indices must be integers, not 'str'
```

**Location**: `synix/cli/build_commands.py` — the exception handler that prints "Pipeline failed: ..."

**Recommendation**: At `-vv` verbosity, print the full traceback. At `-v`, include at minimum the exception type and the innermost frame (file:line). Example:

```
Pipeline failed: TypeError at synix/build/artifacts.py:93
  string indices must be integers, not 'str'
```

---

## Issue 4: Build Progress Not Captured to File/Pipe

**Symptom**: When running the build in the background and tailing the output file, only the initial header appeared — no per-layer progress updates until completion.

**Root cause**: Synix uses Rich terminal rendering with live-updating spinners and progress bars. These use terminal control sequences (cursor movement, line clearing) that don't serialize well to files or non-TTY outputs.

**Impact**: Makes it difficult to monitor long builds from scripts, CI/CD pipelines, or background processes.

**Recommendation**: Add a `--plain` or `--log-format=text` flag that outputs simple line-by-line progress without terminal control sequences:

```
[10:13:54] spec (level 0): 1 cached (0.0s)
[10:13:55] signal_episodes (level 1): starting...
[10:14:02] signal_episodes: LLM call signal-phase-baseline (gpt-4o-mini)
[10:14:18] signal_episodes: LLM call signal-phase-baseline complete (16.8s, 1011 tokens)
...
```

This would make synix much more CI-friendly. The structured JSON logs in `logs/*.jsonl` are good but arrive after the fact — real-time line output is needed for monitoring.

---

## Issue 5: No `--version` Flag

**Minor**: `synix --version` returns `Error: No such option: --version`.

**Recommendation**: Add `--version` to the CLI root command. Essential for reproducibility (logging which synix version produced a build) and debugging.

---

## What Worked Well

1. **DAG resolution and parallel execution**: Synix correctly resolved the 5-layer build order and ran Level 1 layers (signal + distractor generation) concurrently. The `split()` mechanism for per-theme distractor parallelism also worked as expected.

2. **Content-addressed caching**: The spec layer was cached from the first (failed) run attempt and correctly reused on the second run. Zero wasted LLM calls for already-computed artifacts.

3. **Provenance tracking**: `provenance.json` was produced automatically with parent-child artifact relationships. No additional code needed.

4. **Search projection**: FTS5 + semantic index (fastembed `BAAI/bge-small-en-v1.5`) built progressively as layers completed. The progressive materialization (index available after signal episodes, updated after distractors) is a nice touch for downstream transforms that could query mid-build.

5. **Structured logging**: Per-LLM-call timing and token counts in `logs/*.jsonl` — excellent for cost tracking and performance analysis. Timestamps, layer names, and artifact descriptions all present.

6. **Transform/validator registry**: `@register_transform` and `@register_validator` decorators worked cleanly. Custom transforms loaded via `pipeline.py`'s `sys.path.insert` + bare imports pattern worked without issues.

7. **Shared LLM client**: The runner's `_shared_llm_client` pattern (create one client, share across concurrent workers) avoids per-thread connection overhead. Good design.

8. **Fingerprint-based cache invalidation**: The `compute_fingerprint` system that hashes source code + prompt + config + model means cache is automatically invalidated when transform logic changes. Smart approach.

---

## Pipeline-Level Observations (Not Synix Issues)

### Low Distractor Yield
Only 2 of 90 requested distractors passed the word-overlap similarity filter (`max_similarity: 0.3`). This is a prompt/model quality issue combined with an overly aggressive similarity metric — not a synix issue. Needs methodology rework (separate discussion).

### Sequential Episode Expansion
The `episode-expand` calls (word count enforcement) run sequentially within each transform's `execute()` method. Signal expansion made ~30 calls at ~17s each, accounting for most of the 830s signal layer time. These are independent calls that could be parallelized within the transform using a thread pool.

### Parallelism Configuration
- Build ran at `-j 4`. Higher parallelism (`-j 8` or `-j 12`) would help for the distractor split (3 themes) and episode expansion.
- Validate step should also run with higher parallelism — the 2 LLM-powered validators (contamination check, naive baseline) are independent.
- Release step is I/O-only, parallelism irrelevant.
