# Synix SDK

Programmatic access to synix — "boto3 for synix". Build pipelines, manage sources, release, search, and inspect memory programmatically without shelling out to the CLI.

## Quick Start

```python
import synix

# Open an existing project
project = synix.open("~/my-project")
mem = project.release("local")

# Search memory (hybrid = keyword + semantic)
results = mem.search("testing strategies", mode="hybrid", limit=5)
for r in results:
    print(f"  [{r.layer}] {r.label}: {r.score:.2f}")

# Read a specific artifact
core = mem.artifact("core-memory")
print(core.content)
```

## Project Lifecycle

### 1. Initialize a Project

```python
import synix
from synix import Pipeline, Source, Transform, SearchSurface, SynixSearch, FlatFile

# Define your pipeline
pipeline = Pipeline("my-memory", source_dir="./sources")
exports = Source("exports")
pipeline.add(exports)

# Create the project
project = synix.init("~/my-project", pipeline=pipeline)
```

This creates the `.synix/` directory structure and (if a pipeline is provided) creates source directories.

### 2. Open an Existing Project

```python
project = synix.open_project("~/my-project")  # walks upward to find .synix/
project = synix.open_project()                 # uses current directory
```

> **Note**: `synix.open()` is a deprecated alias for `synix.open_project()`. Prefer `open_project` to avoid shadowing Python's builtin `open`.

### 3. Manage Sources

```python
project.source("exports").add("conversation.json")
project.source("exports").add_text("raw text content", "note.txt")
project.source("exports").list()       # ["conversation.json", "note.txt"]
project.source("exports").remove("note.txt")
```

### 4. Build

```python
result = project.build(pipeline=pipeline)
print(f"Built: {result.built}, Cached: {result.cached}")
print(f"Snapshot: {result.snapshot_oid}")

# Dry run — see what would be built
plan = project.build(pipeline=pipeline, dry_run=True)
print(f"Would build: {plan.built}, Already cached: {plan.cached}")
```

### 5. Release

```python
receipt = project.release_to("local")  # materialize projections
print(receipt["adapters"])
```

### 6. Search & Inspect

```python
mem = project.release("local")
results = mem.search("query", mode="hybrid", limit=10)

# Or use a scratch release from HEAD
with project.release("HEAD") as scratch:
    results = scratch.search("query")
```

## Pipeline Interop

The SDK uses the same `Pipeline`, `Source`, `Transform`, `SearchSurface`, `SynixSearch`, and `FlatFile` classes as pipeline definition files. Define your pipeline in Python and pass it directly:

```python
import synix

pipeline = synix.Pipeline("team-memory")
exports = synix.Source("exports")
# ... define transforms ...
pipeline.add(exports, ...)

project = synix.init("./workspace", pipeline=pipeline)
project.source("exports").add("data.json")
project.build(pipeline=pipeline)
project.release_to("local")
```

Or load from a file:

```python
project = synix.open("./workspace")
project.load_pipeline("pipeline.py")
project.build()
```

Pipeline resolution order for `build()`:
1. Explicit `pipeline` argument (highest priority)
2. Previously set via `set_pipeline()` or prior `build()` call
3. Auto-detect `pipeline.py` in project root

## Source Management

```python
src = project.source("exports")

# Add files
src.add("path/to/file.json")           # copies file into source dir
src.add_text("content", "label.txt")    # creates text file

# Inspect
src.list()                               # list file names
src.remove("old-file.json")             # remove a file
src.clear()                              # remove all files
```

Source directory resolution:
- If `Source(dir="./custom/")` is set: resolves relative to project root
- Otherwise: `project_root / pipeline.source_dir / source_name`

## Data Plane

### Artifacts

```python
release = project.release("local")

# Single artifact
art = release.artifact("ep-1")
print(art.label, art.artifact_type, art.layer, art.layer_level)
print(art.content)
print(art.provenance)  # BFS-walked parent labels
print(art.metadata)

# All artifacts
for art in release.artifacts():
    print(art.label)

# Filtered by layer
for art in release.artifacts(layer="episodes"):
    print(art.label)
```

### Layers

```python
for layer in release.layers():
    print(f"{layer.name}: level={layer.level}, count={layer.count}")
    for art in layer.artifacts():
        print(f"  - {art.label}")
```

### Lineage

```python
chain = release.lineage("core-1")
for art in chain:
    print(f"  {art.label} ({art.artifact_type})")
# Output: core-1 (core_memory) → ep-1 (episode) → ep-2 (episode)
```

### Flat Files

```python
content = release.flat_file("context-doc")
path = release.flat_file_path("context-doc")
```

### Receipt

```python
receipt = release.receipt()
print(receipt["released_at"])
print(receipt["adapters"]["memory-search"]["artifacts_applied"])
```

## Search Guide

### Modes

| Mode | Description |
|------|-------------|
| `keyword` | FTS5 full-text search only |
| `semantic` | Embedding cosine-similarity only |
| `hybrid` | Both keyword + semantic fused via RRF (default) |
| `layered` | Like hybrid but boosts higher-level layers |

### Surface Filtering

If a release has multiple search projections, specify which one:

```python
results = release.search("query", surface="memory-search")

# Or bind a handle for repeated queries
handle = release.index("memory-search")
results = handle.search("query")
results = handle.search("another query", layers=["episodes"])
```

If only one search projection exists, it's auto-detected.

### Layer Filtering

```python
results = release.search("query", layers=["episodes", "core"])
```

### Embedding Config

Embeddings are configured in the pipeline's `SearchSurface`:

```python
surface = SearchSurface(
    "memory-surface",
    sources=[episodes, core],
    modes=["fulltext", "semantic"],
    embedding_config={"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
)
```

### Fail-Closed Behavior

Embeddings are **fail-closed**. If `embedding_config` is declared in a pipeline's search surface:

- **Release time**: Embeddings MUST be generated successfully. Failure = release aborted.
- **Query time**: Requesting `mode="hybrid"` or `mode="semantic"` requires embeddings to exist in the release directory. Missing embeddings = `EmbeddingRequiredError`.
- **Keyword-only**: Declare `modes=["fulltext"]` without `embedding_config` for keyword-only search. No embeddings generated, no embeddings required.

```python
# This raises EmbeddingRequiredError if embeddings are missing:
results = release.search("query", mode="hybrid")

# This raises SearchNotAvailableError if surface doesn't declare semantic:
results = release.search("query", mode="semantic")

# This always works if search.db exists:
results = release.search("query", mode="keyword")
```

## Error Handling

```python
from synix.sdk import (
    SdkError,               # Base class
    SynixNotFoundError,     # No .synix/ found
    ReleaseNotFoundError,   # Named release doesn't exist
    ArtifactNotFoundError,  # Label not in snapshot
    SearchNotAvailableError,# No search.db / wrong mode
    EmbeddingRequiredError, # Declared embeddings missing
    PipelineRequiredError,  # Operation needs pipeline
)
```

All SDK errors inherit from `SdkError`. Catch `SdkError` to handle any SDK-specific failure.

## Scratch Releases

Access HEAD without a named release using scratch realization:

```python
# Context manager cleans up automatically
with project.release("HEAD") as scratch:
    results = scratch.search("query")

# Or manually
release = project.release("HEAD")
results = release.search("query")
release.close()  # clean up work dir
```

## Thread Safety

The SDK is not thread-safe. Each thread should create its own `Project` and `Release` handles. `SearchIndex` and `EmbeddingProvider` instances are not safe to share across threads.

## Versioning

```python
import synix
print(synix.SDK_VERSION)  # "0.1.0"
```

SDK version follows semver independently from the package version:
- Patch: bugfixes
- Minor: new methods
- Major: breaking changes

## CLI ↔ SDK Coverage

| CLI Command | SDK Method |
|---|---|
| `synix init <name>` | `synix.init(path, pipeline=...)` |
| `synix build pipeline.py` | `project.build(pipeline=...)` |
| `synix plan pipeline.py` | `project.build(dry_run=True)` |
| `synix release HEAD --to local` | `project.release_to("local")` |
| `synix revert <ref> --to <name>` | `project.release_to(name, ref=ref)` |
| `synix search "q" --release local` | `release.search("q")` |
| `synix search "q" --ref HEAD` | `release("HEAD").search("q")` |
| `synix lineage <label>` | `release.lineage(label)` |
| `synix list` | `release.artifacts()` |
| `synix show <id>` | `release.artifact(id)` |
| `synix releases list` | `project.releases()` |
| `synix releases show <name>` | `release.receipt()` |
| `synix refs list` | `project.refs()` |
| `synix clean` | `project.clean()` |
| (source management) | `project.source(name).add/list/remove` |
