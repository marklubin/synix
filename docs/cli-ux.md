# CLI UX Requirements

The CLI is a selling point. Use Click + Rich to make it feel polished.

## Commands

```bash
synix build pipeline.py                          # Produce immutable snapshot in .synix/
synix release HEAD --to local                    # Materialize projections to a named release
synix search "query" --release local             # Search a release target
synix lineage <artifact-id>                      # Provenance tree (reads from .synix/)
synix list                                       # All artifacts in current snapshot
```

## Live Progress (CRITICAL)

Every CLI command must show live interactive progress. The user must never stare at a frozen terminal.

- Any operation >1 second must show a Rich spinner or progress bar
- Spinner text must indicate what's happening (e.g., "Summarizing conversation abc123 [3/47]")
- Long-running engine functions must accept an `on_progress` callback

## Per-Command UX

- **`synix build`**: Rich progress bars per layer. Show: layer name, artifact count, built/cached/skipped. Final summary table with timing. Report snapshot oid and suggest `synix release` as next step.
- **`synix release`**: Show each adapter's plan (new/removed/unchanged artifacts), then apply progress. Final receipt summary.
- **`synix revert`**: Same output as `synix release` — it is a release of an older snapshot.
- **`synix search`**: Results as Rich panels — layer label colored by level, content snippet, artifact label. Provenance chain as indented tree below each result. Requires `--release <name>` when multiple releases exist.
- **`synix lineage`**: Rich Tree widget — full dependency graph from artifact to raw transcript. Reads from `.synix/objects/` via `SnapshotView`.
- **`synix list`**: Rich table — layer name, artifact count. Reads from `.synix/objects/` via `SnapshotView`.
- **`synix releases list`**: Rich table — release name, snapshot oid (truncated), released timestamp, pipeline name.
- **`synix releases show <name>`**: Receipt details — adapters, targets, artifact counts, status.
- **`synix refs list`**: Rich table — all refs (heads, runs, releases) with their snapshot oids.
- **`synix refs show <ref>`**: Resolved snapshot details — oid, manifest, artifact count, projections.
- **`synix clean`**: Removes `.synix/releases/` and `.synix/work/`. Does not delete snapshot history or objects.

## Color Scheme (by layer level)

- Level 0: dim/grey
- Level 1: blue
- Level 2: green
- Level 3: gold/yellow

## Error Handling

- All errors through Rich console with clear messages
- Never a raw Python traceback in normal operation
- Every command and option has clear `--help` text
