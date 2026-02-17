# Review (Pass 2): `batch-build.md`

## Findings

### 1. High: cassette-mode incompatibility conflicts with the documented recording workflow
- References: `batch-build.md:178`, `batch-build.md:436`, `batch-build.md:443`
- Problem: The spec says `SYNIX_CASSETTE_MODE` set is a hard error for `batch-build`, but the recording steps export `SYNIX_CASSETTE_MODE=record` and then run `synix batch-build run ...`.
- Risk: The documented setup flow will fail as written, or the implementation will diverge from the error contract.
- Recommendation: Make the behavior consistent one way:
1. Require unsetting cassette mode before the batch step in docs (`unset SYNIX_CASSETTE_MODE`), or
2. Narrow the hard error to only replay/incompatible modes, not `record`, if that is intended.

### 2. Medium: `--force` is overloaded for multiple dangerous bypasses without scope control
- References: `batch-build.md:112`, `batch-build.md:176`, `batch-build.md:270`
- Problem: `resume --force` is used both for fingerprint mismatch and corrupted-state restart.
- Risk: Operators can bypass the wrong guardrail unintentionally; auditing intent is harder.
- Recommendation: Split into explicit flags (for example `--allow-fingerprint-mismatch` and `--restart-from-corrupt-state`) or require an explicit confirmation token for corruption recovery.

### 3. Medium: request-key normalization language is ambiguous and can imply semantic-loss normalization
- Reference: `batch-build.md:282`
- Problem: `messages` are described as "whitespace-stripped," which can be read as altering prompt content rather than canonical serialization.
- Risk: If interpreted literally in implementation, distinct prompts could collide.
- Recommendation: Clarify wording to "canonical JSON serialization for hashing; preserve message content bytes" and avoid terms that imply content mutation.

## Resolved Since Last Pass
- Pipeline fingerprint mismatch is now hard-fail by default with explicit override.
- Partial-failure terminal state semantics (`completed_with_errors`) are now defined.
- Corrupted state handling now fails safe instead of silently resetting.
- Transform idempotency expectations are now documented.
- Sync-only layer behavior in `batch-build run` is now explicitly defined.

## Suggested Test Additions
- Enforce cassette incompatibility contract: verify exact behavior when `SYNIX_CASSETTE_MODE=record` during `batch-build run`.
- Force-flag safety: verify each risky override path requires the intended explicit flag.
- Request-key canonicalization: ensure prompt-content whitespace changes affect keys unless content is intentionally normalized by design.
