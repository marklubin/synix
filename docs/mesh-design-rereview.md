# Re-Review: `mesh-design.md`

## Findings

### 1. High: Split-brain is still possible because equal-term leaders are not fenced
- Reference: `mesh-design.md:264`, `mesh-design.md:267`, `mesh-design.md:268`, `mesh-design.md:284`
- The design says each promoted leader increments term locally and fences only requests with a **lower** term.
- In a partition/race, two candidates can both increment from term `N` to `N+1` and both accept writes because neither is lower-term.
- Impact: two active leaders can ingest different sessions and produce divergent bundles.
- Recommendation: define a single-writer lease source (quorum KV, lock service, or designated authority) or add deterministic tie-breaking for equal terms plus write fencing keyed by `(term, leader_id)` where only one leader_id is valid per term.

### 2. Medium: Shared bearer token gives no node identity or scoped authorization
- Reference: `mesh-design.md:20`, `mesh-design.md:21`, `mesh-design.md:201`, `mesh-design.md:205`
- All nodes share one token, so any compromised client can impersonate any other client and perform all write operations.
- Impact: no per-node revocation, no action-level attribution, broad blast radius on secret leak.
- Recommendation: move to per-node credentials (or at minimum role-scoped tokens) and record caller identity in server-side audit fields.

### 3. Medium: RPO statement conflicts with ingestion model and is hard to validate
- Reference: `mesh-design.md:31`, `mesh-design.md:34`, `mesh-design.md:37`
- RPO is stated as "up to `scan_interval` seconds of sessions (files submitted but not yet on any client)." If files are submitted, they originated from a client; that wording is internally inconsistent.
- Impact: operators cannot reason clearly about data-loss boundaries during failover.
- Recommendation: restate RPO in explicit phases, e.g.:
  - unsent local files: bounded by `scan_interval`
  - sent-but-not-durably-replicated files: lost on leader disk loss
  - replayable files: recoverable if still present in any client `watch_dir`

### 4. Medium: Manual build trigger semantics under RUNNING are underspecified
- Reference: `mesh-design.md:241`, `mesh-design.md:242`, `mesh-design.md:243`
- `builds/trigger` returns `409` during RUNNING, but behavior for operator intent is unclear (drop vs queue a follow-up build).
- Impact: forced rebuild workflows may silently fail during long-running builds.
- Recommendation: define trigger policy explicitly: either enqueue one "force build" flag or return `409` with a required retry contract and CLI backoff behavior.

### 5. Low: Public health endpoint may leak operational metadata
- Reference: `mesh-design.md:201`, `mesh-design.md:213`
- `/api/v1/health` is intentionally unauthenticated. That can be fine, but should be explicitly constrained to minimal output.
- Impact: unnecessary reconnaissance surface if endpoint includes version/build stats.
- Recommendation: specify that unauthenticated health returns only liveness (e.g., `200 OK`) and move detailed health to authenticated endpoints.

## What Improved Since Prior Review
- Added explicit auth requirement for most endpoints (`mesh-design.md:201`).
- Added single-flight scheduler semantics (`mesh-design.md:238`-`mesh-design.md:243`).
- Standardized systemd naming and activation examples (`mesh-design.md:521`, `mesh-design.md:540`, `mesh-design.md:561`).
- Added term-aware election/fencing intent and related tests (`mesh-design.md:262`, `mesh-design.md:462`).

## Open Questions
- Is a strict single-leader guarantee required, or is eventual reconciliation acceptable for your use case?
- Do you need operator/user-level auditability for who triggered builds and submitted sessions?
- Will token rotation/revocation be in v1, or explicitly deferred with documented risk acceptance?

## Summary
The design is materially better and close to implementation-ready. The remaining blocker is election safety under equal-term races. Resolve that, tighten credential scoping, and clarify RPO/trigger contracts, and this should be in good shape for build-out.
