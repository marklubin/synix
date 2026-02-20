# Review: `mesh-design.md`

## Findings

### 1. High: No authentication or transport security on a network-exposed control plane
- Reference: `mesh-design.md:73`, `mesh-design.md:169`
- The design exposes an HTTP API on `0.0.0.0:7433` and defines write-capable endpoints (`/sessions`, `/builds/trigger`, `/heartbeat`) without any authn/authz or TLS model.
- Risk: any reachable host can submit arbitrary data, trigger builds, poison cluster state, and read artifacts/search data.
- Recommendation: require mTLS or signed node tokens for all API calls, plus explicit authorization scopes (client submit/pull, operator trigger, admin cluster ops). Document certificate/token provisioning and rotation in the onboarding flow.

### 2. High: Leader election can create split-brain and inconsistent state
- Reference: `mesh-design.md:214`, `mesh-design.md:218`, `mesh-design.md:220`
- Election is based on heartbeat misses + `tailscale ping`, with no fencing/lease mechanism. Network partition scenarios can produce two active leaders, each accepting writes and building independently.
- Risk: divergent `sessions.db`, conflicting bundles, non-deterministic client behavior.
- Recommendation: use a lease/term-based election with monotonic term numbers and leader fencing (reject writes from stale terms). Define write acceptance rules and conflict behavior explicitly.

### 3. High: Failover path implies data loss for non-local or deleted sessions
- Reference: `mesh-design.md:188`, `mesh-design.md:220`
- State is leader-local (`sessions.db` + `sessions/`), and failover recovery relies on clients re-submitting from `watch_dir`.
- Risk: previously ingested sessions are lost if no client still has source files locally (deletions, retention, offline clients). Rebuilt artifacts may regress.
- Recommendation: replicate durable server state (sessions metadata + payloads) to at least one peer, or persist to shared/object storage. Define RPO/RTO targets and recovery guarantees.

### 4. Medium: Systemd unit naming and instantiation are internally inconsistent
- Reference: `mesh-design.md:26`, `mesh-design.md:32`, `mesh-design.md:449`, `mesh-design.md:467`, `mesh-design.md:487`
- The doc alternates between names like `synix-mesh-marks-claude-server` and template units `synix-mesh@server.service`/`synix-mesh@client.service`, then enables `synix-mesh@marks-claude-server`.
- Risk: provisioning scripts and operator commands can drift; unit activation may fail or start unexpected instances.
- Recommendation: standardize on one naming scheme and show exact install/start commands that match the unit filenames. Example options:
  - `synix-mesh-server@.service` + `synix-mesh-server@marks-claude`
  - `synix-mesh@.service` + role in config/state, not in unit suffix.

### 5. Medium: Build scheduling lacks explicit concurrency/serialization contract
- Reference: `mesh-design.md:193`, `mesh-design.md:195`, `mesh-design.md:175`
- The design has debounced scheduling and manual triggers but does not define single-flight behavior (what happens if a trigger arrives during an active build).
- Risk: overlapping builds, stale bundle publication order, race conditions around deploy hooks.
- Recommendation: specify an explicit state machine (`idle`, `queued`, `running`) and enforce one active build per mesh with deterministic queue/coalesce semantics.

### 6. Medium: Test plan misses adversarial and failure-mode coverage for stated guarantees
- Reference: `mesh-design.md:355`, `mesh-design.md:493`
- Current tests validate happy-path distribution and basic failover, but not authentication failures, split-brain behavior, idempotent replay, hook sandboxing, or rollback on failed deploy hooks.
- Risk: critical production failure modes are undocumented and untested.
- Recommendation: add E2E cases for partition-induced dual leaders, stale leader write rejection, unauthorized request rejection, interrupted bundle downloads, and deterministic recovery after crash/restart.

## Open Questions
- Is mesh traffic assumed to be private Tailscale-only, or internet-routable in some deployments?
- What are the intended durability guarantees for ingested sessions (best-effort vs no-loss)?
- Should deploy hooks be trusted-local only, or do you need guardrails/sandboxing for shared environments?

## Summary
The document is strong on operator UX and component decomposition, but it currently under-specifies security, consensus safety, and durability semantics. Those three areas should be resolved before implementation to avoid structural rework.
