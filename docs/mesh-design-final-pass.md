# Final Pass Review: `mesh-design.md`

## Findings

### 1. Medium: Equal-term tie-break depends on config consistency that is not enforced
- Reference: `mesh-design.md:282`
- Conflict resolution uses "lower index in `leader_candidates` wins". This only works if every node has identical `leader_candidates` ordering.
- Risk: with config drift between nodes, different clients can choose different winners for the same term.
- Recommendation: define a hard invariant and validation at startup:
  - hash of election config (`leader_candidates`) must match across cluster
  - reject join/heartbeat when hash mismatches
  - include hash in `/api/v1/cluster` for operator visibility.

### 2. Medium: Winner selection and step-down rely on client traffic for convergence
- Reference: `mesh-design.md:282`, `mesh-design.md:283`
- The losing leader steps down when it receives requests indicating a higher-priority `(term, leader_id)`. If traffic is sparse or partitioned, it can continue serving writes in its partition.
- Risk: prolonged dual-writer windows in degraded networks.
- Recommendation: add active leader-to-leader probing and a periodic self-check against authoritative cluster view; if ambiguity exists, server should fail closed for writes.

### 3. Low: v1 security model still has a known broad blast radius
- Reference: `mesh-design.md:33`
- Shared token with deferred per-node credentials is explicitly accepted risk.
- Risk: one token leak compromises full mesh until rotation.
- Recommendation: keep as v1 if needed, but add mandatory rotation runbook and a short token TTL option if operationally feasible.

## Ship Readiness
- The design is much stronger than prior revisions.
- No remaining **high** severity blockers in the document.
- I would ship to implementation after adding explicit config-hash validation for election inputs and a fail-closed ambiguity rule for leader convergence.
