# Experiential Policy Memory (EPM) — Design Document

**Status:** Idea capture / backlog
**Date:** Feb 9, 2026
**Author:** Mark Lubin
**Relationship to Synix:** Future expansion — runtime learning layer complementing Synix's retrospective memory build system

---

## One-Sentence Summary

A self-improving procedural agent layer where the LLM invents candidate strategies and a bandit-backed policy memory learns, from real outcomes, which strategies work in which states — online, safely, and without weight updates.

---

## The Problem

Agentic systems have three compounding cost problems:

1. **They fail for structural reasons** — missing steps, wrong tool order, brittle UI/tool interaction, untracked constraints, lack of verification. These aren't model capability issues, they're procedural competence issues.

2. **They figure things out from scratch every time** — an agent that successfully navigated a complex workflow yesterday has zero memory of that success today. Every invocation pays the full exploration cost again. Without manual prompt engineering to encode "here's how to do X," the agent rediscovers the wheel on every run.

3. **The exploration cost is real money** — every failed attempt, every redundant tool call, every retry burns tokens and API calls. An agent that takes 5 attempts to get something right costs 5x what an agent that learned from its first success costs. At enterprise scale across thousands of daily tasks, the waste is enormous.

Current approaches optimize the wrong surface area:

- **"Agent memory" (RAG-style):** Stores prose, retrieves by semantic similarity. Doesn't preserve causal structure of competence (state → action → outcome). Useful for knowledge, useless for skill.
- **Manual prompt engineering:** Works but doesn't scale. Someone has to observe failures, diagnose them, and encode fixes as instructions. That's a human doing the learning the system should do itself.
- **LLM routing products:** Treat "paths" as model/tool/temperature configs. That's a bandit over configurations, not over strategies. Helps at the margin but doesn't create procedural competence.
- **Fine-tuning:** Too global, too slow, too risky. Can't safely adapt to per-deployment tool quirks, UI drift, or user-specific preferences without constant retraining.

The gap between "semantic RAG memory" and "full RL with weight updates" is real and underserved. EPM fills it — and the value prop is both **higher success rates** and **lower cost per successful outcome** as the system learns.

## The Insight

The minimal structure that creates real learning is:

1. **Canonical semantic state** — a typed snapshot of what is known, what is missing, and what environment/tool context applies
2. **Constrained action/plan representation** — a structured plan graph that can be executed, compared, and deduplicated independent of wording
3. **Verifiable outcomes** — a reward signal that reflects real success, not vibes
4. **Online selection + update** — explore/exploit that allocates traffic to strategies that work and retires strategies that don't

This is "experiential learning" without weight updates. Learning lives in a policy memory substrate, not in model parameters.

## How It Works

### Runtime flow for each task attempt:

**1. Build canonical state S**

Example fields for a `schedule_meeting` task:
- Task type
- Slots filled/missing (attendees_known, timezone_known, duration_known)
- Environment context (calendar provider, auth state, locale)
- Recent failure codes (conflict_check_failed, ambiguous_attendee)

S is normalized, hashed, and clustered for state abstraction.

**2. Generate candidate plans A₁…Aₙ**

The LLM produces structured plans (not prose):
- Step graph (tool calls, questions, checks)
- Preconditions and invariants
- Commit gates (requires confirmation)
- Verification steps

Plans are compiled into a normalized representation so semantically equivalent plans map to the same "arm."

**3. Select plan via explore/exploit**

For the current state cluster, maintain a posterior over success for each known plan. Use Thompson Sampling or UCB:
- **Exploit:** Pick plans with proven outcomes in this state cluster
- **Explore:** Occasionally try new or uncertain plans, inside safe action sets only

**4. Execute with safety gates**

Hard gates prevent dangerous exploration:
- Irreversible actions require explicit confirmation
- Required verifications must pass before commit
- Tool calls validate schemas and constraints
- Sandboxing for UI/system actions

Outcome scored by verifier strength:
- **Strong:** "Calendar event exists with 3 attendees and no conflicts"
- **Medium:** "Email draft accepted with minimal edits"
- **Weak:** "User corrected facts" (negative) / "user asked for rewrite" (partial failure)

**5. Update policy store**

Store the episode, update plan posterior for that state cluster. Decay old evidence, version by environment/tool versions so drift doesn't poison learning.

---

## Data Flywheel + Moat

The policy store enables a public/private split:

- **Private policies:** Learned from Company A's deployment, stay with Company A. Capture their specific tool quirks, user preferences, environment context.
- **Public/shared policies:** Aggregated across deployments, anonymized, offered as starter policies to new customers. Eliminates cold start.

Every execution generates training signal → better policies → better next execution → more usage. Competitors can copy the architecture but not the accumulated policy data. Classic "gets better with use" defensibility.

---

## Relationship to Synix

Synix and EPM are complementary, not overlapping:

| | Synix | EPM |
|---|---|---|
| **What it stores** | Declarative knowledge (what the agent knows) | Procedural knowledge (what strategies work) |
| **Temporal mode** | Retrospective — builds from historical data | Prospective — learns from live execution |
| **Learning signal** | Content structure (transcripts, summaries) | Outcome signal (success/failure/cost) |
| **Provenance** | Derivation tracking (what came from what) | Causal tracking (state → action → outcome) |

A full agent memory system needs both: "I remember this user prefers morning meetings" (Synix) + "For this calendar API, checking conflicts before invites has 93% success rate" (EPM).

**Integration point:** EPM gives Synix a reason to be in the inference loop from day one. The runtime policy layer forces inference-context-aware provenance — tracking what memory state was active when a decision was made — which is exactly the provenance evolution identified in the Synix v0.9 review.

**Sequencing:** Ship Synix first (memory build system), get traction, then EPM becomes the "expansion into runtime learning" narrative. Two products that reinforce each other, pitched separately.

---

## Steel Case (Why This Wins)

- **Learns the right thing:** Strategies, not configs. Can fix "missing step" failures by promoting plans that include the right checks.
- **Eliminates redundant exploration:** An agent that figured out how to do X yesterday doesn't pay the discovery cost again today. The policy store is institutional memory for procedural knowledge.
- **Direct cost savings:** Fewer failed attempts = fewer token/API costs per successful outcome. Explore/exploit converges toward the cheapest successful strategy, not just any successful strategy. At enterprise scale (thousands of daily agent tasks), the savings compound fast — this is a line item CFOs can see.
- **Replaces manual prompt engineering:** Instead of a human observing failures, diagnosing them, and encoding fixes as prompt instructions, the system does this automatically from outcome signals. The human's job shifts from "teach the agent how" to "define what success looks like."
- **Per-deployment adaptation:** Policy store is conditioned on environment context. No retraining needed for new tool versions or user preferences.
- **Debuggable:** Every decision backed by retrieved evidence ("we chose plan P because 93% success rate in this state cluster over 30 days").
- **Safe:** Updates are local, versioned, reversible. No weight modification risk.
- **Novel positioning:** Individual components exist in literature (bandits, canonical state, plan dedup). The specific combination — bandit-backed policy memory as a runtime layer without weight updates — hasn't shipped as a product. Academic work on "learning from experience in LLM agents" is mostly in-context learning or fine-tuning, not external policy stores with explore/exploit.

## Straw Case (Hard Problems to Solve)

### 1. State representation is a research problem, not a configuration knob

"Build canonical state S" does enormous work in one sentence. State abstraction is the hard part of the entire system:
- Too coarse → aliasing (different situations collapse, policy learns noise)
- Too fine → every situation unique, never enough data to learn
- Who defines the state schema? Developer? LLM? Learned?
- "Detect aliasing via high reward variance" is correct in theory but requires significant data volume per cluster

### 2. Plan deduplication is unsolved at production scale

"Two differently worded plans that do the same thing become the same arm" — how? Semantic equivalence of structured plan graphs is not a solved problem. Without reliable dedup, the bandit has a combinatorial explosion of arms, most semantically identical, and never accumulates enough signal on any single arm.

### 3. Verifier coverage is the real bottleneck

The system needs machine-checkable success signals. Works for: scheduling meetings, filing tickets, structured CRUD. Breaks for: creative tasks, ambiguous user intent, long-horizon outcomes. "Weak" signals like user edits may be too noisy to drive posterior updates meaningfully.

### 4. Credit assignment in multi-step tasks

The doc acknowledges this: "credit assignment and safety complexity rise sharply." Bandit-over-whole-plan is the wedge, but the real value is step-level learning ("when to ask, when to verify, when to branch"). Getting there requires strong intermediate reward signals that most deployments won't have initially.

---

## Success Metrics

- First-try success rate by task type and state cluster
- Median steps and time to success
- Repeat failure rate (same state cluster fails twice → should approach zero)
- Cost per successful outcome
- Drift resilience (performance under tool/UI changes with policy versioning)
- Cold-start time for new deployments using shared policies vs. from scratch

---

## Key Risks and Mitigations

| Risk | Detection | Mitigation |
|------|-----------|------------|
| State aliasing | High reward variance for same (S, plan) | Add discriminating slots, split clusters |
| Reward hacking | Optimize proxy, not reality | Strengthen verifiers, add negative checks, require confirmations |
| Exploration damage | Unsafe actions during explore phase | Restrict to safe subsets, canary new plans, require verification before commits, maintain rollback |
| Plan explosion | Too many unique arms, no convergence | Better plan normalization, minimum execution threshold before creating new arm |
| Drift poisoning | Old policies applied to changed environment | Version policies by environment/tool version, decay old evidence |

---

## Next Steps (When We Get Here)

1. Define a constrained task domain for v0.1 (e.g., calendar scheduling with a specific API)
2. Solve plan normalization for that domain — what makes two plans "the same"?
3. Build state schema + clustering for that domain
4. Implement bandit with Thompson Sampling over whole-plan arms
5. Build verifier for the domain (machine-checkable success criteria)
6. Measure: does success rate actually improve over 100 episodes?

**Do not start until Synix has traction and the memory build system is proven.**

---

## YC "Other Ideas" Blurb

*Experiential Policy Memory — a runtime learning layer for AI agents that learns which strategies work in which situations from real execution outcomes, without fine-tuning. Bandit-backed policy store with explore/exploit, safety gates, and full audit trail. Natural expansion of our memory build system (Synix) into the runtime loop. Public/private policy split creates a data flywheel moat.*
