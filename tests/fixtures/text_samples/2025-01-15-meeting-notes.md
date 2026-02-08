---
title: Q1 Planning Meeting Notes
date: 2025-01-15
tags: [planning, engineering, q1]
author: Mark
---

# Q1 Planning Meeting

## Attendees
- Mark (Engineering Lead)
- Sarah (Product Manager)
- James (Backend Engineer)

## Key Decisions

1. **Migrate to CockroachDB** - Approved the RFC for PostgreSQL to CockroachDB migration. Timeline: 3 months starting February.

2. **Hire two more backend engineers** - Focus on distributed systems experience. Sarah to coordinate with recruiting.

3. **Launch expense tracker v2** - Internal tool getting traction. Add receipt scanning (LLM-based) and budget alerts.

## Action Items

- Mark: Finalize migration roadmap by January 22
- Sarah: Write job descriptions for backend roles
- James: Prototype CockroachDB schema conversion with MOLT

## Notes

The team agreed that data integrity is the top priority for the migration. We will use a phased approach with dual-write verification before any cutover.

Budget for Q1 infrastructure: $50,000 (approved).

Next meeting: January 29, 2025.
