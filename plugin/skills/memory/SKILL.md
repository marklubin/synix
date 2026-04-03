---
name: memory
description: Use when the user references prior work, asks what happened recently, needs context about the project or person, or when you should store notes from a call, meeting, or important conversation. Also activate when the user says "remember this" or asks you to search memory.
version: 1.0.0
---

# Synix Memory

You have access to persistent cross-session memory via the `synix` MCP server. This memory survives across conversations and is shared across all projects.

## When to Search

- User asks about prior work, decisions, or context
- You need to understand what's been happening recently
- User references a person, project, or thread you don't have context for
- Start of a complex task where historical context would help

Call `search(query)` with a descriptive query. Results include episodes (session summaries), research threads, and synthesized context.

## When to Ingest

- After a call or meeting the user mentions
- When the user asks you to "remember" something
- When you produce a significant analysis, decision, or document worth preserving

Call `ingest(bucket, content, filename)`:
- `bucket="documents"` — notes, specs, memos, analysis
- `bucket="reports"` — structured reports, summaries

Use descriptive filenames: `call-receipt-benjamin-2026-04-03.md`, `decision-architecture-reset.md`

## Getting Full Context

Call `get_context()` to retrieve the full synthesized context document. This is a living summary of identity, current work, priorities, and recent activity. Use this when you need the big picture.

## Available Buckets

Call `list_buckets()` to see all configured ingestion endpoints.
