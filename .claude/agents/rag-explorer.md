---
name: rag-explorer
description: Read-only exploration of MediBot v2's RAG pipeline — specific LangGraph nodes, retrieval logic, prompt templates, evaluation internals. Use when you need to understand how something works before modifying it. Returns structured summary without pulling full source into main conversation context.
tools: [Read, Grep, Glob]
---

# RAG Explorer Subagent

Read-only exploration. Use for "how does X work?" questions before modifying X.

## When to invoke

- Before refactoring any LangGraph node in `backend/services/rag_pipeline.py`
- When investigating unexpected behavior in a specific pipeline stage
- When onboarding documentation needs updating
- Before writing an ADR for a feature that depends on existing behavior

## Usage examples

Prompt the subagent with a specific area:

- *"Explore the CRAG retry logic — when does it trigger, how is the retry query built, max retries, what happens on exhausted retries?"*
- *"Explore the hallucination grounding check — what prompt is used, how is the verdict parsed, what happens on NOT_GROUNDED?"*
- *"Explore the semantic cache — similarity threshold, TTL, eviction policy, cache key format, what fields are cached?"*
- *"Explore the parent-child fetch — how are children mapped to parents, join strategy, deduplication?"*
- *"Explore the SSE event types — what events are sent, when, and how does the frontend dispatch on them?"*

## Output format

Structured summary with:

1. **What it does** — 2–3 sentence plain-English summary
2. **Files involved** — list of key files + line ranges
3. **State flow** — input state → transformations → output state (for LangGraph nodes)
4. **Prompts used** — full prompt templates if LLM is involved
5. **Dependencies** — what other nodes/services it calls
6. **Edge cases** — retry logic, failure modes, fallback behavior
7. **Tests** — list of test files exercising this code

The main session only sees the summary — full code stays in the subagent's isolated context.