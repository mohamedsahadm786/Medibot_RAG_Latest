---
name: pipeline-debug
description: Trace a specific query through the 8-node LangGraph pipeline using retrieval_logs in PostgreSQL and LangSmith. Use when a user reports a bad answer, when CRAG verdict returns IRRELEVANT unexpectedly, or when hallucination grounding fails. Pulls full pipeline artifacts for forensic analysis.
user-invocable: true
argument-hint: "[message_id: UUID] or [session_id: UUID]"
---

# Pipeline Debug Skill

Forensic debugger for a single query's trip through the MediBot LangGraph pipeline.

## When to invoke

- User reports a factually wrong or hallucinated answer (feedback = "down")
- CRAG relevance_verdict came back IRRELEVANT unexpectedly
- Hallucination grounding check failed (hallucination_verdict = "not_grounded")
- Latency for a specific query exceeded ~30s (check `retrieval_logs.latency_ms`)
- RAGAS faithfulness for a specific query scored < 0.5

## What it fetches

Given a `message_id`, queries `retrieval_logs` + `chat_messages` for:

- **Input:** original user query, `enhanced_query`, `hyde_answer`
- **Retrieval:** `retrieved_child_ids`, `retrieved_parent_ids`, `reranker_scores` (JSONB)
- **Verdicts:** `relevance_verdict` (RELEVANT/PARTIAL/IRRELEVANT), `hallucination_verdict` (grounded/not_grounded)
- **Evaluation:** `ragas_scores` (all 6 metrics as JSONB)
- **Performance:** `total_tokens_used`, `latency_ms`
- **Output:** final answer from `chat_messages.content`
- **LangSmith trace URL:** reconstruct from env `LANGCHAIN_ENDPOINT` + project

## Output format

Markdown report at `debug_reports/{message_id}.md`:

1. **Summary** — final answer, all verdicts, RAGAS scores, latency
2. **Query evolution** — original → enhanced → HyDE → (any retries)
3. **Retrieval table** — child_id | parent_id | rerank_score | page | section
4. **Parent contexts** — full text of compressed contexts that fed generation
5. **Diagnosis** — which node most likely caused the issue, based on which stage the failure appeared at

## Commands

```bash
# Debug by message ID (most common)
python scripts/pipeline_debug.py --message-id <uuid>

# Debug all messages in a session
python scripts/pipeline_debug.py --session-id <uuid>

# Open LangSmith trace in browser
python scripts/pipeline_debug.py --message-id <uuid> --open-langsmith
```

## Common diagnostic patterns

- **Low RAGAS faithfulness + hallucination_verdict=grounded:** grounding check too lenient → tune the grounding prompt in `check_hallucination` node
- **IRRELEVANT after retry:** HyDE drifted semantically → check `enhanced_query` for over-specificity
- **High `latency_ms` with normal verdicts:** usually cross-encoder cold-start on first request → not a bug
- **Missing parent chunks:** child's `parent_id` not joining → check recent ingestion for orphan children

## Note

If `scripts/pipeline_debug.py` doesn't exist yet, this skill describes the intended debugging workflow. Build the script as a thin wrapper over a SQL query on `retrieval_logs` + a call to LangSmith's API for the trace URL. ~50 lines of Python.