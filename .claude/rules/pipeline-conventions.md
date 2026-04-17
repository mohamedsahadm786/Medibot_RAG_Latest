---
paths:
  - "backend/services/**/*.py"
  - "backend/api/routes/chat.py"
---

# Pipeline conventions

## LangGraph StateGraph

- All RAG logic is a LangGraph StateGraph in `backend/services/rag_pipeline.py`
- New retrieval steps become **nodes**, not inline functions
- Each node receives state, returns updated state
- State schema is a TypedDict — add new fields there, not ad-hoc

## Model routing (hard rule)

- **GPT-4o** → `generate_answer` node ONLY
- **GPT-4o-mini** → everything else: intent classify, enhance query, HyDE, multi-query, contextual compression, CRAG relevance, hallucination grounding, per-turn summary, RAGAS eval
- Reason: auxiliary calls run 7–8× per query. GPT-4o there = cost blowout.

## Pipeline invariants

- **Parent-child chunking**: children (~300–400 tok) embedded, parents (~1500 tok, 200 overlap) fetched from Postgres for generation
- **Hallucination gate is blocking**: full answer generates before any token streams
- **CRAG retry cap: 1**. On IRRELEVANT + retry_count ≥ 1 → generate with "insufficient context" note
- **Every query writes to `retrieval_logs`** — enhanced_query, hyde_answer, retrieved_child_ids, retrieved_parent_ids, reranker_scores (JSONB), relevance_verdict, hallucination_verdict, ragas_scores (JSONB), total_tokens_used, latency_ms
- **SSE events are typed**: `status`, `token`, `sources`, `done`. Don't invent new event types without updating frontend.

## Async discipline

- `AsyncSession` for all SQLAlchemy queries
- `httpx.AsyncClient` for external HTTP (not `requests`)
- Celery tasks fire with `.delay(...)` after response streams, never before
- Never `time.sleep` in request path — use `asyncio.sleep` if truly needed