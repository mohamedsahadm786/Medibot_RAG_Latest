---
name: rag-eval
description: Run RAGAS evaluation (6 metrics) plus custom retrieval quality checks against the medical Q&A test set. Use when pipeline changes need validation, when investigating quality regressions in Grafana, or before merging any PR touching retrieval, reranking, generation, or grounding logic.
user-invocable: true
---

# RAG Evaluation Skill

Validates the MediBot RAG pipeline end-to-end. Runs RAGAS's 6 metrics (faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness, answer_similarity) against the curated test set.

## When to invoke

- Before merging any PR touching `backend/services/rag_pipeline.py`, `retriever.py`, `generator.py`, `reranker.py`, `compressor.py`, `query_transform.py`
- When RAGAS regression CI fails and you need local reproduction
- When Grafana shows `ragas_faithfulness` or `ragas_relevancy` trending down over 24h
- After changing model routing, prompt templates, or chunk strategy

## Steps

1. Confirm Postgres + Redis + Pinecone are reachable (the test hits the real pipeline)
2. Load test set from `tests/ragas_regression/test_suite.json`
3. For each entry `{question, expected_answer, expected_source_page}`:
   - Invoke the LangGraph pipeline directly (not via HTTP — call `rag_pipeline.ainvoke(...)` from the test harness)
   - Capture generated answer and compressed contexts
   - Run RAGAS metrics using GPT-4o-mini as evaluator
4. Aggregate averages + per-query scores
5. Write report to `tests/ragas_regression/results.json`
6. Assert gates (matches CI config):
   - Average faithfulness ≥ 0.85 (hard gate)
   - Average answer_relevancy ≥ 0.80
   - No individual query with faithfulness < 0.70

## Commands

```bash
# Full eval — note: costs real OpenAI tokens
pytest tests/ragas_regression/test_ragas_regression.py -v

# Subset by question category (if tagged in test_suite.json)
pytest tests/ragas_regression/test_ragas_regression.py -v -k "diabetes"

# Run from the RAGAS evaluation service directly (no pytest wrapper)
python -m backend.services.evaluation --test-set tests/ragas_regression/test_suite.json
```

## Cost note

Full RAGAS eval on 50-query test set uses GPT-4o-mini for all 6 metrics per query. This is why the regression is a manual GitHub Actions workflow (`.github/workflows/ragas_regression.yml`), not part of every-push CI. Don't run on every commit.