---
name: ragas-regression
description: Run the full RAGAS regression suite (50+ curated medical Q&A pairs) and compare against baseline. Gates merges on faithfulness ≥ 0.85 and answer_relevancy ≥ 0.80. Costs real OpenAI tokens — this is an explicit skill, not auto-invoked.
user-invocable: true
disable-model-invocation: true
---

# RAGAS Regression Skill

Final quality gate before merging pipeline changes. **NOT auto-invoked** — explicit only, because it costs real money.

## When to invoke

- Before merging any PR labeled `pipeline`, `rag`, or touching `backend/services/` files related to RAG
- After changing any prompt template in `query_transform.py`, `generator.py`, `compressor.py`, or any LangGraph node
- After changing model routing (e.g., switching any auxiliary call away from GPT-4o-mini)
- After changing ingestion chunk strategy (parent size, child size, `SemanticChunker` config)

## What it does

1. Loads `tests/ragas_regression/test_suite.json` (50+ entries: question, expected_answer, expected_source_page)
2. Runs each query through the full LangGraph pipeline via `rag_pipeline.ainvoke(...)`
3. Computes RAGAS metrics (faithfulness, answer_relevancy at minimum; the full 6 metrics in `backend/services/evaluation.py`) using GPT-4o-mini
4. Writes `tests/ragas_regression/results.json` with per-query and aggregate scores
5. Asserts:
   - Average `faithfulness >= 0.85`
   - Average `answer_relevancy >= 0.80`
   - No individual query with `faithfulness < 0.70`
6. Exits 1 if any gate fails (blocks merge)

## Commands

```bash
# Full regression — costs real OpenAI tokens
pytest tests/ragas_regression/ -v

# Trigger via manual GitHub Actions workflow
gh workflow run ragas_regression.yml
```

## CI integration

The GitHub Actions workflow `.github/workflows/ragas_regression.yml` is `workflow_dispatch` only (manual trigger). It's separate from `ci.yml` precisely because RAGAS on 50+ queries with GPT-4o-mini is expensive enough that per-push execution doesn't make sense.

## Interpreting results

- **Passing:** merge when CI is green
- **Regression on a subset:** investigate which queries degraded — often isolated to a question category (e.g., pharmacology questions tanking while anatomy stays stable) and points at a specific prompt
- **Widespread regression:** something fundamental changed — retrieval, chunking, or model routing. Revert and investigate before merging