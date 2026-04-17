---
paths:
  - "tests/**/*.py"
  - "pytest.ini"
---

# Testing conventions

## Test suites

- `tests/test_*.py` — unit and integration tests (run on every PR)
- `tests/ragas_regression/` — RAGAS quality gate (manual workflow, costs OpenAI tokens)

## Gates

- Unit tests: `pytest tests/ --ignore=tests/ragas_regression/` must pass
- RAGAS regression: avg faithfulness ≥ 0.85, avg answer_relevancy ≥ 0.80
- Individual query: faithfulness < 0.70 fails the suite

## Patterns

- Use `pytest-asyncio` for async tests
- Use `httpx.AsyncClient` for API tests (not TestClient for async routes)
- Use `factory_boy` for fixture data
- Mock OpenAI / Pinecone in unit tests; only hit real services in `ragas_regression/`

## What needs a test

- Every new API endpoint
- Every new LangGraph node
- Every change to retrieval, reranking, or generation logic
- Schema changes (validate migration up + down)