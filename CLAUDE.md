# MediBot v2

Production medical RAG chatbot. 8-node LangGraph pipeline, Pinecone hybrid search, AWS deployment.

## Stack (names only — deep details in .claude/rules/)

- **Backend:** FastAPI (async), SQLAlchemy+asyncpg, Alembic, Pydantic v2
- **Orchestration:** LangGraph StateGraph, LangChain
- **LLMs:** OpenAI GPT-4o (generation), GPT-4o-mini (auxiliary)
- **Retrieval:** Pinecone Serverless hybrid, PubMedBERT dense, BM25 sparse, cross-encoder rerank
- **Data:** PostgreSQL (port 5433 local), Redis, Celery
- **Frontend:** React 18 + Vite + Tailwind + Framer Motion
- **Infra:** Docker Compose (8 containers), AWS EC2, ECR, GitHub Actions, Nginx
- **Observability:** Prometheus + Grafana + LangSmith + RAGAS

## Project layout

- `backend/services/` — pipeline logic (LangGraph nodes, retrievers, generators)
- `backend/api/routes/` — HTTP endpoints
- `backend/models/` — SQLAlchemy models
- `backend/core/metrics.py` — Prometheus metrics
- `frontend/src/` — React app
- `tests/ragas_regression/` — RAGAS quality gate (manual CI)
- `alembic/` — DB migrations
- `docker/`, `monitoring/` — infra configs

## Absolute rules

- **GPT-4o is for final answer generation ONLY.** Auxiliary calls use GPT-4o-mini.
- **Hallucination check is blocking** — full answer generates before any token streams.
- **CRAG retry cap is 1.**
- **Never run long commands in Claude Code terminal** — see @.claude/rules/terminal-commands.md

## For detail

- Pipeline conventions → loads when editing `backend/services/**`
- DB schema → loads when editing `backend/models/**`
- API conventions → loads when editing `backend/api/**`
- Testing rules → loads when editing `tests/**`
- Terminal safety → always loaded