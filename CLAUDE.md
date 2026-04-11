# MediBot v2 — Next-Gen Medical RAG Platform

## About
FastAPI + React medical chatbot with advanced RAG pipeline,
LangGraph orchestration, Pinecone hybrid search, and 
full observability stack.

## Tech stack
- Backend: FastAPI, Python 3.11+, LangGraph, LangChain
- Frontend: React + Vite + Tailwind + Framer Motion  
- Database: PostgreSQL (port 5433), Redis
- Vector DB: Pinecone (hybrid: dense + sparse)
- LLM: OpenAI GPT-4o (generation), GPT-4o-mini (auxiliary)
- Task queue: Celery + Redis
- Monitoring: Prometheus, Grafana, LangSmith, RAGAS

## Project structure
backend/ — FastAPI application
frontend/ — React application  
docker/ — Dockerfiles and compose config
tests/ — Test suites including RAGAS regression
docs/ — Architecture docs and ADRs

## Key commands
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
cd frontend && npm run dev
docker compose up -d
pytest tests/

## Conventions
- All API endpoints go in backend/api/routes/
- Environment variables in .env (never commit)
- Use Pydantic models for all request/response schemas
- Every new endpoint needs a test