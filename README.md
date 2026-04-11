# MediBot v2

Production-grade medical RAG chatbot powered by a 700+ page medical PDF knowledge base.

## Quick Start

### 1. Create virtual environment and install dependencies
```bash
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 3. Start Docker services
```bash
cd docker
docker compose up -d
```

### 4. Run database migrations
```bash
# From project root
alembic -c alembic/alembic.ini upgrade head
```

### 5. Start the API server (local dev)
```bash
uvicorn backend.main:app --reload
```

### 6. Verify health
```bash
curl http://localhost:8000/api/health
# Expected: {"status":"healthy","database":"connected","redis":"connected"}
```

### 7. Run tests
```bash
pytest
```

## Project Structure

```
MediBot_latest/
├── backend/          FastAPI app, models, schemas, services, RAG pipeline
├── frontend/         React + Vite frontend (Phase 12)
├── docker/           Dockerfiles and docker-compose
├── alembic/          Database migrations
├── tests/            Pytest test suite
├── data/             Medical PDF (not committed)
└── requirements.txt
```

## Build Phases

| Phase | Description |
|---|---|
| 1 | Project scaffolding (this phase) |
| 2 | PDF ingestion pipeline |
| 3 | Basic retrieval + generation |
| 4 | Query transformation (HyDE, multi-query) |
| 5 | Re-ranking + contextual compression |
| 6 | LangGraph agentic pipeline |
| 7 | Conversation memory |
| 8 | SSE streaming |
| 9 | Redis semantic caching |
| 10 | RAGAS + Celery evaluation |
| 11 | Prometheus + Grafana + LangSmith |
| 12 | React frontend |
| 13 | Feedback + rate limiting |
| 14 | RAGAS regression + CI/CD |
| 15 | AWS deployment |
| 16 | Polish + documentation |
