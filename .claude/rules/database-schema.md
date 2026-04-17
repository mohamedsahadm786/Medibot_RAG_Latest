---
paths:
  - "backend/models/**/*.py"
  - "alembic/versions/**/*.py"
---

# Database schema rules

## Four tables

- **chat_messages**: id (UUID PK), session_id (UUID, indexed), role (user/assistant), content (TEXT), summary (TEXT nullable), created_at
- **document_chunks**: id (UUID PK), chunk_type (parent/child), parent_id (UUID nullable, FK to self), content (TEXT), page_number (INT), section_heading (VARCHAR), source_pdf (VARCHAR), created_at
- **retrieval_logs**: id (UUID PK), message_id (FK → chat_messages), enhanced_query, hyde_answer, retrieved_child_ids (ARRAY UUID), retrieved_parent_ids (ARRAY UUID), reranker_scores (JSONB), relevance_verdict, hallucination_verdict, ragas_scores (JSONB), total_tokens_used, latency_ms, created_at
- **user_feedback**: id (UUID PK), message_id (FK → chat_messages), feedback (up/down), created_at

## Conventions

- All primary keys are UUID
- All timestamps `created_at` default `now()`
- JSONB for flexible scores (reranker_scores, ragas_scores) — don't flatten into columns
- ARRAY UUID for retrieved_*_ids — don't use JSON
- Foreign keys to chat_messages on delete: SET NULL (logs survive deleted messages)

## Migrations

- Alembic for every schema change — never edit tables via raw SQL
- Migration files in `alembic/versions/` named `NNN_description.py`
- Run `alembic upgrade head` manually — never let Claude run it
- After editing models: `alembic revision --autogenerate -m "..."` (user runs, Claude writes the resulting file)