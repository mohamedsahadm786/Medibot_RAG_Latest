---
paths:
  - "backend/api/**/*.py"
---

# API conventions

## Endpoint patterns

- All endpoints in `backend/api/routes/`, one file per domain (chat.py, feedback.py, admin.py, health.py)
- Request/response schemas in `backend/schemas/`, Pydantic v2 (`model_dump()`, not `.dict()`)
- All endpoints async, all use `AsyncSession` dependency
- Every new endpoint needs a test in `tests/`

## Rate limiting

- slowapi + Redis, 10 req/min per IP
- Applied to `/api/chat/stream` and `/api/chat`
- Return 429 with `{"error": "Rate limit exceeded. Please wait before sending another message."}`

## SSE streaming

- `/api/chat/stream` uses `sse-starlette`
- Event types: `status`, `token`, `sources`, `done`
- Status events fire from LangGraph nodes via state's `status_events` list
- Token streaming only after hallucination check passes

## Session handling

- `session_id` comes from frontend (browser-generated UUID)
- No auth — all queries scoped by `session_id`
- Rate limiting uses IP, conversation isolation uses session_id (separate concerns)