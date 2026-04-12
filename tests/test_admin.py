"""
Tests for admin endpoints (Phase 12):
  GET  /api/admin/stats
  POST /api/feedback
  DELETE /api/admin/cache/clear  (already tested implicitly; basic smoke test added)
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_db_override(messages=None, feedback_rows=None, ragas_scores=None, avg_latency=0.0):
    """Build a mock AsyncSession that returns canned query results.

    Matching strategy: compile the statement with literal binds so parameter
    values appear in the SQL text, then pattern-match on that text.
    """
    from sqlalchemy.dialects import sqlite

    def _sql(stmt: object) -> str:
        try:
            compiled = stmt.compile(  # type: ignore[union-attr]
                dialect=sqlite.dialect(),
                compile_kwargs={"literal_binds": True},
            )
            return str(compiled).lower()
        except Exception:
            return str(stmt).lower()

    async def _execute(stmt):
        result = MagicMock()
        sql = _sql(stmt)

        if "count" in sql and "chat_messages" in sql:
            # Total user message count
            result.scalar_one.return_value = len(messages) if messages else 0
        elif "user_feedback" in sql and "group by" in sql:
            # Feedback aggregate rows
            result.all.return_value = feedback_rows or []
        elif "ragas_scores" in sql:
            result.scalar_one_or_none.return_value = ragas_scores
        elif "avg" in sql and "latency" in sql:
            result.scalar_one.return_value = avg_latency
        elif "chat_messages" in sql and "assistant" in sql:
            # Message existence check for the feedback endpoint
            if messages:
                msg = MagicMock()
                msg.id = messages[0]
                result.scalar_one_or_none.return_value = msg
            else:
                result.scalar_one_or_none.return_value = None
        elif "user_feedback" in sql and "message_id" in sql:
            # Duplicate-feedback lookup — return None (no prior feedback)
            result.scalar_one_or_none.return_value = None
        else:
            result.scalar_one.return_value = 0
            result.scalar_one_or_none.return_value = None
            result.all.return_value = []

        return result

    mock_db = MagicMock()
    mock_db.execute = AsyncMock(side_effect=_execute)
    mock_db.add = MagicMock()
    mock_db.delete = AsyncMock()
    mock_db.commit = AsyncMock()
    return mock_db


# ── GET /api/admin/stats ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_admin_stats_returns_200() -> None:
    """GET /api/admin/stats should return 200 with expected keys."""
    from backend.database import get_db

    mock_db = _make_db_override()
    app.dependency_overrides[get_db] = lambda: mock_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/admin/stats")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert "total_queries" in data
    assert "feedback" in data
    assert "latest_ragas" in data
    assert "avg_latency_ms" in data


@pytest.mark.asyncio
async def test_admin_stats_aggregates_feedback() -> None:
    """Feedback counts should be aggregated from DB rows."""
    from backend.database import get_db

    mock_db = _make_db_override(
        feedback_rows=[("up", 10), ("down", 3)],
        avg_latency=450.0,
    )
    app.dependency_overrides[get_db] = lambda: mock_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/admin/stats")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["feedback"]["up"] == 10
    assert data["feedback"]["down"] == 3
    assert data["avg_latency_ms"] == 450.0


@pytest.mark.asyncio
async def test_admin_stats_includes_ragas_when_present() -> None:
    """latest_ragas should contain the scores from the most recent retrieval log."""
    from backend.database import get_db

    scores = {"faithfulness": 0.91, "answer_relevancy": 0.87}
    mock_db = _make_db_override(ragas_scores=scores)
    app.dependency_overrides[get_db] = lambda: mock_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/admin/stats")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["latest_ragas"] == scores


# ── POST /api/feedback ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_records_thumbs_up() -> None:
    """POST /api/feedback with valid message_id and 'up' should return 200."""
    from backend.database import get_db

    msg_id = str(uuid.uuid4())
    mock_db = _make_db_override(messages=[msg_id])

    app.dependency_overrides[get_db] = lambda: mock_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/feedback", json={"message_id": msg_id, "feedback": "up"}
        )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["message_id"] == msg_id
    assert data["feedback"] == "up"
    assert "feedback_id" in data


@pytest.mark.asyncio
async def test_feedback_records_thumbs_down() -> None:
    """POST /api/feedback with 'down' should return 200."""
    from backend.database import get_db

    msg_id = str(uuid.uuid4())
    mock_db = _make_db_override(messages=[msg_id])

    app.dependency_overrides[get_db] = lambda: mock_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/feedback", json={"message_id": msg_id, "feedback": "down"}
        )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["feedback"] == "down"


@pytest.mark.asyncio
async def test_feedback_rejects_invalid_message_id() -> None:
    """POST /api/feedback with a non-UUID message_id should return 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/feedback", json={"message_id": "not-a-uuid", "feedback": "up"}
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_feedback_returns_404_for_missing_message() -> None:
    """POST /api/feedback for a non-existent message should return 404."""
    from backend.database import get_db

    # No messages — DB returns None for the existence check
    mock_db = _make_db_override(messages=[])
    app.dependency_overrides[get_db] = lambda: mock_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/feedback",
            json={"message_id": str(uuid.uuid4()), "feedback": "up"},
        )

    app.dependency_overrides.clear()
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_feedback_rejects_invalid_feedback_value() -> None:
    """POST /api/feedback with a value other than 'up'/'down' should return 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/feedback",
            json={"message_id": str(uuid.uuid4()), "feedback": "maybe"},
        )
    assert response.status_code == 422


# ── DELETE /api/admin/cache/clear (smoke test) ────────────────────────────────

@pytest.mark.asyncio
async def test_cache_clear_returns_count() -> None:
    """DELETE /api/admin/cache/clear should return cleared_entries."""
    with patch(
        "backend.api.routes.admin.clear_cache",
        new=AsyncMock(return_value=5),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/api/admin/cache/clear")

    assert response.status_code == 200
    assert response.json()["cleared_entries"] == 5
