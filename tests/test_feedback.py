"""
Tests for the feedback endpoint (Phase 13):
  POST /api/feedback
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


def _make_db(message_exists: bool = True, has_existing_feedback: bool = False):
    """Build a mock AsyncSession for feedback endpoint tests."""

    async def _execute(stmt):
        result = MagicMock()
        from sqlalchemy.dialects import sqlite

        try:
            sql = str(
                stmt.compile(
                    dialect=sqlite.dialect(),
                    compile_kwargs={"literal_binds": True},
                )
            ).lower()
        except Exception:
            sql = str(stmt).lower()

        if "chat_messages" in sql and "assistant" in sql:
            result.scalar_one_or_none.return_value = MagicMock() if message_exists else None
        elif "user_feedback" in sql and "message_id" in sql:
            result.scalar_one_or_none.return_value = MagicMock() if has_existing_feedback else None
        else:
            result.scalar_one_or_none.return_value = None

        return result

    mock_db = MagicMock()
    mock_db.execute = AsyncMock(side_effect=_execute)
    mock_db.add = MagicMock()
    mock_db.delete = AsyncMock()
    mock_db.commit = AsyncMock()
    return mock_db


@pytest.mark.asyncio
async def test_feedback_thumbs_up() -> None:
    """POST /api/feedback with 'up' should return 200 with correct payload."""
    from backend.database import get_db

    msg_id = str(uuid.uuid4())
    app.dependency_overrides[get_db] = lambda: _make_db()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/feedback", json={"message_id": msg_id, "feedback": "up"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["message_id"] == msg_id
    assert data["feedback"] == "up"
    assert "feedback_id" in data


@pytest.mark.asyncio
async def test_feedback_thumbs_down() -> None:
    """POST /api/feedback with 'down' should return 200."""
    from backend.database import get_db

    msg_id = str(uuid.uuid4())
    app.dependency_overrides[get_db] = lambda: _make_db()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/feedback", json={"message_id": msg_id, "feedback": "down"})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["feedback"] == "down"


@pytest.mark.asyncio
async def test_feedback_replaces_existing() -> None:
    """Submitting feedback twice should succeed (upsert behaviour)."""
    from backend.database import get_db

    msg_id = str(uuid.uuid4())
    app.dependency_overrides[get_db] = lambda: _make_db(has_existing_feedback=True)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/feedback", json={"message_id": msg_id, "feedback": "up"})

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_feedback_404_for_missing_message() -> None:
    """POST /api/feedback for a non-existent message should return 404."""
    from backend.database import get_db

    app.dependency_overrides[get_db] = lambda: _make_db(message_exists=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/feedback", json={"message_id": str(uuid.uuid4()), "feedback": "up"}
        )

    app.dependency_overrides.clear()
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_feedback_422_invalid_uuid() -> None:
    """POST /api/feedback with a non-UUID message_id should return 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/feedback", json={"message_id": "not-a-uuid", "feedback": "up"}
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_feedback_422_invalid_value() -> None:
    """POST /api/feedback with feedback value other than 'up'/'down' should return 422."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/feedback", json={"message_id": str(uuid.uuid4()), "feedback": "meh"}
        )
    assert response.status_code == 422
