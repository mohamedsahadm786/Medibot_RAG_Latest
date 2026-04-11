"""
Tests for POST /api/chat endpoint (Phase 3).
Mocks retriever and generator so no real Pinecone/OpenAI calls are made.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.database import get_db
from backend.main import app


@pytest.fixture
def mock_db():
    """Async DB session mock that accepts add/commit calls."""
    session = AsyncMock()
    session.add = MagicMock()       # db.add() is synchronous in SQLAlchemy
    session.commit = AsyncMock()    # db.commit() is async

    async def override():
        yield session

    app.dependency_overrides[get_db] = override
    yield session
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_returns_answer(mock_db) -> None:
    """POST /api/chat should return answer and sources when retrieval succeeds."""
    fake_chunks = [
        {
            "chunk_id": "aaaaaaaa-0000-0000-0000-000000000001",
            "content": "Diabetes mellitus is a chronic metabolic disease...",
            "page_number": 42,
            "section_heading": "Diabetes",
            "source_pdf": "Medical_book.pdf",
        }
    ]
    fake_generation = {
        "answer": "Diabetes mellitus is a chronic condition characterized by high blood sugar.",
        "sources": [
            {
                "chunk_id": "aaaaaaaa-0000-0000-0000-000000000001",
                "page_number": 42,
                "section_heading": "Diabetes",
                "excerpt": "Diabetes mellitus is a chronic metabolic disease...",
                "source_pdf": "Medical_book.pdf",
            }
        ],
    }

    with patch("backend.api.routes.chat.retrieve", new=AsyncMock(return_value=fake_chunks)), \
         patch("backend.api.routes.chat.generate", new=AsyncMock(return_value=fake_generation)):

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "session_id": "11111111-1111-1111-1111-111111111111",
                    "query": "What is diabetes?",
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "message_id" in data
    assert data["session_id"] == "11111111-1111-1111-1111-111111111111"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["page_number"] == 42


@pytest.mark.asyncio
async def test_chat_empty_retrieval_returns_no_info_message(mock_db) -> None:
    """When retrieval returns no chunks, answer should contain a fallback message."""
    fake_generation = {
        "answer": "I could not find relevant information in the medical reference.",
        "sources": [],
    }

    with patch("backend.api.routes.chat.retrieve", new=AsyncMock(return_value=[])), \
         patch("backend.api.routes.chat.generate", new=AsyncMock(return_value=fake_generation)):

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "session_id": "22222222-2222-2222-2222-222222222222",
                    "query": "What is an unknown condition XYZ?",
                },
            )

    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []


@pytest.mark.asyncio
async def test_chat_missing_query_returns_422(mock_db) -> None:
    """Request without query field should return HTTP 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            json={"session_id": "33333333-3333-3333-3333-333333333333"},
        )
    assert response.status_code == 422
