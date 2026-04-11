"""
Tests for POST /api/chat endpoint (Phase 7).
Mocks the LangGraph pipeline and memory service so no real calls are made.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.database import get_db
from backend.main import app
from backend.services.rag_pipeline import GraphState

# Patch memory and cache functions for all tests in this module
pytestmark = pytest.mark.usefixtures("mock_memory", "mock_cache")


@pytest.fixture
def mock_memory():
    """Patch load_summaries and save_turn_summary for all chat endpoint tests."""
    with patch(
        "backend.api.routes.chat.load_summaries",
        new=AsyncMock(return_value=[]),
    ), patch(
        "backend.api.routes.chat.save_turn_summary",
        new=AsyncMock(return_value=None),
    ):
        yield


@pytest.fixture
def mock_cache():
    """Patch check_cache (miss) and store_cache for all chat endpoint tests."""
    with patch(
        "backend.api.routes.chat.check_cache",
        new=AsyncMock(return_value=None),
    ), patch(
        "backend.api.routes.chat.store_cache",
        new=AsyncMock(return_value=None),
    ):
        yield


@pytest.fixture
def mock_db():
    """Async DB session mock that accepts add/commit calls."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    async def override():
        yield session

    app.dependency_overrides[get_db] = override
    yield session
    app.dependency_overrides.clear()


def _make_pipeline_result(answer: str, sources: list[dict]) -> GraphState:
    """Return a minimal final GraphState for the pipeline mock."""
    return {
        "query": "test",
        "session_id": "test",
        "summaries": [],
        "enhanced_query": "enhanced test query",
        "hyde_answer": "",
        "query_variants": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "parent_chunks": [],
        "compressed_contexts": [],
        "relevance_verdict": "RELEVANT",
        "answer": answer,
        "sources": sources,
        "hallucination_verdict": "grounded",
        "retry_count": 0,
        "hallucination_retry": False,
        "status_events": [],
    }


@pytest.mark.asyncio
async def test_chat_returns_answer(mock_db) -> None:
    """POST /api/chat should return answer and sources when pipeline succeeds."""
    fake_sources = [
        {
            "chunk_id": "aaaaaaaa-0000-0000-0000-000000000001",
            "page_number": 42,
            "section_heading": "Diabetes",
            "excerpt": "Diabetes mellitus is a chronic metabolic disease...",
            "source_pdf": "Medical_book.pdf",
        }
    ]
    pipeline_result = _make_pipeline_result(
        answer="Diabetes mellitus is a chronic condition characterized by high blood sugar.",
        sources=fake_sources,
    )

    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=pipeline_result)

    with patch(
        "backend.api.routes.chat.build_pipeline",
        return_value=mock_compiled,
    ):
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
    """When pipeline returns no sources, response is still 200 with empty sources."""
    pipeline_result = _make_pipeline_result(
        answer="I could not find relevant information in the medical reference.",
        sources=[],
    )

    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=pipeline_result)

    with patch("backend.api.routes.chat.build_pipeline", return_value=mock_compiled):
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


@pytest.mark.asyncio
async def test_chat_passes_summaries_to_pipeline(mock_db) -> None:
    """Summaries loaded from memory should be forwarded to the pipeline state."""
    pipeline_result = _make_pipeline_result(answer="Answer.", sources=[])
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=pipeline_result)

    captured_state: list[dict] = []

    async def capturing_ainvoke(state: dict) -> GraphState:
        captured_state.append(state)
        return pipeline_result

    mock_compiled.ainvoke = capturing_ainvoke

    with patch("backend.api.routes.chat.build_pipeline", return_value=mock_compiled), \
         patch(
             "backend.api.routes.chat.load_summaries",
             new=AsyncMock(return_value=["User asked about aspirin; assistant explained it."]),
         ), \
         patch("backend.api.routes.chat.save_turn_summary", new=AsyncMock()):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/chat",
                json={
                    "session_id": "55555555-5555-5555-5555-555555555555",
                    "query": "Is it safe to take it daily?",
                },
            )

    assert len(captured_state) == 1
    assert captured_state[0]["summaries"] == [
        "User asked about aspirin; assistant explained it."
    ]


@pytest.mark.asyncio
async def test_chat_pipeline_error_returns_500(mock_db) -> None:
    """If the RAG pipeline raises, the endpoint returns HTTP 500."""
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(side_effect=RuntimeError("pipeline down"))

    with patch("backend.api.routes.chat.build_pipeline", return_value=mock_compiled):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "session_id": "44444444-4444-4444-4444-444444444444",
                    "query": "What is aspirin?",
                },
            )

    assert response.status_code == 500
