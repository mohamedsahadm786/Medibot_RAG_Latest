"""
Tests for POST /api/chat/stream SSE endpoint (Phase 8).
Mocks the LangGraph pipeline and memory service so no real calls are made.
"""

import json
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
    """Patch load_summaries and save_turn_summary for all stream endpoint tests."""
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
    """Patch check_cache (miss) and store_cache for all stream endpoint tests."""
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
    """Async DB session mock."""
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


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE response body into a list of event dicts."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


@pytest.mark.asyncio
async def test_stream_returns_sse_content_type(mock_db) -> None:
    """POST /api/chat/stream should respond with text/event-stream content type."""
    pipeline_result = _make_pipeline_result(answer="Diabetes is a metabolic disease.", sources=[])
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=pipeline_result)

    with patch("backend.api.routes.chat.build_pipeline", return_value=mock_compiled):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat/stream",
                json={
                    "session_id": "11111111-1111-1111-1111-111111111111",
                    "query": "What is diabetes?",
                },
            )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_stream_contains_token_and_done_events(mock_db) -> None:
    """Stream must contain at least one token event and exactly one done event."""
    pipeline_result = _make_pipeline_result(answer="Diabetes is a metabolic disease.", sources=[])
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=pipeline_result)

    with patch("backend.api.routes.chat.build_pipeline", return_value=mock_compiled):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat/stream",
                json={
                    "session_id": "11111111-1111-1111-1111-111111111111",
                    "query": "What is diabetes?",
                },
            )

    events = _parse_sse(response.text)
    event_types = [e.get("type") for e in events]
    assert "token" in event_types
    assert event_types.count("done") == 1


@pytest.mark.asyncio
async def test_stream_tokens_reconstruct_answer(mock_db) -> None:
    """Concatenating all token event contents should reproduce the full answer."""
    answer = "Hypertension is high blood pressure."
    pipeline_result = _make_pipeline_result(answer=answer, sources=[])
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=pipeline_result)

    with patch("backend.api.routes.chat.build_pipeline", return_value=mock_compiled):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat/stream",
                json={
                    "session_id": "22222222-2222-2222-2222-222222222222",
                    "query": "What is hypertension?",
                },
            )

    events = _parse_sse(response.text)
    tokens = [e["content"] for e in events if e.get("type") == "token"]
    assert "".join(tokens) == answer


@pytest.mark.asyncio
async def test_stream_sources_event_present(mock_db) -> None:
    """A 'sources' event with correct citation data must be emitted before done."""
    fake_sources = [
        {
            "chunk_id": "aaaaaaaa-0000-0000-0000-000000000001",
            "page_number": 10,
            "section_heading": "Cardiology",
            "excerpt": "The heart pumps blood...",
            "source_pdf": "Medical_book.pdf",
        }
    ]
    pipeline_result = _make_pipeline_result(
        answer="The heart pumps blood.",
        sources=fake_sources,
    )
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=pipeline_result)

    with patch("backend.api.routes.chat.build_pipeline", return_value=mock_compiled):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat/stream",
                json={
                    "session_id": "33333333-3333-3333-3333-333333333333",
                    "query": "What does the heart do?",
                },
            )

    events = _parse_sse(response.text)
    sources_events = [e for e in events if e.get("type") == "sources"]
    assert len(sources_events) == 1
    assert sources_events[0]["data"][0]["page_number"] == 10


@pytest.mark.asyncio
async def test_stream_done_event_contains_ids(mock_db) -> None:
    """The done event must include message_id and session_id."""
    pipeline_result = _make_pipeline_result(answer="Answer.", sources=[])
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=pipeline_result)

    with patch("backend.api.routes.chat.build_pipeline", return_value=mock_compiled):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat/stream",
                json={
                    "session_id": "55555555-5555-5555-5555-555555555555",
                    "query": "Test query.",
                },
            )

    events = _parse_sse(response.text)
    done_events = [e for e in events if e.get("type") == "done"]
    assert len(done_events) == 1
    done = done_events[0]
    assert "message_id" in done
    assert done["session_id"] == "55555555-5555-5555-5555-555555555555"


@pytest.mark.asyncio
async def test_stream_pipeline_error_sends_error_event(mock_db) -> None:
    """If the pipeline raises, the stream should contain an error event (not a 500)."""
    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(side_effect=RuntimeError("pipeline down"))

    with patch("backend.api.routes.chat.build_pipeline", return_value=mock_compiled):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/chat/stream",
                json={
                    "session_id": "44444444-4444-4444-4444-444444444444",
                    "query": "What is aspirin?",
                },
            )

    assert response.status_code == 200  # SSE connection itself succeeded
    events = _parse_sse(response.text)
    error_events = [e for e in events if e.get("type") == "error"]
    assert len(error_events) == 1
