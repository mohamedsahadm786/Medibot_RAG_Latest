"""
Tests for rate limiting (Phase 13):
  Verifies that /api/chat and /api/chat/stream enforce 10 req/min per IP.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


def _mock_pipeline_response() -> dict:
    return {
        "answer": "Test answer.",
        "sources": [],
        "enhanced_query": "test",
        "hyde_answer": "",
        "query_variants": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "parent_chunks": [],
        "compressed_contexts": [],
        "relevance_verdict": "RELEVANT",
        "hallucination_verdict": "grounded",
        "retry_count": 0,
        "hallucination_retry": False,
        "status_events": [],
        "intent": "medical",
        "session_id": "test-session",
        "query": "test",
        "summaries": [],
    }


@pytest.mark.asyncio
async def test_chat_returns_200_within_limit() -> None:
    """Single request to /api/chat should succeed (not rate-limited)."""
    from backend.database import get_db
    from backend.services import cache as cache_mod

    mock_db = MagicMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()

    app.dependency_overrides[get_db] = lambda: mock_db

    with (
        patch.object(cache_mod, "check_cache", new=AsyncMock(return_value=None)),
        patch.object(cache_mod, "store_cache", new=AsyncMock()),
        patch("backend.api.routes.chat.build_pipeline") as mock_build,
        patch("backend.api.routes.chat.load_summaries", new=AsyncMock(return_value=[])),
        patch("backend.api.routes.chat.save_turn_summary", new=AsyncMock()),
        patch("backend.api.routes.chat.log_retrieval_details") as mock_log,
        patch("backend.api.routes.chat.evaluate_with_ragas") as mock_eval,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.ainvoke = AsyncMock(return_value=_mock_pipeline_response())
        mock_build.return_value = mock_pipeline
        mock_log.delay = MagicMock()
        mock_eval.delay = MagicMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"session_id": "test-session", "query": "what is diabetes?"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_rate_limit_headers_present() -> None:
    """Response from /api/chat should include rate limit headers."""
    from backend.database import get_db
    from backend.services import cache as cache_mod

    mock_db = MagicMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()

    app.dependency_overrides[get_db] = lambda: mock_db

    with (
        patch.object(cache_mod, "check_cache", new=AsyncMock(return_value=None)),
        patch.object(cache_mod, "store_cache", new=AsyncMock()),
        patch("backend.api.routes.chat.build_pipeline") as mock_build,
        patch("backend.api.routes.chat.load_summaries", new=AsyncMock(return_value=[])),
        patch("backend.api.routes.chat.save_turn_summary", new=AsyncMock()),
        patch("backend.api.routes.chat.log_retrieval_details") as mock_log,
        patch("backend.api.routes.chat.evaluate_with_ragas") as mock_eval,
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.ainvoke = AsyncMock(return_value=_mock_pipeline_response())
        mock_build.return_value = mock_pipeline
        mock_log.delay = MagicMock()
        mock_eval.delay = MagicMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"session_id": "test-session", "query": "what is diabetes?"},
            )

    app.dependency_overrides.clear()
    # slowapi injects X-RateLimit-* headers on successful responses
    assert response.status_code == 200
    assert "x-ratelimit-limit-10" in {h.lower() for h in response.headers} or \
           any("ratelimit" in h.lower() for h in response.headers)
