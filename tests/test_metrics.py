"""
Tests for GET /metrics endpoint and Prometheus metrics registry (Phase 11).
"""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_200() -> None:
    """GET /metrics should return HTTP 200."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_metrics_endpoint_content_type_is_text_plain() -> None:
    """GET /metrics content-type should be text/plain (Prometheus exposition format)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert "text/plain" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_metrics_contains_request_counter() -> None:
    """medibot_requests_total should appear in the metrics output."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert "medibot_requests_total" in response.text


@pytest.mark.asyncio
async def test_metrics_contains_cache_counters() -> None:
    """Cache hit and miss counters should be present in the output."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert "medibot_cache_hits_total" in response.text
    assert "medibot_cache_misses_total" in response.text


@pytest.mark.asyncio
async def test_metrics_contains_latency_histogram() -> None:
    """Request latency histogram should be present."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert "medibot_request_latency_seconds" in response.text


@pytest.mark.asyncio
async def test_metrics_contains_ragas_gauges() -> None:
    """RAGAS quality gauges should be present."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert "medibot_ragas_faithfulness" in response.text
    assert "medibot_ragas_relevancy" in response.text


@pytest.mark.asyncio
async def test_metrics_contains_user_feedback_counter() -> None:
    """User feedback counter should be present."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/metrics")
    assert "medibot_user_feedback_total" in response.text


@pytest.mark.asyncio
async def test_request_increments_counter() -> None:
    """Making a request should increment medibot_requests_total in /metrics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Make a request that will be counted (health endpoint — no external deps)
        await client.get("/api/health")
        response = await client.get("/metrics")

    # The /api/health path should appear with a status label
    assert "/api/health" in response.text
