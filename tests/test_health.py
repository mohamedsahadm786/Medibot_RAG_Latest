from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_health_returns_healthy() -> None:
    """
    /api/health should return 200 with status=healthy when DB and Redis are reachable.
    Uses FastAPI dependency override for DB and patches redis.asyncio.
    """
    from backend.database import get_db
    from backend.main import app

    # Mock async DB session
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with patch("backend.api.routes.health.aioredis") as mock_aioredis:
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock(return_value=True)
        mock_redis_client.aclose = AsyncMock()
        mock_aioredis.from_url.return_value = mock_redis_client

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["redis"] == "connected"


@pytest.mark.asyncio
async def test_health_degraded_when_db_fails() -> None:
    """
    /api/health should return status=degraded when the database is unreachable.
    """
    from backend.database import get_db
    from backend.main import app

    async def failing_db():
        mock = AsyncMock()
        mock.execute = AsyncMock(side_effect=Exception("DB connection refused"))
        yield mock

    app.dependency_overrides[get_db] = failing_db

    with patch("backend.api.routes.health.aioredis") as mock_aioredis:
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock(return_value=True)
        mock_redis_client.aclose = AsyncMock()
        mock_aioredis.from_url.return_value = mock_redis_client

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["database"] == "disconnected"


@pytest.mark.asyncio
async def test_health_degraded_when_redis_fails() -> None:
    """
    /api/health should return status=degraded when Redis is unreachable.
    """
    from backend.database import get_db
    from backend.main import app

    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db

    with patch("backend.api.routes.health.aioredis") as mock_aioredis:
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock(side_effect=Exception("Redis connection refused"))
        mock_aioredis.from_url.return_value = mock_redis_client

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/health")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["redis"] == "disconnected"
