import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import get_db

router = APIRouter()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Check liveness of the API and connectivity to PostgreSQL and Redis.

    Returns:
        JSON with status, database, and redis fields.
    """
    # Check PostgreSQL
    db_status = "connected"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    # Check Redis
    redis_status = "connected"
    try:
        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
    except Exception:
        redis_status = "disconnected"

    overall = "healthy" if db_status == "connected" and redis_status == "connected" else "degraded"

    return {
        "status": overall,
        "database": db_status,
        "redis": redis_status,
    }
