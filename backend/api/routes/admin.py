"""
MediBot v2 — Admin routes (Phase 9)

Operational endpoints for cache management.
"""

import logging

from fastapi import APIRouter

from backend.services.cache import clear_cache

logger = logging.getLogger(__name__)
router = APIRouter()


@router.delete("/admin/cache/clear")
async def admin_clear_cache() -> dict:
    """
    Flush all semantic cache entries from Redis.

    Returns the number of entries that were removed.
    """
    count = await clear_cache()
    logger.info("Admin: cache cleared (%d entries)", count)
    return {"cleared_entries": count}
