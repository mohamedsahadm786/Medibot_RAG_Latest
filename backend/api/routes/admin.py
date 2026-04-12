"""
MediBot v2 — Admin routes (Phase 13)

Operational endpoints:
  DELETE /admin/cache/clear  — flush semantic cache
  GET    /admin/stats        — aggregate usage stats
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import cast, Date, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.database import ChatMessage, RetrievalLog, UserFeedback
from backend.services.cache import clear_cache

logger = logging.getLogger(__name__)
router = APIRouter()


@router.delete("/admin/cache/clear")
async def admin_clear_cache() -> dict:
    """Flush all entries from the semantic cache."""
    count = await clear_cache()
    logger.info("Admin: cache cleared (%d entries)", count)
    return {"cleared_entries": count}


@router.get("/admin/stats")
async def admin_stats(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Return aggregate usage statistics for the admin panel.

    Includes total queries, avg latency, cache hit rate, average RAGAS
    scores across all evaluations, feedback ratio, and queries per day
    for the last 7 days.
    """
    # ── Total user messages ───────────────────────────────────────────────────
    total_q = await db.execute(
        select(func.count()).select_from(ChatMessage).where(ChatMessage.role == "user")
    )
    total_queries: int = total_q.scalar_one() or 0

    # ── Pipeline runs (retrieval logs) ────────────────────────────────────────
    pipeline_q = await db.execute(
        select(func.count()).select_from(RetrievalLog)
    )
    pipeline_runs: int = pipeline_q.scalar_one() or 0

    # Cache hits = queries that returned early (no retrieval log was created)
    cache_hits = max(total_queries - pipeline_runs, 0)
    cache_hit_rate = round(
        (cache_hits / total_queries * 100) if total_queries > 0 else 0.0, 1
    )

    # ── Average latency ───────────────────────────────────────────────────────
    lat_q = await db.execute(
        select(func.avg(RetrievalLog.latency_ms)).where(
            RetrievalLog.latency_ms.isnot(None)
        )
    )
    avg_latency_ms: float = round(float(lat_q.scalar_one() or 0.0), 1)

    # ── Average RAGAS scores across all logged evaluations ────────────────────
    ragas_q = await db.execute(
        select(RetrievalLog.ragas_scores).where(RetrievalLog.ragas_scores.isnot(None))
    )
    all_ragas_rows = ragas_q.scalars().all()

    avg_ragas: dict = {}
    if all_ragas_rows:
        accumulated: dict[str, list[float]] = defaultdict(list)
        for scores in all_ragas_rows:
            for metric, value in scores.items():
                try:
                    accumulated[metric].append(float(value))
                except (TypeError, ValueError):
                    pass
        avg_ragas = {
            metric: round(sum(vals) / len(vals), 4)
            for metric, vals in accumulated.items()
        }

    # ── Feedback counts ───────────────────────────────────────────────────────
    fb_q = await db.execute(
        select(UserFeedback.feedback, func.count()).group_by(UserFeedback.feedback)
    )
    feedback_counts: dict = {row[0]: row[1] for row in fb_q.all()}
    thumbs_up = feedback_counts.get("up", 0)
    thumbs_down = feedback_counts.get("down", 0)

    # ── Queries per day — last 7 days ─────────────────────────────────────────
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    daily_q = await db.execute(
        select(
            cast(ChatMessage.created_at, Date).label("day"),
            func.count().label("count"),
        )
        .where(
            ChatMessage.role == "user",
            ChatMessage.created_at >= seven_days_ago,
        )
        .group_by(cast(ChatMessage.created_at, Date))
        .order_by(cast(ChatMessage.created_at, Date))
    )
    queries_per_day = [
        {"date": str(row.day), "count": row.count}
        for row in daily_q.all()
    ]

    return {
        "total_queries": total_queries,
        "avg_latency_ms": avg_latency_ms,
        "cache_hit_rate": cache_hit_rate,
        "avg_ragas_scores": avg_ragas,
        "feedback": {"up": thumbs_up, "down": thumbs_down},
        "queries_per_day": queries_per_day,
    }
