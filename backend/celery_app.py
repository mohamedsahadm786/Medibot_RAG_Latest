"""
MediBot v2 — Celery Application (Phase 10)

Single Celery application shared by all background tasks.

Broker and result backend both use Redis DB 1 so they stay separate from the
semantic cache (which uses DB 0).
"""

from celery import Celery

from backend.config import settings


def _celery_redis_url(redis_url: str) -> str:
    """
    Derive the Celery Redis URL from the app's redis_url by switching to DB 1.

    Examples:
        "redis://localhost:6379/0"  →  "redis://localhost:6379/1"
        "redis://redis:6379/0"      →  "redis://redis:6379/1"
    """
    base, _, _ = redis_url.rpartition("/")
    return f"{base}/1"


_broker_url = _celery_redis_url(settings.redis_url)

celery_app = Celery(
    "medibot",
    broker=_broker_url,
    backend=_broker_url,
    include=["backend.services.evaluation"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Fetch one task at a time — prevents a slow RAGAS eval from starving
    # the log task on a single-worker deployment.
    worker_prefetch_multiplier=1,
    # Acknowledge only after the task completes (safe re-delivery on crash).
    task_acks_late=True,
)
