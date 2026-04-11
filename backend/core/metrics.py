"""
MediBot v2 — Prometheus Metrics Registry (Phase 11)

All application metrics are defined here so they can be imported by any
module that needs to increment them.  prometheus_client uses a global
registry, so each metric must be registered exactly once.

Usage:
    from backend.core.metrics import cache_hits, request_count
    cache_hits.inc()
    request_count.labels(endpoint="/api/chat", status="200").inc()
"""

from prometheus_client import Counter, Gauge, Histogram

# ── HTTP request metrics ───────────────────────────────────────────────────────

request_count = Counter(
    "medibot_requests_total",
    "Total number of HTTP requests processed.",
    ["endpoint", "status"],
)

request_latency = Histogram(
    "medibot_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# ── LLM token usage ────────────────────────────────────────────────────────────

tokens_used_total = Counter(
    "medibot_tokens_used_total",
    "Total LLM tokens consumed.",
    ["model", "step"],
)

# ── Semantic cache ─────────────────────────────────────────────────────────────

cache_hits_total = Counter(
    "medibot_cache_hits_total",
    "Number of semantic cache hits (pipeline skipped).",
)

cache_misses_total = Counter(
    "medibot_cache_misses_total",
    "Number of semantic cache misses (pipeline ran).",
)

# ── RAGAS quality gauges ───────────────────────────────────────────────────────
# Updated by the Celery evaluation worker via a Pushgateway or periodic refresh.
# Start at 0; will reflect real averages once evaluations accumulate.

ragas_faithfulness = Gauge(
    "medibot_ragas_faithfulness",
    "Latest average RAGAS faithfulness score (0–1).",
)

ragas_relevancy = Gauge(
    "medibot_ragas_relevancy",
    "Latest average RAGAS answer relevancy score (0–1).",
)

# ── User feedback ──────────────────────────────────────────────────────────────

user_feedback_total = Counter(
    "medibot_user_feedback_total",
    "User thumbs-up / thumbs-down votes.",
    ["feedback_type"],   # "up" | "down"
)

# ── Session activity ───────────────────────────────────────────────────────────

active_sessions = Gauge(
    "medibot_active_sessions",
    "Approximate number of unique sessions active in the last hour.",
)
