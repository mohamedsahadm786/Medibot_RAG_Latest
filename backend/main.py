import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.config import settings
from backend.core.metrics import request_count, request_latency
from backend.database import engine
from backend.api.routes import health, chat, admin, feedback


def _rate_limit_storage_uri() -> str:
    """Use Redis DB 1 for rate limiting, separate from app cache on DB 0."""
    base = settings.redis_url.rsplit("/", 1)[0]
    return f"{base}/1"


limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_rate_limit_storage_uri(),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    yield
    await engine.dispose()


app = FastAPI(
    title="MediBot v2 API",
    description="Production-grade medical RAG chatbot powered by a 700+ page medical knowledge base.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Attach limiter to app state (required by slowapi) ─────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a user-friendly 429 response when rate limit is exceeded."""
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded. Please wait before sending another message."},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    """Record request count and latency for every HTTP request."""
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start

    if request.url.path != "/metrics":
        request_count.labels(
            endpoint=request.url.path,
            status=str(response.status_code),
        ).inc()
        request_latency.labels(endpoint=request.url.path).observe(latency)

    return response


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(admin.router, prefix="/api", tags=["admin"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])
