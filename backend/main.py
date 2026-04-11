import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.requests import Request

from backend.core.metrics import request_count, request_latency
from backend.database import engine
from backend.api.routes import health, chat, admin


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

    # Skip the /metrics endpoint itself to avoid self-referential noise
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
