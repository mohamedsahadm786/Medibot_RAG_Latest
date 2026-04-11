from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import engine
from backend.api.routes import health, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    # Startup: nothing to initialize at Phase 1 beyond DB engine
    yield
    # Shutdown: cleanly close all DB connections
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
    allow_origins=["*"],  # Restrict in production to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
