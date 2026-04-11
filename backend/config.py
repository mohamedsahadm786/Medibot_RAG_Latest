from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://medibot:medibot@localhost:5433/medibot_v2"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # OpenAI
    openai_api_key: str = ""
    openai_model_primary: str = "gpt-4o"
    openai_model_aux: str = "gpt-4o-mini"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "medibot-v2"
    pinecone_namespace: str = "medibot"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "medibot-v2"

    # App
    app_env: str = "development"
    rate_limit_per_minute: int = 10
    semantic_cache_threshold: float = 0.95
    max_crag_retries: int = 1
    memory_window: int = 3


settings = Settings()
