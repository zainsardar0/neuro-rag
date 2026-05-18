from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # App
    app_name: str = Field(default="NeuroRAG")
    app_env: str = Field(default="development")
    log_level: str = Field(default="DEBUG")

    # Groq LLM
    groq_api_key: str = Field(..., description="Groq API key")

    # ChromaDB
    chroma_db_path: str = Field(default="./data/processed/chroma")

    # Redis Cache — V2 Phase 5
    redis_url: str = Field(default="", description="Redis URL for caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_enabled: bool = Field(default=True, description="Enable/disable caching")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Returns the application settings instance."""
    return Settings()