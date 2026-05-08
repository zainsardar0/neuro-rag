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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Returns the application settings instance."""
    return Settings()