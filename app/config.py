"""Centralised application settings — driven entirely by environment variables / .env."""
from __future__ import annotations

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pydantic v1 fallback
    from pydantic import BaseSettings  # type: ignore
    SettingsConfigDict = None  # type: ignore


class Settings(BaseSettings):
    # --- LLM ---
    gemini_api_key: str = ""
    google_api_key: str = ""
    model_name: str = "gemini-1.5-flash"

    # --- Embeddings ---
    embedding_model: str = "models/embedding-001"
    st_model_name: str = "all-MiniLM-L6-v2"

    # --- Retrieval ---
    top_candidates: int = 5
    llm_analyze_top_n: int = 3

    # --- Storage ---
    upload_folder: str = "uploads/"
    db_path: str = "candidates.db"

    # --- Observability: LangSmith ---
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "CVInsight"

    # --- Observability: Langfuse ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"


settings = Settings()
