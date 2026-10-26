"""Settings loaded from the environment and `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_host: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3.5:9b"
    embed_model: str = "qwen3-embedding:0.6b"
    db_path: Path = Path("data/radar.db")
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    semantic_scholar_api_key: str = ""

    # LangSmith reads its own LANGSMITH_* variables; these are here so `radar config` can show them.
    langsmith_tracing: bool = False
    langsmith_project: str = "ai-research-radar"

    # Q&A loop
    max_tool_steps: int = 5
    retrieval_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
