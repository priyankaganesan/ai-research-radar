"""Settings loaded from the environment and `.env`."""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from radar.sources.hackernews import DEFAULT_QUERIES
from radar.sources.lab_blogs import DEFAULT_FEEDS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_host: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3.5:9b"
    embed_model: str = "qwen3-embedding:0.6b"
    db_path: Path = Path("data/radar.db")
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    semantic_scholar_api_key: str = ""
    vector_backend: Literal["sqlite", "lance"] = "sqlite"
    lance_path: Path = Path("data/lance")
    lance_nprobes: int = 20  # IVF partitions scanned per search (only once an ANN index exists)
    lance_refine_factor: int = 0  # 0 = off; N = re-rank k*N candidates with exact distances
    arxiv_categories: str = "cs.AI,cs.CL,cs.LG"

    # Other sources (M3)
    # Lab blogs as JSON, e.g. LAB_FEEDS='{"OpenAI": "https://openai.com/news/rss.xml"}'
    lab_feeds: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_FEEDS))
    hn_queries: str = ",".join(DEFAULT_QUERIES)  # Hacker News search terms, comma-separated
    hn_min_points: int = 50

    # LangSmith reads its own LANGSMITH_* variables; these are here so `radar config` can show them.
    langsmith_tracing: bool = False
    langsmith_project: str = "ai-research-radar"

    # Q&A loop
    max_tool_steps: int = 5
    # Thinking mode for the reason step: shows Qwen's reasoning in --steps, but each step is slower.
    reason_thinking: bool = False
    retrieval_k: int = 5


@lru_cache
def get_settings() -> Settings:
    # Also copy .env into the process environment, so libraries that read os.environ directly
    # (LangSmith tracing reads LANGSMITH_*) see the same values. Real env vars win.
    load_dotenv(".env", override=False)
    _disable_tracing_without_key()
    return Settings()


def _disable_tracing_without_key() -> None:
    """LangSmith tracing with no API key just fails noisily on every run, so turn it off."""
    on = os.environ.get("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")
    if on and not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "false"
        print(
            "Note: LangSmith tracing is off because LANGSMITH_API_KEY isn't set (see .env).",
            file=sys.stderr,
        )
