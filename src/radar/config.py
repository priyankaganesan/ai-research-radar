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
    # Local tool-call log (the tool_calls table, read by `radar calls`). Works with LangSmith off.
    tool_log: bool = True
    tool_log_keep_days: int = 30  # rows older than this are deleted when the app starts

    # Q&A loop
    max_tool_steps: int = 5
    # Thinking mode for the reason step: shows Qwen's reasoning in --steps, but each step is slower.
    reason_thinking: bool = False
    retrieval_k: int = 5

    # Full text (M5)
    # Fetch a paper's full text once this many answered questions have cited it (0 = never
    # automatically; the agent's fetch_full_text tool and `radar fulltext` still work).
    full_text_after_questions: int = 2
    full_text_chunk_tokens: int = 500
    full_text_overlap_tokens: int = 50
    # A cap on chunks per item, so one 80-page paper can't take minutes of embedding.
    full_text_max_chunks: int = 120
    full_text_retry_days: int = 7  # a failed fetch is tried again after this many days
    # Daily digest (M2)
    # Seed interests for the relevance score, until ratings (M4) refine the profile.
    interest_topics: str = (
        "LLM agents and tool use,retrieval-augmented generation,reasoning in language models,"
        "efficient LLM inference,evaluation of language models"
    )
    digest_window_days: int = 7  # candidates: items published in the last N days
    # Candidates need this much real text beyond the title (title-only HN stories can't be
    # summarized or judged).
    digest_min_text_chars: int = 200
    # Main picks: at least this relevance to the profile while enough items clear it
    # (scoring.slot_candidates). Show/Launch/Ask/Tell HN posts never take main slots.
    digest_min_relevance: float = 0.5
    digest_main: int = 6
    digest_reserved: int = 2  # early-buzz + lab-launch slots
    digest_dir: Path = Path("data/digests")
    digest_mode: Literal["single", "multi"] = "single"  # multi = M2.5 agents
    # M2.5: run the Verifier agent on every Q&A answer (one more model call per question).
    verify_answers: bool = False

    # Safety: when a digest is marked "needs review" (radar/safety.py review_reasons).
    review_min_support: float = 0.8  # Verifier support rate in the final drafts, below this
    review_max_cut_summaries: int = 2  # more summaries than this lost claims
    # Topic removals proposed by the Profile agent wait for `radar profile --approve`.
    profile_confirm_removals: bool = True

    # M1.6: load the Q&A tools from Radar's own MCP server instead of importing them.
    tools_via_mcp: bool = False


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
