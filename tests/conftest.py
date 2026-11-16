import hashlib
import math
import re
from datetime import date
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from radar import tracing
from radar.config import Settings, get_settings
from radar.context import AppContext
from radar.rag.vector_index import SqliteVecIndex
from radar.schema import Item
from radar.store.db import EMBED_DIM, connect

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """Tests never read the real .env, never trace to LangSmith, and use a temp store."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "radar.db"))
    monkeypatch.setenv("CHECKPOINT_DB_PATH", str(tmp_path / "checkpoints.db"))
    get_settings.cache_clear()
    tracing.configure(None)  # the local tool-call log is off unless a test turns it on
    yield
    tracing.configure(None)
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return get_settings()


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words vectors: texts sharing words get similar vectors.

    Not semantic like a real model, but enough to test indexing and ranking without Ollama.
    """

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * EMBED_DIM
        for word in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % EMBED_DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text.split("Query: ", 1)[-1])


class FakeChat(GenericFakeChatModel):
    """Scripted chat model that also accepts bind_tools (it just returns itself)."""

    def bind_tools(self, tools, **kwargs):
        return self


def fake_chat(*replies) -> FakeChat:
    """Each reply is a string or an AIMessage (use AIMessage to script tool calls)."""
    msgs = [r if isinstance(r, AIMessage) else AIMessage(r) for r in replies]
    return FakeChat(messages=iter(msgs))


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])


@pytest.fixture
def ctx(settings) -> AppContext:
    conn = connect(settings.db_path)
    return AppContext(
        settings=settings, conn=conn, embeddings=FakeEmbeddings(), vectors=SqliteVecIndex(conn)
    )


def make_item(id: str, title: str, abstract: str, published: str = "2026-10-20") -> Item:
    return Item(
        id=id,
        source="arxiv",
        title=title,
        abstract=abstract,
        published=date.fromisoformat(published),
        url=f"https://arxiv.org/abs/{id}",
    )


SAMPLE_ITEMS = [
    make_item(
        "2401.10774",
        "Medusa: multiple decoding heads",
        "Extra decoding heads predict several future tokens to speed up LLM inference.",
    ),
    make_item(
        "2005.11401",
        "Retrieval-augmented generation",
        "A retriever fetches documents that a generator conditions on.",
        "2020-05-22",
    ),
    make_item(
        "2410.00001",
        "Knowledge conflicts in retrieval augmented models",
        "When retrieved context contradicts parametric knowledge, models often prefer "
        "what they memorized.",
    ),
    make_item(
        "2410.00002",
        "Speculative decoding survey",
        "Draft models propose tokens that a larger model verifies, for faster inference.",
    ),
]


@pytest.fixture
def filled_ctx(ctx) -> AppContext:
    from radar.rag.index import index_items

    index_items(ctx.conn, ctx.embeddings, ctx.vectors, SAMPLE_ITEMS)
    return ctx
