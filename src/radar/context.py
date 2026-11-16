"""The shared dependencies every part of the agent needs, built once per process."""

import sqlite3
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from radar.config import Settings
from radar.rag.vector_index import SqliteVecIndex, VectorIndex


@dataclass
class AppContext:
    settings: Settings
    conn: sqlite3.Connection
    embeddings: Embeddings
    vectors: VectorIndex


def build_context(settings: Settings, embeddings: Embeddings | None = None) -> AppContext:
    from radar import tracing
    from radar.models import get_embeddings
    from radar.store.db import connect

    conn = connect(settings.db_path)
    # Traced calls write their local log rows through this connection (see radar/tracing.py).
    tracing.configure(conn if settings.tool_log else None, settings.tool_log_keep_days)
    return AppContext(
        settings=settings,
        conn=conn,
        embeddings=embeddings or get_embeddings(settings),
        vectors=make_vector_index(settings, conn),
    )


def make_vector_index(
    settings: Settings, conn: sqlite3.Connection, backend: str | None = None
) -> VectorIndex:
    """The vector index for `backend` (default: VECTOR_BACKEND)."""
    backend = backend or settings.vector_backend
    if backend == "sqlite":
        return SqliteVecIndex(conn)
    if backend == "lance":
        from radar.rag.lance_index import LanceIndex
        from radar.store.db import EMBED_DIM

        return LanceIndex(
            settings.lance_path,
            dim=EMBED_DIM,
            nprobes=settings.lance_nprobes,
            refine_factor=settings.lance_refine_factor or None,
        )
    raise ValueError(f"Unknown VECTOR_BACKEND {backend!r} (use sqlite or lance)")
