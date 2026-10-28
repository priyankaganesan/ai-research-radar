"""Writing items into the store: item row, chunk rows, keyword index and vector index."""

import sqlite3

from langchain_core.embeddings import Embeddings

from radar.rag.chunking import abstract_chunk
from radar.rag.vector_index import SqliteVecIndex, VectorIndex
from radar.schema import Item
from radar.store import items as item_store

EMBED_BATCH = 32


def index_items(
    conn: sqlite3.Connection,
    embeddings: Embeddings,
    vectors: VectorIndex,
    new_items: list[Item],
) -> int:
    """Add items not already in the store. Returns how many were added.

    Documents are embedded as-is (title + abstract). Only queries get an instruction prefix;
    see retrieve.py.

    Vectors always go into SQLite (`chunks_vec`), the source of truth. If another backend is
    active (LanceDB), they are copied there after the SQLite transaction commits; if that copy
    fails, `radar reindex --backend lance` rebuilds it from SQLite.
    """
    store_vectors = SqliteVecIndex(conn)
    mirror = None if isinstance(vectors, SqliteVecIndex) else vectors
    todo = [i for i in new_items if not item_store.exists(conn, i.id)]
    for start in range(0, len(todo), EMBED_BATCH):
        batch = todo[start : start + EMBED_BATCH]
        chunks = [abstract_chunk(i) for i in batch]
        vecs = embeddings.embed_documents([c.text for c in chunks])  # one Ollama call per batch
        with conn:  # one transaction per batch: all tables updated together, or none
            chunk_ids = []
            for item, chunk in zip(batch, chunks, strict=True):
                item_store.upsert_item(conn, item)
                cur = conn.execute(
                    "INSERT INTO chunks (item_id, kind, section, text, token_count)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (chunk.item_id, chunk.kind, chunk.section, chunk.text, chunk.token_count),
                )
                chunk_ids.append(cur.lastrowid)
                conn.execute(
                    "INSERT INTO chunks_fts (rowid, title, text) VALUES (?, ?, ?)",
                    (cur.lastrowid, item.title, chunk.text),
                )
            store_vectors.add(chunk_ids, vecs)
        if mirror is not None:
            mirror.add(chunk_ids, vecs)
    return len(todo)
