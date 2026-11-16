"""Writing items into the store: item row, chunk rows, keyword index and vector index."""

import sqlite3

from langchain_core.embeddings import Embeddings

from radar.rag.chunking import Chunk, abstract_chunk
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


def replace_section_chunks(
    conn: sqlite3.Connection,
    embeddings: Embeddings,
    vectors: VectorIndex,
    item: Item,
    chunks: list[Chunk],
) -> list[int]:
    """M5: store an item's full-text chunks, replacing any it had before. Returns their IDs.

    The abstract chunk stays: it's still the best one-paragraph description of the paper. Old
    section chunks are removed from all three places (chunks, FTS, vectors) first, so fetching
    again (e.g. with new chunk sizes) doesn't leave duplicates behind.
    """
    store_vectors = SqliteVecIndex(conn)
    mirror = None if isinstance(vectors, SqliteVecIndex) else vectors
    vecs: list[list[float]] = []
    for start in range(0, len(chunks), EMBED_BATCH):  # embed before writing: slow, can fail
        vecs += embeddings.embed_documents([c.text for c in chunks[start : start + EMBED_BATCH]])
    old = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM chunks WHERE item_id = ? AND kind = 'section'", (item.id,)
        )
    ]
    with conn:
        if old:
            store_vectors.delete(old)
            conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in old])
            conn.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in old])
        ids = []
        for c in chunks:
            cur = conn.execute(
                "INSERT INTO chunks (item_id, kind, section, text, token_count)"
                " VALUES (?, 'section', ?, ?, ?)",
                (item.id, c.section, c.text, c.token_count),
            )
            ids.append(cur.lastrowid)
            conn.execute(
                "INSERT INTO chunks_fts (rowid, title, text) VALUES (?, ?, ?)",
                (cur.lastrowid, item.title, c.text),
            )
        store_vectors.add(ids, vecs)
    if mirror is not None:
        if old:
            mirror.delete(old)
        mirror.add(ids, vecs)
    return ids
