"""Hybrid retrieval: vector search + keyword search, merged with reciprocal rank fusion.

Why both: embeddings find text with similar *meaning* ("speeding up decoding" finds papers that
say "faster inference"), while keyword search (FTS5/BM25) finds exact *names* ("Medusa", "GRPO")
that embeddings often miss.
"""

import re
import sqlite3

from langchain_core.embeddings import Embeddings

from radar.rag.vector_index import VectorIndex
from radar.schema import Passage

CANDIDATES = 20  # taken from each search before merging
RRF_K = 60  # standard constant; dampens the gap between rank 1 and rank 2
QUERY_INSTRUCTION = "Instruct: Given a question about AI research, find relevant papers\nQuery: "


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: each word quoted, joined with OR.

    Without quoting, words like AND/NOT or characters like '-' and ':' are FTS5 syntax and can
    raise errors.
    """
    words = [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1]
    return " OR ".join(f'"{w}"' for w in words)


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion: score(doc) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def keyword_search(conn: sqlite3.Connection, query: str, limit: int) -> list[int]:
    q = fts_query(query)
    if not q:
        return []
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
        (q, limit),
    ).fetchall()
    return [r[0] for r in rows]


def hybrid_search(
    conn: sqlite3.Connection,
    embeddings: Embeddings,
    vectors: VectorIndex,
    query: str,
    k: int = 5,
    kind: str | None = None,
    since: str | None = None,
) -> list[Passage]:
    """Top-k passages for a query, at most one per item, best first."""
    qvec = embeddings.embed_query(QUERY_INSTRUCTION + query)
    # Filters are applied after the search, so ask for extra candidates when filtering.
    extra = 4 if (kind or since) else 1
    by_vector = [cid for cid, _ in vectors.search(qvec, CANDIDATES * extra)]
    by_keyword = keyword_search(conn, query, CANDIDATES * extra)
    scores = rrf([by_vector, by_keyword])
    if not scores:
        return []

    placeholders = ",".join("?" * len(scores))
    rows = conn.execute(
        f"""
        SELECT c.id, c.item_id, c.text, i.title, i.url, i.published, i.kind
        FROM chunks c JOIN items i ON i.id = c.item_id
        WHERE c.id IN ({placeholders})
        """,
        list(scores),
    ).fetchall()

    best: dict[str, Passage] = {}
    for r in sorted(rows, key=lambda r: scores[r["id"]], reverse=True):
        if kind and r["kind"] != kind:
            continue
        if since and r["published"] < since:
            continue
        if r["item_id"] in best:
            continue  # keep only the best chunk per item
        best[r["item_id"]] = Passage(
            item_id=r["item_id"],
            title=r["title"],
            url=r["url"],
            published=r["published"],
            text=r["text"],
            score=round(scores[r["id"]], 5),
        )
        if len(best) == k:
            break
    return list(best.values())
