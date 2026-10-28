from conftest import SAMPLE_ITEMS

from radar.rag.index import index_items
from radar.rag.retrieve import fts_query, hybrid_search, rrf


def test_fts_query_is_safe():
    assert fts_query('What about "RAG" AND NOT-this?') == (
        '"what" OR "about" OR "rag" OR "and" OR "not" OR "this"'
    )
    assert fts_query("?!") == ""


def test_rrf_rewards_items_in_both_lists():
    scores = rrf([[1, 2, 3], [3, 4]])
    assert max(scores, key=scores.get) == 3  # ranked by both searches
    assert scores[1] > scores[2]


def test_hybrid_search_finds_exact_name(filled_ctx):
    passages = hybrid_search(
        filled_ctx.conn, filled_ctx.embeddings, filled_ctx.vectors, "Medusa", k=3
    )
    assert passages[0].item_id == "2401.10774"


def test_one_passage_per_item_and_k(filled_ctx):
    passages = hybrid_search(
        filled_ctx.conn, filled_ctx.embeddings, filled_ctx.vectors, "faster inference decoding", k=2
    )
    ids = [p.item_id for p in passages]
    assert len(ids) == 2 and len(set(ids)) == 2


def test_since_filter(filled_ctx):
    passages = hybrid_search(
        filled_ctx.conn,
        filled_ctx.embeddings,
        filled_ctx.vectors,
        "retrieval",
        k=5,
        since="2026-01-01",
    )
    assert "2005.11401" not in [p.item_id for p in passages]


def test_index_skips_duplicates(ctx):
    assert index_items(ctx.conn, ctx.embeddings, ctx.vectors, SAMPLE_ITEMS) == 4
    assert index_items(ctx.conn, ctx.embeddings, ctx.vectors, SAMPLE_ITEMS) == 0
    counts = [
        ctx.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("items", "chunks", "chunks_fts", "chunks_vec")
    ]
    assert counts == [4, 4, 4, 4]


def test_empty_store_returns_nothing(ctx):
    assert hybrid_search(ctx.conn, ctx.embeddings, ctx.vectors, "anything") == []
