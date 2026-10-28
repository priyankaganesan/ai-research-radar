"""The LanceDB backend: same results as sqlite-vec, SQLite stays the source of truth."""

import random

import pytest
from conftest import SAMPLE_ITEMS, FakeEmbeddings

from radar import services
from radar.context import AppContext, make_vector_index
from radar.rag.index import index_items
from radar.rag.lance_index import LanceIndex
from radar.rag.retrieve import hybrid_search
from radar.rag.vector_index import SqliteVecIndex
from radar.store.db import EMBED_DIM


@pytest.fixture
def lance_ctx(ctx, tmp_path) -> AppContext:
    lance = LanceIndex(tmp_path / "lance", dim=EMBED_DIM)
    return AppContext(
        settings=ctx.settings, conn=ctx.conn, embeddings=ctx.embeddings, vectors=lance
    )


def random_vectors(n: int, dim: int = EMBED_DIM, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


def test_add_search_delete_count(tmp_path):
    index = LanceIndex(tmp_path / "lance", dim=4)
    index.add([1, 2, 3], [[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]])
    assert index.count() == 3
    assert [cid for cid, _ in index.search([1, 0, 0, 0], k=2)] == [1, 3]
    index.delete([1])
    assert index.count() == 2
    assert index.search([1, 0, 0, 0], k=1)[0][0] == 3


def test_reopening_keeps_vectors(tmp_path):
    LanceIndex(tmp_path / "lance", dim=4).add([7], [[0, 0, 1, 0]])
    assert LanceIndex(tmp_path / "lance", dim=4).count() == 1


def test_indexing_writes_sqlite_and_lance(lance_ctx):
    index_items(lance_ctx.conn, lance_ctx.embeddings, lance_ctx.vectors, SAMPLE_ITEMS)
    assert SqliteVecIndex(lance_ctx.conn).count() == len(SAMPLE_ITEMS)  # source of truth
    assert lance_ctx.vectors.count() == len(SAMPLE_ITEMS)


def test_hybrid_search_matches_sqlite_backend(filled_ctx, lance_ctx):
    services.reindex(lance_ctx, "lance")
    query = "faster LLM decoding"
    on_sqlite = hybrid_search(filled_ctx.conn, FakeEmbeddings(), filled_ctx.vectors, query)
    on_lance = hybrid_search(lance_ctx.conn, FakeEmbeddings(), lance_ctx.vectors, query)
    # Items 3 and 4 tie on vector distance, so only the top two have a fixed order.
    assert [p.item_id for p in on_lance][:2] == [p.item_id for p in on_sqlite][:2]
    assert {p.item_id for p in on_lance} == {p.item_id for p in on_sqlite}


def test_reindex_lance_copies_from_sqlite(filled_ctx, settings, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "lance_path", tmp_path / "lance")
    result = services.reindex(filled_ctx, "lance")
    assert result == {"backend": "lance", "vectors": len(SAMPLE_ITEMS), "ann": None}
    services.reindex(filled_ctx, "lance")  # running it again replaces, never duplicates
    assert make_vector_index(settings, filled_ctx.conn, "lance").count() == len(SAMPLE_ITEMS)


def test_reindex_sqlite_reembeds(filled_ctx):
    assert services.reindex(filled_ctx, "sqlite")["vectors"] == len(SAMPLE_ITEMS)


def test_ann_needs_enough_rows(tmp_path):
    index = LanceIndex(tmp_path / "lance", dim=EMBED_DIM)
    index.add([1], random_vectors(1))
    with pytest.raises(ValueError, match="at least"):
        index.build_ann()


def test_ann_index_and_comparison(tmp_path):
    from radar.rag import compare

    vectors = random_vectors(300, dim=32)
    ids = list(range(1, 301))
    lance = LanceIndex(tmp_path / "lance", dim=32)
    lance.add(ids, vectors)
    settings = lance.build_ann(num_sub_vectors=4)
    assert settings["num_partitions"] == 1 and lance.has_ann()  # 300 rows: one partition

    class ExactIndex:
        """Plain-Python exact search, standing in for sqlite-vec at 32 dimensions."""

        def search(self, v, k):
            dist = [
                (sum((a - b) ** 2 for a, b in zip(v, w, strict=True)), i)
                for i, w in zip(ids, vectors, strict=True)
            ]
            return [(i, d) for d, i in sorted(dist)[:k]]

        def count(self):
            return len(ids)

        def disk_bytes(self):
            return None

    report = compare.compare(ExactIndex(), lance, random_vectors(10, dim=32, seed=1), k=5)
    by_name = {r.name: r for r in report["runs"]}
    assert by_name["sqlite-vec (exact)"].recall_vs_exact == 1.0
    # PQ's compression loses some neighbours; re-ranking with exact distances wins them back.
    plain = by_name["lancedb IVF-PQ nprobes=50"].recall_vs_exact
    refined = by_name["lancedb IVF-PQ nprobes=50 refine=10"].recall_vs_exact
    assert plain < 1.0 and refined >= 0.9
    assert "| Configuration |" in compare.to_markdown(report)


def test_compare_backends_end_to_end(filled_ctx, settings, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "lance_path", tmp_path / "lance")
    with pytest.raises(ValueError, match="reindex"):
        services.compare_backends(filled_ctx, out_dir=None)
    services.reindex(filled_ctx, "lance")
    report = services.compare_backends(filled_ctx, k=2, out_dir=tmp_path / "results")
    assert [r.recall_vs_exact for r in report["runs"]] == [1.0, 1.0]  # flat scan is exact too
    assert (tmp_path / "results").glob("backends-*.md")
