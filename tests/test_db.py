import sqlite_vec

from radar.store.db import EMBED_DIM, connect, migrate


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_connect_creates_schema(tmp_path):
    conn = connect(tmp_path / "radar.db")
    assert {"papers", "chunks", "chunks_vec", "chunks_fts", "schema_migrations"} <= _tables(conn)


def test_migrations_are_idempotent(tmp_path):
    conn = connect(tmp_path / "radar.db")
    assert migrate(conn) == []


def test_vector_and_keyword_indexes_work():
    conn = connect(":memory:")
    conn.execute(
        "INSERT INTO papers (id, source, title, abstract, published, url) VALUES (?,?,?,?,?,?)",
        ("2401.00001", "arxiv", "Medusa decoding heads", "Faster decoding.", "2024-01-01", "u"),
    )
    conn.execute(
        "INSERT INTO chunks (id, paper_id, kind, text, token_count) VALUES (1, '2401.00001',"
        " 'abstract', 'Medusa adds decoding heads for faster inference.', 9)"
    )
    vec = [0.0] * EMBED_DIM
    vec[0] = 1.0
    conn.execute(
        "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (1, ?)",
        (sqlite_vec.serialize_float32(vec),),
    )
    conn.execute(
        "INSERT INTO chunks_fts (rowid, title, text) VALUES (1, 'Medusa decoding heads',"
        " 'Medusa adds decoding heads for faster inference.')"
    )

    hit = conn.execute(
        "SELECT chunk_id FROM chunks_vec WHERE embedding MATCH ? AND k = 1",
        (sqlite_vec.serialize_float32(vec),),
    ).fetchone()
    assert hit[0] == 1

    kw = conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'medusa'").fetchone()
    assert kw[0] == 1
