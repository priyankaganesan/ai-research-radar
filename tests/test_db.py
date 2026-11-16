import sqlite3
from importlib import resources

import sqlite_vec

from radar.store.db import EMBED_DIM, connect, migrate


def _tables(conn):
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def test_connect_creates_schema(tmp_path):
    conn = connect(tmp_path / "radar.db")
    assert {"items", "chunks", "chunks_vec", "chunks_fts", "schema_migrations"} <= _tables(conn)
    assert "papers" not in _tables(conn)


def test_wal_mode_on(tmp_path):
    conn = connect(tmp_path / "radar.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_migrations_are_idempotent(tmp_path):
    conn = connect(tmp_path / "radar.db")
    assert migrate(conn) == []


def test_002_keeps_data_from_an_m0_store(tmp_path):
    """A store created by M0 (papers table) upgrades without losing rows."""
    conn = sqlite3.connect(tmp_path / "old.db")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    m0 = resources.files("radar.store.migrations").joinpath("001_init.sql").read_text()
    conn.executescript(m0.replace("{EMBED_DIM}", str(EMBED_DIM)))
    conn.execute("CREATE TABLE schema_migrations (name TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO schema_migrations VALUES ('001_init.sql')")
    conn.execute(
        "INSERT INTO papers (id, source, title, abstract, published, url)"
        " VALUES ('p1', 'arxiv', 'T', 'A', '2026-01-01', 'u')"
    )
    conn.execute(
        "INSERT INTO chunks (id, paper_id, kind, text, token_count)"
        " VALUES (7, 'p1', 'abstract', 'x', 1)"
    )
    conn.commit()

    assert migrate(conn) == [
        "002_items.sql",
        "003_sources.sql",
        "004_digest.sql",
        "005_memory.sql",
        "006_tool_calls.sql",
        "007_full_text.sql",
    ]
    item = conn.execute("SELECT kind FROM items WHERE id = 'p1'").fetchone()
    assert item["kind"] == "paper"
    chunk = conn.execute("SELECT item_id, start_s FROM chunks WHERE id = 7").fetchone()
    assert chunk["item_id"] == "p1" and chunk["start_s"] is None


def test_transcript_chunks_allowed(tmp_path):
    conn = connect(tmp_path / "radar.db")
    conn.execute(
        "INSERT INTO items (id, kind, source, title, abstract, published, url)"
        " VALUES ('pod1', 'podcast', 'rss', 'Ep 1', '', '2026-01-01', 'u')"
    )
    conn.execute(
        "INSERT INTO chunks (item_id, kind, text, token_count, start_s, end_s)"
        " VALUES ('pod1', 'transcript', 'hello', 1, 60.0, 120.0)"
    )


def test_vector_and_keyword_indexes_work():
    conn = connect(":memory:")
    conn.execute(
        "INSERT INTO items (id, source, title, abstract, published, url) VALUES (?,?,?,?,?,?)",
        ("2401.00001", "arxiv", "Medusa decoding heads", "Faster decoding.", "2024-01-01", "u"),
    )
    conn.execute(
        "INSERT INTO chunks (id, item_id, kind, text, token_count) VALUES (1, '2401.00001',"
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


def test_signals_merge_columns_from_different_sources(tmp_path):
    from datetime import date

    from radar.store import signals

    conn = connect(tmp_path / "s.db")
    conn.execute(
        "INSERT INTO items (id, source, title, abstract, published, url)"
        " VALUES ('2401.10774', 'arxiv', 'Medusa', 'A', '2026-01-01', 'u')"
    )
    today = date(2026, 10, 27)
    signals.record(conn, "2401.10774", today, hf_upvotes=42)
    signals.record(conn, "2401.10774", today, citations=310, hf_upvotes=None)  # S2 later
    signals.record(conn, "2401.10774", date(2026, 10, 20), citations=290)
    latest = signals.latest(conn, "2401.10774")
    assert latest["hf_upvotes"] == 42  # not wiped by the None
    assert latest["citations"] == 310  # the newest day wins
    assert latest["hn_points"] is None
