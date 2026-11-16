"""Full texts (M5): which items have one, where it came from, and the sections themselves.

Plain SQL, like the other store modules. The chunks of a full text live in `chunks`
(kind = 'section'); `rag/index.py` writes those.
"""

import json
import sqlite3


def get(conn: sqlite3.Connection, item_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM full_texts WHERE item_id = ?", (item_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["sections"] = json.loads(d["sections"]) if d["sections"] else []
    return d


def save_ok(
    conn: sqlite3.Connection,
    item_id: str,
    source: str,
    url: str,
    sections: list[dict],
    n_chunks: int,
    reason: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO full_texts (item_id, status, source, url, sections, n_sections,"
        " n_chunks, chars, reason) VALUES (?, 'ok', ?, ?, ?, ?, ?, ?, ?)",
        (
            item_id,
            source,
            url,
            json.dumps(sections),
            len(sections),
            n_chunks,
            sum(len(s["text"]) for s in sections),
            reason,
        ),
    )


def save_failed(conn: sqlite3.Connection, item_id: str, error: str, reason: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO full_texts (item_id, status, error, reason)"
        " VALUES (?, 'failed', ?, ?)",
        (item_id, error[:500], reason),
    )


def listing(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT f.item_id, i.title, f.status, f.source, f.n_sections, f.n_chunks, f.chars,"
        " f.reason, f.error, f.fetched_at FROM full_texts f JOIN items i ON i.id = f.item_id"
        " ORDER BY f.fetched_at DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


def times_cited(conn: sqlite3.Connection, item_ids: list[str]) -> dict[str, int]:
    """How many logged questions cited each item in their answer. json_each() turns the JSON
    list in `questions.cited_ids` into rows, so SQLite can count inside it."""
    if not item_ids:
        return {}
    marks = ",".join("?" * len(item_ids))
    rows = conn.execute(
        f"SELECT j.value, COUNT(DISTINCT q.id) FROM questions q, json_each(q.cited_ids) j"
        f" WHERE j.value IN ({marks}) GROUP BY j.value",
        item_ids,
    )
    return {r[0]: r[1] for r in rows}
