"""Reading and writing digests, summaries and the interest profile."""

import json
import sqlite3


def save_digest(conn: sqlite3.Connection, on: str, rows: list[dict]) -> None:
    """Replace the digest for one date. rows: {item_id, rank, slot, score, reason, parts?}.

    `parts` (M4) is the score broken into its terms; a rating uses it to tune the weights."""
    conn.execute("DELETE FROM digests WHERE date = ?", (on,))
    conn.executemany(
        "INSERT INTO digests (date, item_id, rank, slot, score, reason, parts)"
        " VALUES (:date, :item_id, :rank, :slot, :score, :reason, :parts)",
        [
            {**r, "date": on, "parts": json.dumps(r["parts"]) if r.get("parts") else None}
            for r in rows
        ],
    )


def featured_before(conn: sqlite3.Connection, on: str) -> set[str]:
    """Items already shown in an earlier digest, so a paper isn't repeated day after day."""
    rows = conn.execute("SELECT DISTINCT item_id FROM digests WHERE date < ?", (on,))
    return {r[0] for r in rows}


def get_digest(conn: sqlite3.Connection, on: str) -> list[dict]:
    rows = conn.execute(
        "SELECT d.*, i.title, i.url FROM digests d JOIN items i ON i.id = d.item_id"
        " WHERE d.date = ? ORDER BY d.rank",
        (on,),
    )
    return [dict(r) for r in rows]


def save_summary(conn: sqlite3.Connection, item_id: str, text: str, model: str, mode: str) -> None:
    conn.execute(
        "INSERT INTO summaries (item_id, mode, text, model) VALUES (?, ?, ?, ?)"
        " ON CONFLICT(item_id, mode) DO UPDATE SET text = excluded.text,"
        " model = excluded.model, created_at = datetime('now')",
        (item_id, mode, text, model),
    )


def get_profile(conn: sqlite3.Connection) -> dict:
    return {r[0]: json.loads(r[1]) for r in conn.execute("SELECT key, value FROM profile")}


def set_profile(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT INTO profile (key, value) VALUES (?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value)),
    )
