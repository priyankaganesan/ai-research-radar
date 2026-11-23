"""Long-term memory (M4): ratings, questions, gaps, the early-buzz watchlist and profile changes.

Plain SQL, no logic: deciding what a rating *means* lives in `radar/feedback.py` (pure
functions) and `radar/personalize.py` (the update run).
"""

import json
import sqlite3
from datetime import date, timedelta

RATING_VALUES = ("useful", "not_relevant", "skipped")


# ---- ratings -------------------------------------------------------------------------------


def add_rating(conn: sqlite3.Connection, item_id: str, value: str, on: date) -> str | None:
    """Store a rating. Returns the digest slot the item was last shown in (or None)."""
    if value not in RATING_VALUES:
        raise ValueError(f"Rating must be one of {', '.join(RATING_VALUES)}, not {value!r}")
    row = conn.execute(
        "SELECT slot FROM digests WHERE item_id = ? AND date <= ? ORDER BY date DESC LIMIT 1",
        (item_id, on.isoformat()),
    ).fetchone()
    slot = row[0] if row else None
    seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM ratings").fetchone()[0]
    conn.execute(
        "INSERT INTO ratings (item_id, date, value, slot, seq) VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(item_id, date) DO UPDATE SET value = excluded.value,"
        " slot = excluded.slot, seq = excluded.seq",
        (item_id, on.isoformat(), value, slot, seq),
    )
    return slot


def ratings_since(conn: sqlite3.Connection, seq: int) -> list[dict]:
    """Ratings newer than `seq`, with the score parts of the digest pick they refer to."""
    rows = conn.execute(
        "SELECT r.item_id, r.date, r.value, r.slot, r.seq, i.title,"
        " (SELECT d.parts FROM digests d WHERE d.item_id = r.item_id AND d.date <= r.date"
        "  ORDER BY d.date DESC LIMIT 1) AS parts"
        " FROM ratings r JOIN items i ON i.id = r.item_id WHERE r.seq > ? ORDER BY r.seq",
        (seq,),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["parts"] = json.loads(d["parts"]) if d["parts"] else None
        out.append(d)
    return out


def latest_ratings(conn: sqlite3.Connection) -> dict[str, str]:
    """Each rated item's most recent rating: {item_id: value}. A change of mind wins."""
    rows = conn.execute(
        "SELECT item_id, value FROM ratings r WHERE date ="
        " (SELECT MAX(date) FROM ratings r2 WHERE r2.item_id = r.item_id)"
    )
    return {r[0]: r[1] for r in rows}


# ---- questions and gaps --------------------------------------------------------------------


def log_question(
    conn: sqlite3.Connection,
    text: str,
    answered: bool,
    cited: list[str],
    gap_topic: str = "",
    thread_id: str | None = None,
    answer: str | None = None,
) -> int:
    """Record a question; an unanswered one also opens a gap on `gap_topic`.
    `thread_id` and `answer` (migration 009) let the web UI list and reopen conversations."""
    cur = conn.execute(
        "INSERT INTO questions (text, answered, cited_ids, thread_id, answer)"
        " VALUES (?, ?, ?, ?, ?)",
        (text, int(answered), json.dumps(cited), thread_id, answer),
    )
    qid = cur.lastrowid
    if not answered:
        conn.execute(
            "INSERT INTO gaps (question_id, topic) VALUES (?, ?)", (qid, gap_topic or text)
        )
    return qid


def open_gaps(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT g.id, g.topic, g.created_at, g.attempts, q.text AS question FROM gaps g"
        " LEFT JOIN questions q ON q.id = g.question_id WHERE g.resolved_at IS NULL ORDER BY g.id"
    )
    return [dict(r) for r in rows]


def gap_searched(conn: sqlite3.Connection, gap_id: int, found_new: bool) -> None:
    """Count a collection attempt; a gap is resolved once a search brings in new items."""
    conn.execute("UPDATE gaps SET attempts = attempts + 1 WHERE id = ?", (gap_id,))
    if found_new:
        conn.execute("UPDATE gaps SET resolved_at = datetime('now') WHERE id = ?", (gap_id,))


def recent_questions(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT id, asked_at, text, answered FROM questions ORDER BY id DESC LIMIT ?", (limit,)
    )
    return [dict(r) for r in rows]


def conversations(conn: sqlite3.Connection, limit: int = 30) -> list[dict]:
    """Recent conversations (questions sharing a thread), newest first. Questions asked before
    migration 009 have no thread and aren't listed."""
    rows = conn.execute(
        "SELECT thread_id, COUNT(*) AS questions, MIN(asked_at) AS started,"
        " MAX(asked_at) AS last_asked,"
        " (SELECT q2.text FROM questions q2 WHERE q2.thread_id = q.thread_id"
        "  ORDER BY q2.id LIMIT 1) AS first_question"
        " FROM questions q WHERE thread_id IS NOT NULL"
        " GROUP BY thread_id ORDER BY MAX(id) DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


def thread_questions(conn: sqlite3.Connection, thread_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, asked_at, text, answered, cited_ids, answer FROM questions"
        " WHERE thread_id = ? ORDER BY id",
        (thread_id,),
    )
    return [{**dict(r), "cited_ids": json.loads(r["cited_ids"])} for r in rows]


# ---- early-buzz watchlist ------------------------------------------------------------------


def watch(conn: sqlite3.Connection, item_id: str, on: date, citations: int, days: int) -> None:
    """Put an item on the watchlist (no-op if it's already there)."""
    conn.execute(
        "INSERT OR IGNORE INTO watchlist (item_id, added_at, recheck_after, citations_at_add)"
        " VALUES (?, ?, ?, ?)",
        (item_id, on.isoformat(), (on + timedelta(days=days)).isoformat(), citations),
    )


def due_for_recheck(conn: sqlite3.Connection, on: date) -> list[dict]:
    rows = conn.execute(
        "SELECT w.*, i.title FROM watchlist w JOIN items i ON i.id = w.item_id"
        " WHERE w.status = 'watching' AND w.recheck_after <= ? ORDER BY w.added_at",
        (on.isoformat(),),
    )
    return [dict(r) for r in rows]


def set_watch_status(
    conn: sqlite3.Connection,
    item_id: str,
    status: str,
    on: date,
    note: str,
    recheck_after: date | None = None,
) -> None:
    conn.execute(
        "UPDATE watchlist SET status = ?, checked_at = ?, note = ?,"
        " recheck_after = COALESCE(?, recheck_after) WHERE item_id = ?",
        (status, on.isoformat(), note, recheck_after and recheck_after.isoformat(), item_id),
    )


def get_watchlist(conn: sqlite3.Connection, status: str | None = None) -> list[dict]:
    sql = (
        "SELECT w.*, i.title, i.url FROM watchlist w JOIN items i ON i.id = w.item_id"
        + (" WHERE w.status = ?" if status else "")
        + " ORDER BY w.status = 'watching' DESC, w.recheck_after"
    )
    return [dict(r) for r in conn.execute(sql, (status,) if status else ())]


def watching_ids(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT item_id FROM watchlist WHERE status = 'watching'")]


# ---- profile change log --------------------------------------------------------------------


def log_change(conn: sqlite3.Connection, kind: str, detail: dict) -> None:
    conn.execute(
        "INSERT INTO profile_changes (kind, detail) VALUES (?, ?)", (kind, json.dumps(detail))
    )


def recent_changes(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        "SELECT at, kind, detail FROM profile_changes ORDER BY id DESC LIMIT ?", (limit,)
    )
    return [{"at": r[0], "kind": r[1], "detail": json.loads(r[2])} for r in rows]
