"""Reading and writing the safety tables (migration 008). Plain SQL, no decisions."""

import json
import sqlite3

# ---- source runs ---------------------------------------------------------------------------


def record_source_run(
    conn: sqlite3.Connection, on: str, source: str, fetched: int, added: int, error: str = ""
) -> None:
    """One row per source per day; a second run the same day replaces the first."""
    conn.execute(
        "INSERT INTO source_runs (date, source, fetched, added, error) VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(date, source) DO UPDATE SET fetched = excluded.fetched,"
        " added = excluded.added, error = excluded.error",
        (on, source, fetched, added, error or None),
    )


def source_runs(conn: sqlite3.Connection, days: int = 14) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM source_runs WHERE date >= date('now', ?) ORDER BY date DESC, source",
        (f"-{days} days",),
    )
    return [dict(r) for r in rows]


# ---- flagged items -------------------------------------------------------------------------


def flag_item(conn: sqlite3.Connection, item_id: str, snippet: str, about_attacks: bool) -> bool:
    """Record a flag. Returns True if the item wasn't flagged before. A released item stays
    released: a person already looked at it."""
    cur = conn.execute(
        "INSERT INTO flagged_items (item_id, snippet, about_attacks) VALUES (?, ?, ?)"
        " ON CONFLICT(item_id) DO NOTHING",
        (item_id, snippet, int(about_attacks)),
    )
    return cur.rowcount > 0


def held_ids(conn: sqlite3.Connection) -> set[str]:
    """Items kept out of the digest: flagged, not about attacks, not released."""
    rows = conn.execute(
        "SELECT item_id FROM flagged_items WHERE about_attacks = 0 AND released_at IS NULL"
    )
    return {r[0] for r in rows}


def flagged(conn: sqlite3.Connection, include_released: bool = False) -> list[dict]:
    where = "" if include_released else " WHERE f.released_at IS NULL"
    rows = conn.execute(
        "SELECT f.*, i.title, i.source, i.url FROM flagged_items f"
        f" JOIN items i ON i.id = f.item_id{where} ORDER BY f.flagged_at DESC"
    )
    return [dict(r) for r in rows]


def release(conn: sqlite3.Connection, item_id: str) -> bool:
    cur = conn.execute(
        "UPDATE flagged_items SET released_at = datetime('now')"
        " WHERE item_id = ? AND released_at IS NULL",
        (item_id,),
    )
    return cur.rowcount > 0


# ---- digest reviews ------------------------------------------------------------------------


def save_review(conn: sqlite3.Connection, on: str, reasons: list[str]) -> None:
    conn.execute("DELETE FROM digest_reviews WHERE date = ?", (on,))
    if reasons:
        conn.execute(
            "INSERT INTO digest_reviews (date, reasons) VALUES (?, ?)", (on, json.dumps(reasons))
        )


def get_review(conn: sqlite3.Connection, on: str) -> list[str]:
    row = conn.execute("SELECT reasons FROM digest_reviews WHERE date = ?", (on,)).fetchone()
    return json.loads(row[0]) if row else []


# ---- critic decisions ----------------------------------------------------------------------


def save_critic_decisions(conn: sqlite3.Connection, on: str, log: list[dict]) -> None:
    """From the digest team's handoff log: every {item, critic: keep|drop, concerns} entry.

    Re-running a day's digest adds to that day's decisions rather than replacing them (an
    item judged again keeps only its latest decision), so a rerun doesn't erase evidence."""
    conn.executemany(
        "INSERT OR REPLACE INTO critic_decisions (date, item_id, decision, concerns)"
        " VALUES (?, ?, ?, ?)",
        [
            (on, e["item"], e["critic"], e.get("concerns") or "")
            for e in log
            if e.get("critic") in ("keep", "drop")
        ],
    )


def critic_vs_ratings(conn: sqlite3.Connection) -> list[dict]:
    """Each Critic decision you also rated useful or not relevant (latest rating wins)."""
    rows = conn.execute(
        "SELECT c.date, c.item_id, c.decision, c.concerns, i.title,"
        " (SELECT r.value FROM ratings r WHERE r.item_id = c.item_id"
        "   AND r.value IN ('useful', 'not_relevant') ORDER BY r.seq DESC LIMIT 1) AS rating"
        " FROM critic_decisions c JOIN items i ON i.id = c.item_id"
        " ORDER BY c.date DESC"
    )
    return [dict(r) for r in rows if r["rating"]]


# ---- pending profile changes ---------------------------------------------------------------


def add_pending(conn: sqlite3.Connection, topic: str, why: str, action: str = "remove") -> None:
    exists = conn.execute(
        "SELECT 1 FROM profile_pending WHERE topic = ? AND action = ? AND status = 'pending'",
        (topic, action),
    ).fetchone()
    if not exists:
        conn.execute(
            "INSERT INTO profile_pending (action, topic, why) VALUES (?, ?, ?)",
            (action, topic, why),
        )


def pending(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM profile_pending WHERE status = 'pending' ORDER BY id")
    return [dict(r) for r in rows]


def decide_pending(conn: sqlite3.Connection, pending_id: int, approve: bool) -> dict | None:
    row = conn.execute(
        "SELECT * FROM profile_pending WHERE id = ? AND status = 'pending'", (pending_id,)
    ).fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE profile_pending SET status = ?, decided_at = datetime('now') WHERE id = ?",
        ("approved" if approve else "rejected", pending_id),
    )
    return dict(row)
