"""Attention signals per item per day: Hugging Face upvotes, citations, Hacker News points."""

import sqlite3
from datetime import date

COLUMNS = ("hf_upvotes", "citations", "influential_citations", "hn_points", "hn_comments")


def record(conn: sqlite3.Connection, item_id: str, on: date, **values: int | None) -> None:
    """Upsert today's numbers for one item. Columns not given keep what's already stored, so
    Hugging Face and Semantic Scholar can each write their own columns into the same row."""
    unknown = set(values) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Unknown signal columns: {sorted(unknown)}")
    cols = ", ".join(values)
    marks = ", ".join("?" * len(values))
    # COALESCE(new, old): a None from one source never wipes a number another source wrote.
    updates = ", ".join(f"{c} = COALESCE(excluded.{c}, {c})" for c in values)
    conn.execute(
        f"INSERT INTO signals (item_id, date, {cols}) VALUES (?, ?, {marks})"
        f" ON CONFLICT(item_id, date) DO UPDATE SET {updates}",
        (item_id, on.isoformat(), *values.values()),
    )


def latest(conn: sqlite3.Connection, item_id: str) -> dict[str, int | None]:
    """The most recent non-empty value of each signal, plus the date it was read."""
    out: dict = {}
    for col in COLUMNS:
        row = conn.execute(
            f"SELECT {col}, date FROM signals WHERE item_id = ? AND {col} IS NOT NULL"
            " ORDER BY date DESC LIMIT 1",
            (item_id,),
        ).fetchone()
        out[col] = row[0] if row else None
    return out


def set_tldr(conn: sqlite3.Connection, item_id: str, tldr: str) -> None:
    conn.execute("UPDATE items SET tldr = ? WHERE id = ?", (tldr, item_id))
