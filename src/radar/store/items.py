"""Reading and writing items (papers, posts, ...) in the store."""

import json
import sqlite3
from datetime import date

from radar.schema import Item


def exists(conn: sqlite3.Connection, item_id: str) -> bool:
    return conn.execute("SELECT 1 FROM items WHERE id = ?", (item_id,)).fetchone() is not None


def upsert_item(conn: sqlite3.Connection, item: Item) -> None:
    """Insert or update one item. Doesn't commit; callers group writes in a transaction."""
    conn.execute(
        """
        INSERT INTO items (id, kind, source, title, abstract, authors, published, url, pdf_url,
                           categories)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            title = excluded.title, abstract = excluded.abstract, authors = excluded.authors,
            url = excluded.url, pdf_url = excluded.pdf_url, categories = excluded.categories
        """,
        (
            item.id,
            item.kind,
            item.source,
            item.title,
            item.abstract,
            json.dumps(item.authors),
            item.published.isoformat(),
            item.url,
            item.pdf_url,
            json.dumps(item.categories),
        ),
    )


def get_item(conn: sqlite3.Connection, item_id: str) -> Item | None:
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    return row_to_item(row) if row else None


def count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]


def row_to_item(row: sqlite3.Row) -> Item:
    return Item(
        id=row["id"],
        kind=row["kind"],
        source=row["source"],
        title=row["title"],
        abstract=row["abstract"],
        authors=json.loads(row["authors"]),
        published=date.fromisoformat(row["published"]),
        url=row["url"],
        pdf_url=row["pdf_url"],
        categories=json.loads(row["categories"]),
        tldr=row["tldr"] if "tldr" in row.keys() else None,
    )
