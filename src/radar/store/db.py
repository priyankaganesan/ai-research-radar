"""SQLite connection with sqlite-vec loaded, and a small migration runner."""

import sqlite3
from importlib import resources
from pathlib import Path

import sqlite_vec

EMBED_DIM = 1024  # qwen3-embedding:0.6b


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the store, load sqlite-vec, and apply any pending migrations."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Apply migrations/NNN_*.sql in order. Returns the names applied this call."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY)")
    done = {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}
    files = sorted(
        (f for f in resources.files("radar.store.migrations").iterdir() if f.name.endswith(".sql")),
        key=lambda f: f.name,
    )
    applied = []
    for f in files:
        if f.name in done:
            continue
        sql = f.read_text().replace("{EMBED_DIM}", str(EMBED_DIM))
        with conn:
            conn.executescript(sql)
            conn.execute("INSERT INTO schema_migrations (name) VALUES (?)", (f.name,))
        applied.append(f.name)
    return applied
