"""Reading the local tool-call log (written by radar/tracing.py)."""

import sqlite3


def recent(
    conn: sqlite3.Connection,
    limit: int = 30,
    name: str | None = None,
    kind: str | None = None,
    errors_only: bool = False,
    roots_only: bool = False,
) -> list[dict]:
    """The newest calls first. `name` matches anywhere in the name ("arxiv" finds them all)."""
    where, params = [], []
    if name:
        where.append("name LIKE ?")
        params.append(f"%{name}%")
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if errors_only:
        where.append("error IS NOT NULL")
    if roots_only:
        where.append("parent_id IS NULL")
    sql = "SELECT * FROM tool_calls"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY started_at DESC, rowid DESC LIMIT ?"
    return [dict(r) for r in conn.execute(sql, (*params, limit))]


def tree(conn: sqlite3.Connection, root_id: str) -> list[dict]:
    """Every call under one top-level call, parents before children (depth-first, siblings in
    the order they started), each with its `depth`."""
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM tool_calls WHERE root_id = ? ORDER BY started_at, rowid", (root_id,)
        )
    ]
    children: dict[str | None, list[dict]] = {}
    ids = {r["id"] for r in rows}
    for r in rows:
        parent = r["parent_id"] if r["parent_id"] in ids else None
        children.setdefault(parent, []).append(r)
    out: list[dict] = []

    def walk(parent: str | None, depth: int) -> None:
        for r in children.get(parent, []):
            r["depth"] = depth
            out.append(r)
            walk(r["id"], depth + 1)

    walk(None, 0)
    return out


def find_root(conn: sqlite3.Connection, call_id: str) -> str | None:
    """The root of the call whose ID starts with `call_id` (IDs can be shortened)."""
    row = conn.execute(
        "SELECT root_id FROM tool_calls WHERE id LIKE ? ORDER BY started_at DESC LIMIT 1",
        (f"{call_id}%",),
    ).fetchone()
    return row[0] if row else None


def stats(conn: sqlite3.Connection, since: str | None = None) -> list[dict]:
    """Calls, errors and average/max duration per name."""
    sql = (
        "SELECT name, kind, COUNT(*) AS calls, SUM(error IS NOT NULL) AS errors,"
        " ROUND(AVG(duration_ms), 1) AS avg_ms, ROUND(MAX(duration_ms), 1) AS max_ms"
        " FROM tool_calls"
    )
    params: tuple = ()
    if since:
        sql += " WHERE started_at >= ?"
        params = (since,)
    sql += " GROUP BY name, kind ORDER BY calls DESC"
    return [dict(r) for r in conn.execute(sql, params)]
