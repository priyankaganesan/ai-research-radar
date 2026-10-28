"""The vector index interface, and its sqlite-vec implementation.

Everything that stores or searches vectors goes through `VectorIndex`, so a second backend
(LanceDB in M1.5) can be swapped in with one setting.
"""

import sqlite3
import struct
from collections.abc import Iterator
from typing import Protocol

import sqlite_vec


class VectorIndex(Protocol):
    def add(self, chunk_ids: list[int], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int) -> list[tuple[int, float]]:
        """Return up to k (chunk_id, distance) pairs, closest first."""
        ...

    def delete(self, chunk_ids: list[int]) -> None: ...

    def count(self) -> int: ...


class SqliteVecIndex:
    """Exact (brute-force) search over the `chunks_vec` table in the main store."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add(self, chunk_ids: list[int], vectors: list[list[float]]) -> None:
        self.conn.executemany(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            [
                (cid, sqlite_vec.serialize_float32(v))
                for cid, v in zip(chunk_ids, vectors, strict=True)
            ],
        )

    def search(self, vector: list[float], k: int) -> list[tuple[int, float]]:
        rows = self.conn.execute(
            "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ?"
            " ORDER BY distance",
            (sqlite_vec.serialize_float32(vector), k),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def delete(self, chunk_ids: list[int]) -> None:
        self.conn.executemany(
            "DELETE FROM chunks_vec WHERE chunk_id = ?", [(c,) for c in chunk_ids]
        )

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]

    def iter_all(self, batch: int = 500) -> Iterator[tuple[list[int], list[list[float]]]]:
        """Every stored (chunk_id, vector), in batches. Used to copy vectors to another backend."""
        cur = self.conn.execute("SELECT chunk_id, embedding FROM chunks_vec ORDER BY chunk_id")
        while rows := cur.fetchmany(batch):
            ids = [r[0] for r in rows]
            vecs = [list(struct.unpack(f"{len(r[1]) // 4}f", r[1])) for r in rows]
            yield ids, vecs

    def disk_bytes(self) -> int | None:
        """Bytes used by the vector tables, if this SQLite build has the dbstat table."""
        try:
            row = self.conn.execute(
                "SELECT SUM(pgsize) FROM dbstat WHERE name LIKE 'chunks_vec%'"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        return row[0] or 0
