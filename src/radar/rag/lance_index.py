"""LanceDB implementation of `VectorIndex` (M1.5).

LanceDB is an embedded vector database: like SQLite it runs inside our process and keeps its
data in files (`data/lance/`), with no server. It stores columns in the Lance format (built on
Apache Arrow) and can build an approximate nearest-neighbour (ANN) index, IVF-PQ, which
sqlite-vec doesn't have.

LanceDB holds only (chunk_id, vector). SQLite stays the source of truth, so this whole folder
can be deleted and rebuilt with `radar reindex --backend lance`.
"""

from pathlib import Path

import pyarrow as pa

TABLE = "chunks"
# PQ trains 256 centroids per sub-vector, so it needs at least that many rows.
ANN_MIN_ROWS = 256


class LanceIndex:
    """Exact search by default; approximate once `build_ann()` has created an IVF-PQ index."""

    def __init__(self, path: Path, dim: int, nprobes: int = 20, refine_factor: int | None = None):
        import lancedb  # imported here so the sqlite backend never pays its start-up cost

        path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(path))
        self.path = path
        self.dim = dim
        # Search-time knobs for the ANN index (ignored by a flat scan):
        #   nprobes: how many IVF partitions to scan. More = better recall, slower.
        #   refine_factor: fetch k * refine_factor candidates and re-rank them with exact
        #   distances, undoing most of PQ's compression error.
        self.nprobes = nprobes
        self.refine_factor = refine_factor
        self.schema = pa.schema(
            [pa.field("chunk_id", pa.int64()), pa.field("vector", pa.list_(pa.float32(), dim))]
        )
        self.table = self.db.create_table(TABLE, schema=self.schema, exist_ok=True)

    def add(self, chunk_ids: list[int], vectors: list[list[float]]) -> None:
        if chunk_ids:
            rows = [{"chunk_id": c, "vector": v} for c, v in zip(chunk_ids, vectors, strict=True)]
            self.table.add(rows)

    def search(self, vector: list[float], k: int) -> list[tuple[int, float]]:
        if self.table.count_rows() == 0:
            return []
        query = self.table.search(vector).metric("l2").limit(k).nprobes(self.nprobes)
        if self.refine_factor:
            query = query.refine_factor(self.refine_factor)
        # LanceDB reports squared L2 distance; sqlite-vec reports plain L2. Same ranking.
        return [(r["chunk_id"], r["_distance"]) for r in query.to_list()]

    def delete(self, chunk_ids: list[int]) -> None:
        if chunk_ids:
            self.table.delete(f"chunk_id IN ({','.join(str(int(c)) for c in chunk_ids)})")

    def count(self) -> int:
        return self.table.count_rows()

    def clear(self) -> None:
        """Drop every vector (used by reindex before copying from SQLite)."""
        self.table = self.db.create_table(TABLE, schema=self.schema, mode="overwrite")

    def has_ann(self) -> bool:
        return bool(self.table.list_indices())

    def build_ann(self, num_sub_vectors: int | None = None) -> dict:
        """Train an IVF-PQ index. Returns the settings used.

        IVF splits the vectors into `num_partitions` clusters; a search scans only the nearest
        `nprobes` of them. PQ compresses each vector into `num_sub_vectors` one-byte codes, so
        1024 floats (4 KB) shrink to 64 bytes. Both trade a little recall for speed.
        """
        n = self.count()
        if n < ANN_MIN_ROWS:
            raise ValueError(f"IVF-PQ needs at least {ANN_MIN_ROWS} vectors; the index has {n}.")
        from lancedb.index import IvfPq

        # The usual rule of thumb is about sqrt(n) partitions, but LanceDB trains each partition
        # on at least 256 vectors, so small tables get fewer (a few hundred papers get just one).
        partitions = max(1, min(round(n**0.5), n // 256))
        sub_vectors = num_sub_vectors or max(1, self.dim // 16)  # 16 dimensions per code
        config = IvfPq(distance_type="l2", num_partitions=partitions, num_sub_vectors=sub_vectors)
        self.table.create_index("vector", config=config, replace=True)
        return {"rows": n, "num_partitions": partitions, "num_sub_vectors": sub_vectors}

    def disk_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.path.rglob("*") if f.is_file())
