"""Exact (sqlite-vec) vs. approximate (LanceDB) vector search, measured on the same queries.

Three numbers per configuration:
- recall@k vs. exact: of the k chunks exact search returns, how many the backend also returns.
  This is the standard way to measure an ANN index, and needs no hand-labelled answers.
- latency: p50 and p95 milliseconds per vector search (embedding time not included).
- disk: bytes on disk for the vectors.
"""

import statistics
import time
from dataclasses import dataclass, field

from radar.rag.lance_index import LanceIndex
from radar.rag.vector_index import SqliteVecIndex, VectorIndex


@dataclass
class RunStats:
    name: str
    recall_vs_exact: float
    p50_ms: float
    p95_ms: float
    results: list[list[int]] = field(default_factory=list, repr=False)


def _timed(index: VectorIndex, qvecs: list[list[float]], k: int) -> tuple[list[list[int]], list]:
    index.search(qvecs[0], k)  # warm-up: the first query loads files and caches
    results, times = [], []
    for v in qvecs:
        start = time.perf_counter()
        hits = index.search(v, k)
        times.append((time.perf_counter() - start) * 1000)
        results.append([cid for cid, _ in hits])
    return results, times


def _pct(times: list[float], q: float) -> float:
    ordered = sorted(times)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)


def run_backend(
    name: str, index: VectorIndex, qvecs: list[list[float]], k: int, exact: list[list[int]]
) -> RunStats:
    results, times = _timed(index, qvecs, k)
    overlaps = [len(set(r) & set(e)) / max(1, len(e)) for r, e in zip(results, exact, strict=True)]
    return RunStats(
        name=name,
        recall_vs_exact=round(statistics.mean(overlaps), 3),
        p50_ms=_pct(times, 0.50),
        p95_ms=_pct(times, 0.95),
        results=results,
    )


def compare(
    sqlite_index: SqliteVecIndex, lance: LanceIndex, qvecs: list[list[float]], k: int = 5
) -> dict:
    """Run every configuration. Without an ANN index LanceDB scans everything (exact), so the
    sweep over nprobes / refine_factor only happens once `radar reindex --backend lance --ann`
    has built one."""
    exact_results, _ = _timed(sqlite_index, qvecs, k)
    runs = [run_backend("sqlite-vec (exact)", sqlite_index, qvecs, k, exact_results)]

    original = (lance.nprobes, lance.refine_factor)
    if lance.has_ann():
        for nprobes in (1, 5, 20, 50):
            for refine in (None, 10):
                lance.nprobes, lance.refine_factor = nprobes, refine
                label = f"lancedb IVF-PQ nprobes={nprobes}" + (
                    f" refine={refine}" if refine else ""
                )
                runs.append(run_backend(label, lance, qvecs, k, exact_results))
    else:
        runs.append(run_backend("lancedb (flat, exact)", lance, qvecs, k, exact_results))
    lance.nprobes, lance.refine_factor = original

    return {
        "k": k,
        "queries": len(qvecs),
        "vectors": sqlite_index.count(),
        "ann_index": lance.has_ann(),
        "disk_bytes": {"sqlite-vec": sqlite_index.disk_bytes(), "lancedb": lance.disk_bytes()},
        "runs": runs,
    }


def to_markdown(report: dict) -> str:
    def size(b: int | None) -> str:
        return "n/a" if b is None else f"{b / 1_048_576:.1f} MB"

    lines = [
        "# Vector backend comparison",
        "",
        f"{report['queries']} queries, {report['vectors']} vectors, k={report['k']}, "
        f"LanceDB ANN index: {'yes' if report['ann_index'] else 'no (flat scan)'}",
        "",
        "| Configuration | Recall@k vs. exact | p50 ms | p95 ms |",
        "|---|---|---|---|",
    ]
    for r in report["runs"]:
        lines.append(f"| {r.name} | {r.recall_vs_exact:.3f} | {r.p50_ms} | {r.p95_ms} |")
    disk = report["disk_bytes"]
    lines += [
        "",
        f"Disk: sqlite-vec {size(disk['sqlite-vec'])}, LanceDB {size(disk['lancedb'])}",
    ]
    if "gold" in report:
        lines += ["", "Recall@k against expected papers (full hybrid search):", ""]
        lines += [f"- {name}: {value:.2f}" for name, value in report["gold"].items()]
    return "\n".join(lines) + "\n"
