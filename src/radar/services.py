"""What the agent can do, as plain functions. The CLI (and later MCP and the web app) call these.

Nothing outside this module builds graphs or touches the store directly.
"""

import json
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from radar.context import AppContext, make_vector_index
from radar.graphs import prompts
from radar.rag.index import EMBED_BATCH, index_items
from radar.rag.lance_index import LanceIndex
from radar.rag.retrieve import QUERY_INSTRUCTION, hybrid_search
from radar.rag.vector_index import SqliteVecIndex
from radar.schema import Item, Passage
from radar.sources import arxiv, hackernews, hf_papers, lab_blogs, semantic_scholar
from radar.sources.http import SourceUnavailable
from radar.store import items as item_store
from radar.store import memory
from radar.store import safety as safety_store
from radar.store import signals as signal_store
from radar.tracing import passages_as_documents, traced


@dataclass
class AskResult:
    answer: str
    cited: list[str]
    thread_id: str
    trace: list[dict] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list)
    verification: list[dict] = field(default_factory=list)
    full_texts: list[dict] = field(default_factory=list)  # M5: fetched after this question


# ---- Progress while a graph runs (used by the web UI) ---------------------------------------

StepCallback = Callable[[str, list[dict]], None]


class Cancelled(Exception):
    """Raised between graph steps when the caller asked to stop."""


def run_graph(graph, inputs: dict, config: dict, on_step=None, should_stop=None) -> dict:
    """Run a compiled graph to the end and return its final state, like `graph.invoke`.

    With no callbacks it *is* `graph.invoke`. Otherwise it uses `graph.stream`, which runs the
    same graph but hands back control after every node: "updates" says which node just
    finished and what it returned, "values" is the whole state after that step. Each node in
    our graphs appends readable entries to `trace`, so the new entries are the progress report.
    """
    if on_step is None and should_stop is None:
        return graph.invoke(inputs, config=config)
    state: dict = {}
    seen = 0
    for mode, chunk in graph.stream(inputs, config=config, stream_mode=["updates", "values"]):
        if mode == "values":
            state = chunk
            continue
        for node, update in chunk.items():
            trace = (update or {}).get("trace") if isinstance(update, dict) else None
            new: list[dict] = []
            if trace is not None:
                seen = min(seen, len(trace))  # `prepare` starts a fresh trace
                new, seen = trace[seen:], len(trace)
            if on_step:
                on_step(node, new)
        if should_stop and should_stop():
            raise Cancelled("stopped between steps")
    return state


@traced("services.ingest", "chain")
def ingest(
    ctx: AppContext,
    days: int = 7,
    categories: list[str] | None = None,
    max_results: int = 500,
    source: str = "auto",
) -> tuple[int, int, str]:
    """Fetch recent arXiv papers into the store. Returns (fetched, newly added, source used).

    source: "api" (date range, paged), "rss" (latest announcement day only), or "auto" (the API,
    falling back to RSS when the API is rate-limited).
    """
    cats = categories or ctx.settings.arxiv_categories.split(",")
    used = source
    if source == "rss":
        found = arxiv.fetch_latest_rss(cats)
    else:
        try:
            found = arxiv.fetch_recent(days=days, categories=cats, max_results=max_results)
            used = "api"
        except arxiv.ArxivUnavailable:
            if source == "api":
                raise
            found = arxiv.fetch_latest_rss(cats)
            used = "rss"
    found = found[:max_results]
    added = index_items(ctx.conn, ctx.embeddings, ctx.vectors, found)
    return len(found), added, used


@dataclass
class SourceReport:
    """What one source run did, for the CLI to print."""

    source: str
    fetched: int = 0
    added: int = 0
    signals: int = 0
    errors: dict[str, str] = field(default_factory=dict)


def _save_tldrs(conn: sqlite3.Connection, items: list[Item]) -> None:
    with conn:
        for i in items:
            if i.tldr and item_store.exists(conn, i.id):
                signal_store.set_tldr(conn, i.id, i.tldr)


def _s2_client(ctx: AppContext):
    return semantic_scholar.make_client(ctx.settings.semantic_scholar_api_key)


@traced("services.ingest_hf", "chain")
def ingest_hf(ctx: AppContext, days: int = 3, client=None) -> SourceReport:
    """Hugging Face Daily Papers from the last `days` days, with today's upvote counts."""
    papers = hf_papers.fetch_recent(days, client)
    added = index_items(ctx.conn, ctx.embeddings, ctx.vectors, [p.item for p in papers])
    today = date.today()
    with ctx.conn:
        for p in papers:
            signal_store.record(ctx.conn, p.item.id, today, hf_upvotes=p.upvotes)
    return SourceReport("hf", fetched=len(papers), added=added, signals=len(papers))


@traced("services.ingest_labs", "chain")
def ingest_labs(ctx: AppContext, days: int = 14, client=None) -> SourceReport:
    """Posts from the AI lab feeds in LAB_FEEDS. A broken feed is reported, not fatal."""
    got = lab_blogs.fetch_all(ctx.settings.lab_feeds, days, client)
    added = index_items(ctx.conn, ctx.embeddings, ctx.vectors, got.items)
    return SourceReport("labs", fetched=len(got.items), added=added, errors=got.errors)


@traced("services.ingest_hn", "chain")
def ingest_hn(
    ctx: AppContext, days: int = 2, min_points: int | None = None, client=None, s2_client=None
) -> SourceReport:
    """AI stories from Hacker News. Stories linking to arXiv add `hn_points` to that paper
    (fetching it from Semantic Scholar if it's not in the store yet); others become posts."""
    queries = [q.strip() for q in ctx.settings.hn_queries.split(",") if q.strip()]
    points = ctx.settings.hn_min_points if min_points is None else min_points
    stories = hackernews.fetch_stories(queries, days, points, client)
    report = SourceReport("hn", fetched=len(stories))

    # The most-discussed story per paper (stories arrive sorted by points).
    by_paper: dict[str, hackernews.HNStory] = {}
    for s in stories:
        if s.arxiv_id:
            by_paper.setdefault(s.arxiv_id, s)
    new_items: list[Item] = []
    missing = [a for a in by_paper if not item_store.exists(ctx.conn, a)]
    if missing:  # the arXiv API is often rate-limited, so get abstracts from S2 instead
        try:
            found = semantic_scholar.lookup_arxiv(missing, s2_client or _s2_client(ctx))
            new_items += [i for p in found.values() if (i := p.to_item())]
        except SourceUnavailable as e:
            report.errors["Semantic Scholar"] = str(e)
    for s in stories:
        if s.arxiv_id:
            continue
        discussion = f"Discussed on Hacker News ({s.points} points, {s.comments} comments)."
        new_items.append(
            Item(
                id=f"hn:{s.id}",
                source="hn",
                kind="post",
                title=s.title,
                abstract=" ".join(filter(None, [s.title + ".", s.text, discussion])),
                published=s.created,
                url=s.url,
            )
        )
    report.added = index_items(ctx.conn, ctx.embeddings, ctx.vectors, new_items)
    _save_tldrs(ctx.conn, new_items)

    today = date.today()
    with ctx.conn:
        targets = list(by_paper.items()) + [(f"hn:{s.id}", s) for s in stories if not s.arxiv_id]
        for item_id, s in targets:
            if item_store.exists(ctx.conn, item_id):
                signal_store.record(
                    ctx.conn, item_id, today, hn_points=s.points, hn_comments=s.comments
                )
                report.signals += 1
    return report


@traced("services.enrich", "chain")
def enrich(ctx: AppContext, days: int = 30, limit: int = 500, client=None) -> SourceReport:
    """Citation counts and TLDRs from Semantic Scholar for recent arXiv papers in the store."""
    since = (date.today() - timedelta(days=days)).isoformat()
    ids = [
        r[0]
        for r in ctx.conn.execute(
            "SELECT id FROM items WHERE kind = 'paper' AND id NOT LIKE '%:%' AND published >= ?"
            " ORDER BY published DESC LIMIT ?",
            (since, limit),
        )
    ]
    # M4: watched early-buzz papers need fresh citation counts even once they're older.
    ids += [i for i in memory.watching_ids(ctx.conn) if i not in ids and ":" not in i]
    found = semantic_scholar.lookup_arxiv(ids, client or _s2_client(ctx)) if ids else {}
    today = date.today()
    with ctx.conn:
        for arxiv_id, p in found.items():
            if not item_store.exists(ctx.conn, arxiv_id):
                continue
            signal_store.record(
                ctx.conn,
                arxiv_id,
                today,
                citations=p.citations,
                influential_citations=p.influential_citations,
            )
            if p.tldr:
                signal_store.set_tldr(ctx.conn, arxiv_id, p.tldr)
    return SourceReport("s2", fetched=len(ids), signals=len(found))


@traced("services.show", "tool")
def show(ctx: AppContext, item_id: str) -> tuple[Item, dict] | None:
    """One item with its latest signals, for `radar show`."""
    item = item_store.get_item(ctx.conn, item_id)
    return (item, signal_store.latest(ctx.conn, item_id)) if item else None


@traced("services.search", "retriever", process_outputs=passages_as_documents)
def search(
    ctx: AppContext, query: str, k: int = 5, kind: str | None = None, since: str | None = None
) -> list[Passage]:
    return hybrid_search(ctx.conn, ctx.embeddings, ctx.vectors, query, k=k, kind=kind, since=since)


@traced("services.get_item", "tool")
def get_item(ctx: AppContext, item_id: str) -> Item | None:
    return item_store.get_item(ctx.conn, item_id)


@traced("services.stats", "tool")
def stats(ctx: AppContext) -> dict:
    return {
        "items": item_store.count(ctx.conn),
        "vectors": ctx.vectors.count(),
        "backend": ctx.settings.vector_backend,
    }


ANN_AUTO_ROWS = 5000  # below this a flat scan takes a few ms, so reindex skips the ANN index


@traced("services.reindex", "chain")
def reindex(ctx: AppContext, backend: str, ann: bool = False) -> dict:
    """Rebuild a vector index.

    lance: copy every vector from SQLite into a fresh LanceDB table (no re-embedding), then
    build an IVF-PQ index if asked (`ann`) or if there are at least ANN_AUTO_ROWS vectors.
    sqlite: re-embed every chunk with the current EMBED_MODEL (after changing models).
    """
    store_vectors = SqliteVecIndex(ctx.conn)
    if backend == "lance":
        lance = _lance(ctx)
        lance.clear()
        for ids, vecs in store_vectors.iter_all():
            lance.add(ids, vecs)
        result = {"backend": "lance", "vectors": lance.count(), "ann": None}
        if ann or lance.count() >= ANN_AUTO_ROWS:
            result["ann"] = lance.build_ann()
        return result
    if backend == "sqlite":
        rows = ctx.conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
        with ctx.conn:
            ctx.conn.execute("DELETE FROM chunks_vec")
            for start in range(0, len(rows), EMBED_BATCH):
                batch = rows[start : start + EMBED_BATCH]
                vecs = ctx.embeddings.embed_documents([r[1] for r in batch])
                store_vectors.add([r[0] for r in batch], vecs)
        return {"backend": "sqlite", "vectors": store_vectors.count(), "ann": None}
    raise ValueError(f"Unknown backend {backend!r} (use sqlite or lance)")


def _lance(ctx: AppContext) -> LanceIndex:
    """The active LanceDB index, or one opened from LANCE_PATH when another backend is active."""
    if isinstance(ctx.vectors, LanceIndex):
        return ctx.vectors
    return make_vector_index(ctx.settings, ctx.conn, "lance")


@traced("services.compare_backends", "chain")
def compare_backends(
    ctx: AppContext,
    questions_file: Path | None = None,
    k: int = 5,
    sample: int = 50,
    out_dir: Path | None = Path("evals/results"),
) -> dict:
    """Compare sqlite-vec and LanceDB on the same queries; see rag/compare.py for the metrics.

    Queries come from the eval file if given, otherwise from `sample` item titles in the store
    (enough to measure ANN recall, which needs no labelled answers).
    """
    import random
    from datetime import datetime

    from radar.rag import compare

    store_vectors = SqliteVecIndex(ctx.conn)
    lance = _lance(ctx)
    if lance.count() != store_vectors.count():
        raise ValueError(
            f"LanceDB has {lance.count()} vectors but SQLite has {store_vectors.count()}. "
            "Run `radar reindex --backend lance` first."
        )
    rows = []
    if questions_file is not None:
        rows = [json.loads(x) for x in questions_file.read_text().splitlines() if x.strip()]
        queries = [r["question"] for r in rows]
    else:
        titles = [r[0] for r in ctx.conn.execute("SELECT title FROM items ORDER BY id")]
        queries = random.Random(0).sample(titles, min(sample, len(titles)))
    if not queries:
        raise ValueError("No queries: the store is empty. Run `radar ingest` first.")

    qvecs = [ctx.embeddings.embed_query(QUERY_INSTRUCTION + q) for q in queries]
    report = compare.compare(store_vectors, lance, qvecs, k=k)
    if rows and any(r.get("expected_ids") for r in rows):
        report["gold"] = {
            name: eval_retrieval(replace(ctx, vectors=index), questions_file, k=k)["recall"]
            for name, index in (("sqlite-vec", store_vectors), ("lancedb", lance))
        }
    report["markdown"] = compare.to_markdown(report)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"backends-{datetime.now():%Y%m%d-%H%M}.md"
        path.write_text(report["markdown"])
        report["path"] = str(path)
    return report


DEFAULT_DAYS = {"arxiv": 7, "hf": 3, "labs": 14, "hn": 2}


@traced("services.collect_all", "chain")
def collect_all(ctx: AppContext) -> list[str]:
    """Run every source, then Semantic Scholar enrichment. One failing source doesn't stop the
    rest; it's noted in the returned log instead."""
    log = []
    today = date.today().isoformat()

    def record(source: str, fetched: int = 0, added: int = 0, error: str = "") -> None:
        # Safety: one row per source per day, so a source that silently breaks shows up.
        with ctx.conn:
            safety_store.record_source_run(ctx.conn, today, source, fetched, added, error)

    try:
        fetched, added, used = ingest(ctx, days=DEFAULT_DAYS["arxiv"])
        log.append(f"arxiv ({used}): fetched {fetched}, added {added}")
        record("arxiv", fetched, added)
    except SourceUnavailable as e:
        log.append(f"arxiv: unavailable ({e})")
        record("arxiv", error=str(e))
    for name, fn in (("hf", ingest_hf), ("labs", ingest_labs), ("hn", ingest_hn)):
        try:
            r = fn(ctx, days=DEFAULT_DAYS[name])
            log.append(f"{name}: fetched {r.fetched}, added {r.added}")
            record(name, r.fetched, r.added, "; ".join(r.errors.values()) if not r.fetched else "")
        except SourceUnavailable as e:
            log.append(f"{name}: unavailable ({e})")
            record(name, error=str(e))
    try:
        r = enrich(ctx)
        log.append(f"s2 enrich: {r.signals} papers updated")
        record("s2", r.signals, 0)
    except SourceUnavailable as e:
        log.append(f"s2 enrich: unavailable ({e})")
        record("s2", error=str(e))
    log += collect_gaps(ctx)
    for source in quiet_sources(ctx):
        log.append(f"WARNING: {source} returned nothing on its last two runs. Is it broken?")
    return log


def quiet_sources(ctx: AppContext) -> list[str]:
    """Sources that fetched nothing (or failed) on each of their last two collection days."""
    from radar import safety

    return safety.quiet_sources(safety_store.source_runs(ctx.conn))


@traced("services.collect_gaps", "chain")
def collect_gaps(ctx: AppContext, search=None) -> list[str]:
    """M4: search arXiv for each open knowledge gap (an unanswered question), so the store has
    something on it next time. A gap counts as resolved once a search adds new items; after
    GAP_MAX_ATTEMPTS fruitless runs it's left alone (still listed by `radar gaps`)."""
    from radar.personalize import GAP_MAX_ATTEMPTS

    search = search or arxiv.search
    log = []
    for gap in memory.open_gaps(ctx.conn):
        if gap["attempts"] >= GAP_MAX_ATTEMPTS:
            continue
        try:
            found = search(gap["topic"], max_results=5)
        except SourceUnavailable as e:
            log.append(f"gap {gap['id']}: arXiv unavailable ({e})")
            break
        added = index_items(ctx.conn, ctx.embeddings, ctx.vectors, found)
        with ctx.conn:
            memory.gap_searched(ctx.conn, gap["id"], found_new=added > 0)
        log.append(f"gap {gap['id']} ({gap['topic'][:50]}): {added} new item(s)")
    return log


@dataclass
class DailyResult:
    date: str
    path: str
    picks: list[dict]
    collect_log: list[str]
    timings: dict[str, float]
    stats: dict = field(default_factory=dict)
    candidates: int = 0
    notes: list[str] = field(default_factory=list)
    team_log: list[dict] = field(default_factory=list)
    profile_log: list[str] = field(default_factory=list)
    watch_log: list[str] = field(default_factory=list)
    review: list[str] = field(default_factory=list)  # why the digest needs review (safety)


@traced("services.daily", "chain")
def daily(
    ctx: AppContext,
    on: date | None = None,
    collect: bool = True,
    mode: str | None = None,
    llm: BaseChatModel | None = None,
    learn: bool = True,
    on_step: StepCallback | None = None,
) -> DailyResult:
    """Run the daily graph and write the digest. mode: "single" (M2) or "multi" (M2.5).

    While it runs, a lock file (`data/daily.lock`) says so, so the web UI can warn that Qwen is
    busy and refuse to start a second run. A second run started meanwhile raises
    `DailyRunning` (the scheduled run turns that into "skipped")."""
    from radar.graphs.daily import build_daily_graph
    from radar.models import get_chat_model

    mode = mode or ctx.settings.digest_mode
    llm = llm or get_chat_model(ctx.settings)  # thinking off: summaries don't need it
    graph = build_daily_graph(
        ctx, llm, collect=collect_all if collect else None, mode=mode, learn=learn
    )
    on = on or date.today()
    with _daily_lock(ctx):
        state = run_graph(
            graph, {"date": on.isoformat()}, {"run_name": f"radar-daily-{mode}"}, on_step
        )
    return DailyResult(
        date=state["date"],
        path=state["path"],
        picks=state["picks"],
        collect_log=state["collect_log"],
        timings=state["timings"],
        stats=state.get("stats", {}),
        candidates=len(state["candidates"]),
        notes=state.get("notes", []),
        team_log=state.get("team_log", []),
        profile_log=state.get("profile_log", []),
        watch_log=state.get("watch_log", []),
        review=state.get("review", []),
    )


class DailyRunning(RuntimeError):
    """Another daily run holds the lock."""


def _lock_path(ctx: AppContext) -> Path:
    return Path(ctx.settings.db_path).parent / "daily.lock"


def daily_running(ctx: AppContext) -> dict | None:
    """{pid, started} if a daily run is going right now (in any process), else None.

    A lock file rather than a database row: if the run crashes, its PID is gone, and a lock
    whose process no longer exists is ignored instead of blocking every later run."""
    import os

    path = _lock_path(ctx)
    try:
        info = json.loads(path.read_text())
        os.kill(int(info["pid"]), 0)  # signal 0 checks the process exists, sends nothing
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return info


@contextmanager
def _daily_lock(ctx: AppContext):
    import os
    from datetime import datetime

    if (other := daily_running(ctx)) is not None:
        raise DailyRunning(f"A daily run (pid {other['pid']}) started at {other['started']}.")
    path = _lock_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps({"pid": os.getpid(), "started": started}))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


# ---- M6: the scheduled morning run ---------------------------------------------------------


@traced("services.health", "tool")
def health(ctx: AppContext, client=None) -> dict:
    """Is everything the agent needs reachable? The database, Ollama, and both models.
    (The M7 web app's /api/health can use this too.)"""
    import httpx

    out = {"db": False, "ollama": False, "models": {}, "error": ""}
    try:
        ctx.conn.execute("SELECT 1").fetchone()
        out["db"] = True
    except sqlite3.Error as e:
        out["error"] = f"database: {e}"
    try:
        client = client or httpx.Client(timeout=5)
        tags = client.get(f"{ctx.settings.ollama_host}/api/tags").json()
        out["ollama"] = True
        have = {m["name"] for m in tags.get("models", [])}
        for m in (ctx.settings.chat_model, ctx.settings.embed_model):
            # Ollama lists "name:tag"; a model asked for without a tag means ":latest"
            out["models"][m] = m in have or f"{m}:latest" in have
    except (httpx.HTTPError, ValueError) as e:
        out["error"] = f"Ollama at {ctx.settings.ollama_host}: {type(e).__name__}"
    out["ok"] = out["db"] and out["ollama"] and all(out["models"].values())
    return out


@traced("services.scheduled_daily", "chain")
def scheduled_daily(
    ctx: AppContext,
    mode: str | None = None,
    wait_s: float = 600,
    poll_s: float = 30,
    sleep=None,
    run_daily=None,
    check=None,
) -> dict:
    """What the launchd job runs each morning (`radar daily --scheduled`).

    Differences from a manual `radar daily`:
    - Skip if today already has a digest (e.g. you ran it by hand, or launchd ran it on wake).
    - Wait up to `wait_s` for Ollama: right after the Mac wakes, Wi-Fi and Ollama may not be up.
    - Never raise: return {"status": "done" | "empty" | "skipped" | "failed", "message": ...},
      so the job can log one clear line and post a notification either way.
    - "empty" means the run finished but picked nothing (e.g. every fetch failed, or every
      candidate was filtered out). It gets its own status so the notification doesn't announce
      an empty digest as if it were a normal one.
    """
    import time

    sleep = sleep or time.sleep
    check = check or (lambda: health(ctx))
    today = date.today().isoformat()
    if ctx.conn.execute("SELECT 1 FROM digests WHERE date = ? LIMIT 1", (today,)).fetchone():
        return {"status": "skipped", "message": f"The digest for {today} already exists."}
    waited = 0.0
    while not (h := check())["ok"]:
        if waited >= wait_s:
            missing = [m for m, ok in h["models"].items() if not ok]
            why = h["error"] or f"models not pulled: {', '.join(missing)}"
            return {"status": "failed", "message": f"Not ready after {waited:.0f}s: {why}"}
        sleep(poll_s)
        waited += poll_s
    try:
        result = (run_daily or daily)(ctx, mode=mode)
    except DailyRunning as e:  # e.g. started from the web UI a minute earlier
        return {"status": "skipped", "message": str(e)}
    except Exception as e:  # logged and notified rather than a silent non-zero exit
        return {"status": "failed", "message": f"{type(e).__name__}: {e}"}
    review = getattr(result, "review", None) or []
    if not result.picks:
        candidates = getattr(result, "candidates", 0)
        why = "nothing new was fetched" if not candidates else "every candidate was filtered out"
        return {
            "status": "empty",
            "message": f"No picks for {result.date}: {candidates} candidates, {why}",
            "path": result.path,
            "result": result,
        }
    return {
        "status": "done",
        "message": f"{len(result.picks)} picks for {result.date}"
        + (f"; NEEDS REVIEW: {review[0]}" if review else ""),
        "review": review,
        "path": result.path,
        "result": result,
    }


# ---- Reading digests and conversations (for the web UI) ------------------------------------


def digest_dates(ctx: AppContext, limit: int = 60) -> list[str]:
    rows = ctx.conn.execute(
        "SELECT DISTINCT date FROM digests ORDER BY date DESC LIMIT ?", (limit,)
    )
    return [r[0] for r in rows]


def digest_view(ctx: AppContext, on: str | None = None) -> dict | None:
    """One day's digest as data rather than Markdown: each pick with its summary, the Critic's
    concerns and your latest rating, plus the items the Critic dropped and why the digest needs
    review. `on` = None means the latest digest. None if there's no digest for that day."""
    from radar.store import digests as digest_store

    dates = digest_dates(ctx, limit=10_000)
    if not dates:
        return None
    on = on or dates[0]
    if on not in dates:
        return None
    ratings = memory.latest_ratings(ctx.conn)
    critic = {
        r["item_id"]: dict(r)
        for r in ctx.conn.execute(
            "SELECT c.*, i.title, i.url FROM critic_decisions c JOIN items i ON i.id = c.item_id"
            " WHERE c.date = ?",
            (on,),
        )
    }
    picks = []
    for row in digest_store.get_digest(ctx.conn, on):
        summary = ctx.conn.execute(
            "SELECT text, mode FROM summaries WHERE item_id = ? ORDER BY created_at DESC LIMIT 1",
            (row["item_id"],),
        ).fetchone()
        decision = critic.get(row["item_id"], {})
        picks.append(
            {
                **{k: row[k] for k in ("rank", "slot", "item_id", "title", "url", "reason")},
                "score": round(row["score"], 3),
                "summary": summary["text"] if summary else "",
                "summary_mode": summary["mode"] if summary else "",
                "concerns": decision.get("concerns", ""),
                "rating": ratings.get(row["item_id"]),
            }
        )
    in_digest = {p["item_id"] for p in picks}
    dropped = [
        {**c, "rating": ratings.get(c["item_id"])}
        for c in critic.values()
        if c["decision"] == "drop" and c["item_id"] not in in_digest
    ]
    i = dates.index(on)
    return {
        "date": on,
        "newer": dates[i - 1] if i > 0 else None,
        "older": dates[i + 1] if i + 1 < len(dates) else None,
        "picks": picks,
        "dropped": dropped,
        "review": safety_store.get_review(ctx.conn, on),
    }


def conversations(ctx: AppContext, limit: int = 30) -> list[dict]:
    return memory.conversations(ctx.conn, limit)


def conversation(ctx: AppContext, thread_id: str) -> list[dict]:
    """Every question in one conversation, with the answer it got."""
    return memory.thread_questions(ctx.conn, thread_id)


def item_rating(ctx: AppContext, item_id: str) -> str | None:
    return memory.latest_ratings(ctx.conn).get(item_id)


def watch_status(ctx: AppContext, item_id: str) -> dict | None:
    row = ctx.conn.execute("SELECT * FROM watchlist WHERE item_id = ?", (item_id,)).fetchone()
    return dict(row) if row else None


def full_text_record(ctx: AppContext, item_id: str) -> dict | None:
    from radar.store import full_texts

    return full_texts.get(ctx.conn, item_id)


# ---- M4: ratings, profile, watchlist, gaps --------------------------------------------------


@traced("services.rate", "tool")
def rate(ctx: AppContext, item_id: str, value: str, on: date | None = None) -> dict:
    """Record a rating. It's applied to the profile at the next `radar daily` (or
    `radar profile --update`), so rating several items in a row is cheap."""
    if not item_store.exists(ctx.conn, item_id):
        raise ValueError(f"No item {item_id!r} in the store.")
    with ctx.conn:
        slot = memory.add_rating(ctx.conn, item_id, value, on or date.today())
    return {"item_id": item_id, "value": value, "slot": slot}


@traced("services.watchlist", "tool")
def watchlist(ctx: AppContext, status: str | None = None) -> list[dict]:
    return memory.get_watchlist(ctx.conn, status)


@traced("services.profile", "tool")
def profile(ctx: AppContext) -> dict:
    """The interest profile as the digest will use it, plus recent changes."""
    from radar import feedback
    from radar.personalize import load_profile

    p = load_profile(ctx)
    return {
        "topics": p.topics,
        "weights": p.weights,
        "reserved_slots": p.reserved_slots,
        "reserved_plan": feedback.reserved_plan(p.reserved_slots, p.slot_order),
        "slot_stats": p.slot_stats,
        "ratings_applied": p.rated_seq,
        "pending_ratings": len(memory.ratings_since(ctx.conn, p.rated_seq)),
        "awaiting_approval": safety_store.pending(ctx.conn),
        "changes": memory.recent_changes(ctx.conn),
    }


@traced("services.decide_profile_change", "chain")
def decide_profile_change(ctx: AppContext, pending_id: int, approve: bool) -> dict:
    """Approve or reject a topic removal the Profile agent proposed. Only the CLI and the web UI
    call this, both driven by a person clicking or typing: approving is a person's decision, so
    it is deliberately not an MCP tool another agent could call."""
    from radar.personalize import load_profile, save_profile

    with ctx.conn:
        row = safety_store.decide_pending(ctx.conn, pending_id, approve)
        if row is None:
            raise ValueError(f"No pending change {pending_id}. `radar profile` lists them.")
        if approve:
            prof = load_profile(ctx)
            if row["topic"] in prof.topics:
                prof.topics = [t for t in prof.topics if t != row["topic"]]
                save_profile(ctx, prof)
            memory.log_change(
                ctx.conn,
                "topics",
                {"added": [], "removed": [row["topic"]], "why": row["why"], "approved_by": "you"},
            )
    return {**row, "status": "approved" if approve else "rejected"}


# ---- safety: flagged items, source health, the Critic vs. your ratings ----------------------


@traced("services.flagged", "tool")
def flagged(ctx: AppContext, include_released: bool = False) -> list[dict]:
    return safety_store.flagged(ctx.conn, include_released)


@traced("services.release_flagged", "chain")
def release_flagged(ctx: AppContext, item_id: str) -> bool:
    """You read a held item and it's fine: let it into future digests."""
    with ctx.conn:
        return safety_store.release(ctx.conn, item_id)


@traced("services.source_health", "tool")
def source_health(ctx: AppContext, days: int = 14) -> dict:
    return {"runs": safety_store.source_runs(ctx.conn, days), "quiet": quiet_sources(ctx)}


@traced("services.critic_agreement", "tool")
def critic_agreement(ctx: AppContext) -> dict:
    """How often the Critic's keep/drop matches your rating of the same item.

    keep + useful and drop + not relevant agree; the other two disagree. Dropped items are
    listed in the digest's notes, so you can rate them too (`radar rate ID ...`)."""
    from radar import safety

    rows = safety_store.critic_vs_ratings(ctx.conn)
    return {"rows": rows, **safety.agreement(rows)}


def critic_unrated(ctx: AppContext, limit: int = 20) -> list[dict]:
    """Recent Critic decisions on items you haven't rated useful or not relevant yet: rating
    them is what gives `critic_agreement` something to compare."""
    rows = ctx.conn.execute(
        "SELECT c.date, c.item_id, c.decision, c.concerns, i.title FROM critic_decisions c"
        " JOIN items i ON i.id = c.item_id WHERE NOT EXISTS (SELECT 1 FROM ratings r"
        "  WHERE r.item_id = c.item_id AND r.value IN ('useful', 'not_relevant'))"
        " ORDER BY c.date DESC, c.decision DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


@traced("services.update_profile", "chain")
def update_profile(ctx: AppContext, llm: BaseChatModel | None = None, agent: bool = True):
    """Apply pending ratings now (the daily run does this too). agent=False skips the LLM."""
    from radar.models import get_chat_model
    from radar.personalize import update_profile as run

    if agent and llm is None:
        llm = get_chat_model(ctx.settings)
    return run(ctx, llm if agent else None)


@traced("services.reset_profile", "chain")
def reset_profile(ctx: AppContext) -> None:
    from radar.personalize import reset_profile as run

    run(ctx)


@traced("services.gaps", "tool")
def gaps(ctx: AppContext) -> list[dict]:
    return memory.open_gaps(ctx.conn)


# ---- M5: full text ------------------------------------------------------------------------


@dataclass
class FullTextResult:
    item_id: str
    status: str  # "ok" | "failed" | "cached"
    source: str = ""
    sections: int = 0
    chunks: int = 0
    error: str = ""
    headings: list[str] = field(default_factory=list)


@traced("services.full_text", "chain")
def full_text(
    ctx: AppContext, item_id: str, reason: str = "manual", force: bool = False, client=None
) -> FullTextResult:
    """Fetch an item's full text, split it into section chunks and index them.

    Already fetched -> nothing to do ("cached") unless `force`. A failed fetch is remembered,
    so the agent doesn't hammer arXiv for a paper with no HTML or PDF, and retried after
    FULL_TEXT_RETRY_DAYS.
    """
    from radar.rag.chunking import section_chunks
    from radar.rag.index import replace_section_chunks
    from radar.sources import full_text as ft
    from radar.store import full_texts

    item = item_store.get_item(ctx.conn, item_id)
    if item is None:
        raise ValueError(f"No item {item_id!r} in the store.")
    known = full_texts.get(ctx.conn, item_id)
    if known and not force:
        if known["status"] == "ok":
            return FullTextResult(
                item_id,
                "cached",
                known["source"],
                known["n_sections"],
                known["n_chunks"],
                headings=[s["heading"] for s in known["sections"]],
            )
        retry_on = date.fromisoformat(known["fetched_at"][:10]) + timedelta(
            days=ctx.settings.full_text_retry_days
        )
        if date.today() < retry_on:
            return FullTextResult(item_id, "failed", error=known["error"] or "")
    try:
        got = ft.fetch(item, client)
    except SourceUnavailable as e:  # FullTextUnavailable is one too
        with ctx.conn:
            full_texts.save_failed(ctx.conn, item_id, str(e), reason)
        return FullTextResult(item_id, "failed", error=str(e))
    s = ctx.settings
    chunks = section_chunks(
        item.id, item.title, got.sections, s.full_text_chunk_tokens, s.full_text_overlap_tokens
    )[: s.full_text_max_chunks]
    replace_section_chunks(ctx.conn, ctx.embeddings, ctx.vectors, item, chunks)
    sections = [{"heading": x.heading, "text": x.text} for x in got.sections]
    with ctx.conn:
        full_texts.save_ok(ctx.conn, item_id, got.source, got.url, sections, len(chunks), reason)
    return FullTextResult(
        item_id,
        "ok",
        got.source,
        len(sections),
        len(chunks),
        headings=[x.heading for x in got.sections],
    )


def fetch_repeated(ctx: AppContext, cited: list[str]) -> list[dict]:
    """The M5 trigger: after an answer, fetch the full text of every cited item that has now
    been cited by FULL_TEXT_AFTER_QUESTIONS questions or more. The question that crosses the
    line still got its answer from abstracts; the next one about that paper can use the
    method details. Plain code, not an LLM decision: counting is something code does better.
    Not traced itself (it runs after every question); a fetch it starts is traced.
    """
    from radar.store import full_texts

    after = ctx.settings.full_text_after_questions
    if after <= 0 or not cited:
        return []
    counts = full_texts.times_cited(ctx.conn, cited)
    out = []
    for item_id in cited:
        n = counts.get(item_id, 0)
        if n < after or full_texts.get(ctx.conn, item_id):
            continue
        try:
            r = full_text(ctx, item_id, reason=f"cited by {n} questions")
        except Exception as e:  # never let the background fetch break an answered question
            r = FullTextResult(item_id, "failed", error=str(e))
        out.append({**asdict(r), "times_cited": n})
    return out


def full_text_list(ctx: AppContext, limit: int = 50) -> list[dict]:
    from radar.store import full_texts

    return full_texts.listing(ctx.conn, limit)


@traced("services.search_full_text", "retriever", process_outputs=passages_as_documents)
def search_full_text(ctx: AppContext, item_id: str, query: str, k: int = 3) -> list[Passage]:
    """The best sections of one item's full text for a query (fetch it first)."""
    from radar.rag.retrieve import search_item

    return search_item(ctx.conn, ctx.embeddings, item_id, query, k=k)


# ---- Tracing: the local tool-call log (not traced itself, so reading it doesn't add rows) ---


def calls(
    ctx: AppContext,
    limit: int = 30,
    name: str | None = None,
    kind: str | None = None,
    errors_only: bool = False,
    roots_only: bool = False,
) -> list[dict]:
    from radar.store import tool_calls

    return tool_calls.recent(ctx.conn, limit, name, kind, errors_only, roots_only)


def call_tree(ctx: AppContext, call_id: str = "last") -> list[dict]:
    """Every call under one top-level call. `call_id` may be any call in the tree (or a prefix
    of its ID); "last" means the most recent top-level call."""
    from radar.store import tool_calls

    if call_id == "last":
        latest = tool_calls.recent(ctx.conn, limit=1, roots_only=True)
        root = latest[0]["id"] if latest else None
    else:
        root = tool_calls.find_root(ctx.conn, call_id)
    return tool_calls.tree(ctx.conn, root) if root else []


def call_stats(ctx: AppContext, days: int = 7) -> list[dict]:
    from datetime import UTC, datetime

    from radar.store import tool_calls

    since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
    return tool_calls.stats(ctx.conn, since)


def _log_question(ctx: AppContext, question: str, state: dict, thread_id: str | None = None):
    """Remember the question; an abstained one becomes a gap (a topic to collect next run)."""
    cited = state.get("cited") or []
    with ctx.conn:
        memory.log_question(
            ctx.conn,
            question,
            bool(cited),
            cited,
            gap_topic=state.get("question") or question,
            thread_id=thread_id,
            answer=state.get("answer"),
        )


class Agent:
    """Holds the compiled Q&A graph so it's built once per process."""

    def __init__(
        self,
        ctx: AppContext,
        llm: BaseChatModel | None = None,
        judge: BaseChatModel | None = None,
        checkpointer=None,
        tools: dict | None = None,
        verify: bool | None = None,
    ):
        from radar.graphs.qa import build_qa_graph
        from radar.models import get_chat_model

        verify = ctx.settings.verify_answers if verify is None else verify
        self.ctx = ctx
        self.llm = llm or get_chat_model(ctx.settings)
        reasoner = None
        if llm is None and ctx.settings.reason_thinking:
            reasoner = get_chat_model(ctx.settings, reasoning=True)
        judge = judge or (llm if llm is not None else get_chat_model(ctx.settings, reasoning=True))
        if checkpointer is None:
            checkpointer = _sqlite_checkpointer(ctx.settings.checkpoint_db_path)
        self.graph = build_qa_graph(
            self.llm,
            judge,
            tools or self._load_tools(),
            max_steps=ctx.settings.max_tool_steps,
            checkpointer=checkpointer,
            reasoner=reasoner,
            verifier=self.llm if verify else None,
        )

    def _load_tools(self) -> dict:
        from radar.tools import make_tools

        local = make_tools(self.ctx)
        if not self.ctx.settings.tools_via_mcp:
            return local
        # M1.6 dogfooding: same tools, reached through `radar mcp` over stdio.
        from radar.mcp_client import MCPToolset

        self.mcp = MCPToolset()
        return self.mcp.tools(list(local))

    @traced("services.Agent.ask", "chain")
    def ask(
        self,
        question: str,
        thread_id: str | None = None,
        no_retrieval: bool = False,
        remember: bool = True,
        on_step: StepCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> AskResult:
        """Answer a question. remember=False (evals) skips the long-term side effects: no row in
        `questions`, no gap for an abstention, no full-text fetch.

        on_step(node, new_trace_entries) is called after every graph step, so a caller can show
        progress while the answer is being worked out (the web UI streams it to the browser).
        should_stop() is checked between steps; returning True raises `Cancelled`."""
        thread_id = thread_id or uuid.uuid4().hex[:8]
        if no_retrieval:
            # The comparison baseline: same model, no tools, no store.
            out = self.llm.invoke(prompts.NO_RETRIEVAL.format(question=question))
            return AskResult(answer=str(out.content), cited=[], thread_id=thread_id)
        config = {"configurable": {"thread_id": thread_id}, "run_name": "radar-ask"}
        state = run_graph(
            self.graph, {"messages": [HumanMessage(question)]}, config, on_step, should_stop
        )
        if remember:
            _log_question(self.ctx, question, state, thread_id)
        return AskResult(
            answer=state["answer"],
            cited=state["cited"],
            thread_id=thread_id,
            trace=state["trace"],
            passages=state["passages"],
            verification=state.get("verification", []),
            full_texts=fetch_repeated(self.ctx, state.get("cited") or []) if remember else [],
        )


def _sqlite_checkpointer(path: Path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


@traced("services.eval_retrieval", "chain")
def eval_retrieval(ctx: AppContext, questions_file: Path, k: int = 5) -> dict:
    """Retrieval recall@k: for each question, did any expected item appear in the top k?

    The file has one JSON object per line: {"question": ..., "expected_ids": [...]}.
    """
    rows = [json.loads(line) for line in questions_file.read_text().splitlines() if line.strip()]
    # Questions the store shouldn't answer (should_abstain, no expected IDs) don't count
    # toward retrieval recall; they're for checking that `ask` abstains.
    rows = [r for r in rows if r.get("expected_ids")]
    results = []
    for row in rows:
        got = [p.item_id for p in search(ctx, row["question"], k=k)]
        expected = set(row.get("expected_ids", []))
        results.append(
            {
                "question": row["question"],
                "hit": bool(expected & set(got)),
                "got": got,
                "expected": sorted(expected),
            }
        )
    hits = sum(r["hit"] for r in results)
    return {"k": k, "recall": hits / len(results) if results else 0.0, "results": results}
