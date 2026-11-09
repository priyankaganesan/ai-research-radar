"""What the agent can do, as plain functions. The CLI (and later MCP and the web app) call these.

Nothing outside this module builds graphs or touches the store directly.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, replace
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
from radar.store import signals as signal_store


@dataclass
class AskResult:
    answer: str
    cited: list[str]
    thread_id: str
    trace: list[dict] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list)
    verification: list[dict] = field(default_factory=list)


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


def ingest_hf(ctx: AppContext, days: int = 3, client=None) -> SourceReport:
    """Hugging Face Daily Papers from the last `days` days, with today's upvote counts."""
    papers = hf_papers.fetch_recent(days, client)
    added = index_items(ctx.conn, ctx.embeddings, ctx.vectors, [p.item for p in papers])
    today = date.today()
    with ctx.conn:
        for p in papers:
            signal_store.record(ctx.conn, p.item.id, today, hf_upvotes=p.upvotes)
    return SourceReport("hf", fetched=len(papers), added=added, signals=len(papers))


def ingest_labs(ctx: AppContext, days: int = 14, client=None) -> SourceReport:
    """Posts from the AI lab feeds in LAB_FEEDS. A broken feed is reported, not fatal."""
    got = lab_blogs.fetch_all(ctx.settings.lab_feeds, days, client)
    added = index_items(ctx.conn, ctx.embeddings, ctx.vectors, got.items)
    return SourceReport("labs", fetched=len(got.items), added=added, errors=got.errors)


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


def show(ctx: AppContext, item_id: str) -> tuple[Item, dict] | None:
    """One item with its latest signals, for `radar show`."""
    item = item_store.get_item(ctx.conn, item_id)
    return (item, signal_store.latest(ctx.conn, item_id)) if item else None


def search(
    ctx: AppContext, query: str, k: int = 5, kind: str | None = None, since: str | None = None
) -> list[Passage]:
    return hybrid_search(ctx.conn, ctx.embeddings, ctx.vectors, query, k=k, kind=kind, since=since)


def get_item(ctx: AppContext, item_id: str) -> Item | None:
    return item_store.get_item(ctx.conn, item_id)


def stats(ctx: AppContext) -> dict:
    return {
        "items": item_store.count(ctx.conn),
        "vectors": ctx.vectors.count(),
        "backend": ctx.settings.vector_backend,
    }


ANN_AUTO_ROWS = 5000  # below this a flat scan takes a few ms, so reindex skips the ANN index


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


def collect_all(ctx: AppContext) -> list[str]:
    """Run every source, then Semantic Scholar enrichment. One failing source doesn't stop the
    rest; it's noted in the returned log instead."""
    log = []
    try:
        fetched, added, used = ingest(ctx, days=DEFAULT_DAYS["arxiv"])
        log.append(f"arxiv ({used}): fetched {fetched}, added {added}")
    except SourceUnavailable as e:
        log.append(f"arxiv: unavailable ({e})")
    for name, fn in (("hf", ingest_hf), ("labs", ingest_labs), ("hn", ingest_hn)):
        try:
            r = fn(ctx, days=DEFAULT_DAYS[name])
            log.append(f"{name}: fetched {r.fetched}, added {r.added}")
        except SourceUnavailable as e:
            log.append(f"{name}: unavailable ({e})")
    try:
        r = enrich(ctx)
        log.append(f"s2 enrich: {r.signals} papers updated")
    except SourceUnavailable as e:
        log.append(f"s2 enrich: unavailable ({e})")
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


def daily(
    ctx: AppContext,
    on: date | None = None,
    collect: bool = True,
    mode: str | None = None,
    llm: BaseChatModel | None = None,
) -> DailyResult:
    """Run the daily graph and write the digest. mode: "single" (M2) or "multi" (M2.5)."""
    from radar.graphs.daily import build_daily_graph
    from radar.models import get_chat_model

    mode = mode or ctx.settings.digest_mode
    llm = llm or get_chat_model(ctx.settings)  # thinking off: summaries don't need it
    graph = build_daily_graph(ctx, llm, collect=collect_all if collect else None, mode=mode)
    on = on or date.today()
    state = graph.invoke({"date": on.isoformat()}, config={"run_name": f"radar-daily-{mode}"})
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

    def ask(
        self, question: str, thread_id: str | None = None, no_retrieval: bool = False
    ) -> AskResult:
        thread_id = thread_id or uuid.uuid4().hex[:8]
        if no_retrieval:
            # The comparison baseline: same model, no tools, no store.
            out = self.llm.invoke(prompts.NO_RETRIEVAL.format(question=question))
            return AskResult(answer=str(out.content), cited=[], thread_id=thread_id)
        state = self.graph.invoke(
            {"messages": [HumanMessage(question)]},
            config={"configurable": {"thread_id": thread_id}, "run_name": "radar-ask"},
        )
        return AskResult(
            answer=state["answer"],
            cited=state["cited"],
            thread_id=thread_id,
            trace=state["trace"],
            passages=state["passages"],
            verification=state.get("verification", []),
        )


def _sqlite_checkpointer(path: Path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))


def eval_retrieval(ctx: AppContext, questions_file: Path, k: int = 5) -> dict:
    """Retrieval recall@k: for each question, did any expected item appear in the top k?

    The file has one JSON object per line: {"question": ..., "expected_ids": [...]}.
    """
    rows = [json.loads(line) for line in questions_file.read_text().splitlines() if line.strip()]
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
