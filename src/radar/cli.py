"""`radar` command line. Each command is a thin wrapper over `radar.services`."""

from pathlib import Path

import typer

from radar.config import get_settings

app = typer.Typer(help="AI Research Radar: track, summarize and explain new AI research.")


def _todo(milestone: str) -> None:
    typer.echo(f"Not built yet (planned for {milestone}, see PLAN.md).")
    raise typer.Exit(code=1)


def _ctx():
    from radar.context import build_context
    from radar.rag.vector_index import SqliteVecIndex

    ctx = build_context(get_settings())
    if ctx.settings.vector_backend != "sqlite":
        in_store = SqliteVecIndex(ctx.conn).count()
        if ctx.vectors.count() != in_store:
            typer.echo(
                f"Note: {ctx.settings.vector_backend} has {ctx.vectors.count()} vectors, the store"
                f" has {in_store}. Run `radar reindex --backend {ctx.settings.vector_backend}`.",
                err=True,
            )
    return ctx


@app.command()
def config() -> None:
    """Show the active settings (secrets hidden)."""
    s = get_settings()
    for key, value in s.model_dump().items():
        if "key" in key:
            value = "set" if value else "not set"
        typer.echo(f"{key} = {value}")


@app.command("init-db")
def init_db() -> None:
    """Create the store and apply migrations."""
    from radar.store.db import connect

    s = get_settings()
    conn = connect(s.db_path)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    typer.echo(f"Store ready at {s.db_path} ({len(tables)} tables).")


SOURCES = ("arxiv", "hf", "labs", "hn", "all")
DEFAULT_DAYS = {"arxiv": 7, "hf": 3, "labs": 14, "hn": 2}


@app.command()
def ingest(
    what: str = typer.Argument("arxiv", help="arxiv, hf, labs, hn, or all."),
    days: int | None = typer.Option(
        None, help="How many days back (default: arxiv 7, hf 3, labs 14, hn 2)."
    ),
    categories: str = typer.Option(
        "", help="arXiv only: comma-separated categories (default: settings)."
    ),
    max_results: int = typer.Option(500, "--max", help="arXiv only: upper limit on papers."),
    via: str = typer.Option(
        "auto", "--via", "--source", help="arXiv only: api, rss (latest day only), or auto."
    ),
    min_points: int | None = typer.Option(None, help="hn only: minimum points (default 50)."),
) -> None:
    """Fetch new items into the store (embeds them with the local model).

    arxiv = papers by category; hf = Hugging Face Daily Papers + upvotes; labs = AI lab blog
    posts; hn = Hacker News AI stories + points; all = every source, then `enrich`.
    """
    from radar import services
    from radar.sources.http import SourceUnavailable

    if what not in SOURCES:
        typer.echo(f"Unknown source {what!r}. Choose from: {', '.join(SOURCES)}.")
        raise typer.Exit(code=1)
    ctx = _ctx()
    todo = ["arxiv", "hf", "labs", "hn"] if what == "all" else [what]
    failed = False
    for name in todo:
        n_days = days if days is not None else DEFAULT_DAYS[name]
        try:
            if name == "arxiv":
                _ingest_arxiv(ctx, n_days, categories, max_results, via)
                continue
            typer.echo(f"Fetching {name} (last {n_days} days)...")
            if name == "hf":
                report = services.ingest_hf(ctx, days=n_days)
            elif name == "labs":
                report = services.ingest_labs(ctx, days=n_days)
            else:
                report = services.ingest_hn(ctx, days=n_days, min_points=min_points)
            _print_report(report)
        except SourceUnavailable as e:
            typer.secho(f"{name}: {e}", fg=typer.colors.RED)
            failed = True
    if what == "all":
        _run_enrich(ctx, days=30, limit=500)
    typer.echo(f"Store now has {services.stats(ctx)['items']} items.")
    if failed:
        raise typer.Exit(code=1)


def _ingest_arxiv(ctx, days: int, categories: str, max_results: int, via: str) -> None:
    from radar import services

    cats = [c.strip() for c in categories.split(",") if c.strip()] or None
    if via == "rss":
        typer.echo("Fetching the latest announcement from the arXiv RSS feed...")
    else:
        typer.echo("Fetching from the arXiv API (it asks for a 3-second pause between pages)...")
    fetched, added, used = services.ingest(
        ctx, days=days, categories=cats, max_results=max_results, source=via
    )
    if used == "rss" and via == "auto":
        typer.echo("The arXiv API was rate-limited, so this used the RSS feed (latest day only).")
    typer.echo(f"arxiv: fetched {fetched}, added {added} new.")


def _print_report(r) -> None:
    line = f"{r.source}: fetched {r.fetched}, added {r.added} new"
    if r.signals:
        line += f", recorded signals for {r.signals}"
    typer.echo(line + ".")
    for name, problem in r.errors.items():
        typer.secho(f"  skipped {name}: {problem}", fg=typer.colors.YELLOW)


def _run_enrich(ctx, days: int, limit: int) -> None:
    from radar import services
    from radar.sources.http import SourceUnavailable

    typer.echo("Looking up citations and TLDRs on Semantic Scholar...")
    try:
        r = services.enrich(ctx, days=days, limit=limit)
    except SourceUnavailable as e:
        typer.secho(f"s2: {e} A free API key (SEMANTIC_SCHOLAR_API_KEY) helps.", fg="yellow")
        return
    typer.echo(f"s2: checked {r.fetched} papers, found {r.signals} on Semantic Scholar.")


@app.command()
def enrich(
    days: int = typer.Option(30, help="Papers published in the last N days."),
    limit: int = typer.Option(500, help="At most this many papers."),
) -> None:
    """Add Semantic Scholar citation counts and TLDRs to recent papers in the store."""
    _run_enrich(_ctx(), days=days, limit=limit)


@app.command()
def show(item_id: str) -> None:
    """Show one item: its text, TLDR and latest signals (upvotes, citations, HN points)."""
    from radar import services

    found = services.show(_ctx(), item_id)
    if not found:
        typer.echo(f"No item {item_id!r} in the store.")
        raise typer.Exit(code=1)
    item, sig = found
    typer.echo(f"[{item.id}] {item.title}")
    typer.echo(f"{item.kind} from {item.source}, published {item.published}  {item.url}")
    if item.tldr:
        typer.echo(f"TLDR: {item.tldr}")
    numbers = ", ".join(f"{k} {v}" for k, v in sig.items() if v is not None)
    typer.echo(f"Signals: {numbers or 'none yet'}")
    typer.echo(f"\n{item.abstract}")


@app.command()
def search(
    query: str,
    k: int = typer.Option(5, help="How many results."),
    since: str = typer.Option("", help="Only items published on or after YYYY-MM-DD."),
) -> None:
    """Search the store directly (no LLM). Useful for checking retrieval on its own."""
    from radar import services

    passages = services.search(_ctx(), query, k=k, since=since or None)
    if not passages:
        typer.echo("No results. Is the store empty? Try `radar ingest`.")
    for n, p in enumerate(passages, 1):
        typer.echo(f"{n}. [{p.item_id}] {p.title} ({p.published})  score={p.score}")
        typer.echo(f"   {p.url}")


@app.command()
def ask(
    question: str,
    no_retrieval: bool = typer.Option(False, "--no-retrieval", help="Answer without tools."),
    thread: str | None = typer.Option(None, help="Conversation ID, to ask a follow-up."),
    show_steps: bool = typer.Option(False, "--steps", help="Print each step of the loop."),
    verify: bool | None = typer.Option(
        None, "--verify/--no-verify", help="Check each claim with the Verifier agent (M2.5)."
    ),
) -> None:
    """Ask a question, answered from the store with citations."""
    from radar import services

    agent = services.Agent(_ctx(), verify=verify)
    result = agent.ask(question, thread_id=thread, no_retrieval=no_retrieval)
    if show_steps:
        for t in result.trace:
            typer.secho(f"  {t}", fg=typer.colors.BRIGHT_BLACK)
        typer.echo()
    typer.echo(result.answer)
    if not no_retrieval:
        typer.secho(
            f"\n(conversation: {result.thread_id}; follow up with --thread {result.thread_id})",
            fg=typer.colors.BRIGHT_BLACK,
        )


@app.command()
def mcp(
    http: bool = typer.Option(False, "--http", help="Serve streamable HTTP on localhost."),
    port: int = typer.Option(8765, help="Port for --http."),
) -> None:
    """Run the MCP server (stdio by default) so MCP clients can use Radar's tools."""
    from radar.mcp_server import run

    run(http=http, port=port)


@app.command()
def daily(
    no_collect: bool = typer.Option(
        False, "--no-collect", help="Skip fetching; build the digest from the store as it is."
    ),
    on: str | None = typer.Option(None, "--date", help="Digest date, YYYY-MM-DD (default today)."),
    mode: str | None = typer.Option(None, help="single (one summarize step) or multi (agents)."),
) -> None:
    """Run the daily pipeline and write today's digest."""
    from datetime import date

    from radar import services

    if mode not in (None, "single", "multi"):
        typer.echo("--mode must be single or multi.")
        raise typer.Exit(code=1)
    if not no_collect:
        typer.echo("Collecting from all sources (this takes a few minutes)...")
    result = services.daily(
        _ctx(),
        on=date.fromisoformat(on) if on else None,
        collect=not no_collect,
        mode=mode,
    )
    for line in result.collect_log:
        typer.echo(f"  {line}")
    typer.echo(f"\n{result.candidates} candidates -> {len(result.picks)} picks:")
    for n, p in enumerate(result.picks, 1):
        typer.echo(f"{n}. [{p['slot']}] {p['title']} ({p['item_id']})")
        typer.secho(f"   {p['reason']}", fg=typer.colors.BRIGHT_BLACK)
        if p.get("concerns"):
            typer.secho(f"   critic: {p['concerns']}", fg=typer.colors.YELLOW)
    for note in result.notes:
        typer.secho(f"  * {note}", fg=typer.colors.YELLOW)
    if result.stats:
        typer.secho(f"Agents: {result.stats}", fg=typer.colors.BRIGHT_BLACK)
    timing = ", ".join(f"{k} {v}s" for k, v in result.timings.items())
    typer.secho(f"\nTimings: {timing}", fg=typer.colors.BRIGHT_BLACK)
    typer.echo(f"Digest written to {result.path}")


@app.command()
def rate(item_id: str, value: str) -> None:
    """Rate a digest item: useful, not_relevant or skipped."""
    _todo("M4")


@app.command()
def watchlist() -> None:
    """Show the early-buzz watchlist."""
    _todo("M4")


@app.command()
def reindex(
    backend: str = typer.Option(..., help="sqlite (re-embed every chunk) or lance (copy vectors)."),
    ann: bool = typer.Option(False, "--ann", help="LanceDB: also build an IVF-PQ index."),
) -> None:
    """Rebuild a vector index from the SQLite store."""
    from radar import services

    try:
        result = services.reindex(_ctx(), backend, ann=ann)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from None
    typer.echo(f"{result['backend']}: {result['vectors']} vectors indexed.")
    if result["ann"]:
        a = result["ann"]
        typer.echo(
            f"Built IVF-PQ: {a['num_partitions']} partitions, {a['num_sub_vectors']} sub-vectors."
        )


@app.command("compare-backends")
def compare_backends(
    file: Path | None = typer.Option(None, help="Eval questions (default: sample item titles)."),
    k: int = typer.Option(5, help="Results per search."),
    sample: int = typer.Option(50, help="How many item titles to use when no file is given."),
) -> None:
    """Compare sqlite-vec (exact) and LanceDB (approximate) on recall, latency and disk."""
    from radar import services

    try:
        report = services.compare_backends(_ctx(), file, k=k, sample=sample)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from None
    typer.echo(report["markdown"])
    typer.echo(f"Saved to {report['path']}")


@app.command("compare-agents")
def compare_agents(
    on: str | None = typer.Option(None, "--date", help="Digest date, YYYY-MM-DD (default today)."),
    file: Path | None = typer.Option(
        None, help="Optional JSONL of questions to also compare Q&A with and without the Verifier."
    ),
    limit: int = typer.Option(5, help="Max questions from the file."),
) -> None:
    """Single- vs. multi-agent: claim support, model calls and time (M2.5). Slow: ~20 min."""
    import json
    from datetime import date

    from radar import evals_agents
    from radar.models import get_chat_model

    ctx = _ctx()
    questions = []
    if file is not None:
        rows = [json.loads(x) for x in file.read_text().splitlines() if x.strip()]
        questions = [r["question"] for r in rows][:limit]
    typer.echo("Building the digest twice (single, then multi) and judging every claim...")
    report = evals_agents.compare(
        ctx,
        date.fromisoformat(on) if on else date.today(),
        get_chat_model(ctx.settings),
        questions=questions,
    )
    typer.echo(report["markdown"])
    typer.echo(f"Saved to {report['path']}")


@app.command("eval")
def run_eval(
    file: Path = typer.Option(Path("evals/questions.jsonl"), help="Questions with expected IDs."),
    k: int = typer.Option(5, help="Count a hit if an expected item is in the top k."),
) -> None:
    """Retrieval eval: recall@k over a question file (LangSmith version comes in M6)."""
    from radar import services

    if not file.exists():
        typer.echo(f"{file} not found. Copy evals/questions.example.jsonl and fill it in.")
        raise typer.Exit(code=1)
    ctx = _ctx()
    report = services.eval_retrieval(ctx, file, k=k)
    for r in report["results"]:
        mark = "hit " if r["hit"] else "MISS"
        typer.echo(f"{mark}  {r['question']}")
        if not r["hit"]:
            typer.echo(f"      expected {r['expected']}, got {r['got']}")
    typer.echo(
        f"\nrecall@{k} = {report['recall']:.2f} over {len(report['results'])} questions"
        f" ({ctx.settings.vector_backend} backend)"
    )


if __name__ == "__main__":
    app()
