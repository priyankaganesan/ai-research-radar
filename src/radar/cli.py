"""`radar` command line. Each command is a thin wrapper over `radar.services`."""

from pathlib import Path

import typer

from radar.config import get_settings

app = typer.Typer(help="AI Research Radar: track, summarize and explain new AI research.")


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
    for f in result.full_texts:  # M5: this paper keeps coming up, so its full text was fetched
        if f["status"] == "ok":
            typer.secho(
                f"\n(Fetched the full text of {f['item_id']}: cited by {f['times_cited']}"
                f" questions. {f['chunks']} section chunks are now searchable.)",
                fg=typer.colors.BRIGHT_BLACK,
            )
        elif f["status"] == "failed":
            typer.secho(
                f"\n(Tried to fetch the full text of {f['item_id']}: {f['error']})",
                fg=typer.colors.BRIGHT_BLACK,
            )
    if not no_retrieval:
        typer.secho(
            f"\n(conversation: {result.thread_id}; follow up with --thread {result.thread_id})",
            fg=typer.colors.BRIGHT_BLACK,
        )


@app.command("fulltext")
def fulltext(
    item_id: str | None = typer.Argument(None, help="Item to fetch; omit to list full texts."),
    query: str | None = typer.Option(
        None, "--query", "-q", help="Show the best sections for this."
    ),
    force: bool = typer.Option(False, help="Fetch again even if it's already stored."),
) -> None:
    """M5: fetch a paper's full text (sections, ~500-token chunks), or list fetched ones."""
    from radar import services

    ctx = _ctx()
    if item_id is None:
        rows = services.full_text_list(ctx)
        if not rows:
            typer.echo(
                "No full texts yet. They're fetched when a paper is cited by "
                f"{ctx.settings.full_text_after_questions} questions, or with "
                "`radar fulltext ITEM_ID`."
            )
        for r in rows:
            what = (
                f"{r['source']}, {r['n_sections']} sections, {r['n_chunks']} chunks"
                if r["status"] == "ok"
                else f"failed: {r['error']}"
            )
            typer.echo(f"{r['fetched_at'][:16]}  [{r['item_id']}] {r['title'][:60]}")
            typer.secho(f"    {what}  ({r['reason']})", fg=typer.colors.BRIGHT_BLACK)
        return
    try:
        r = services.full_text(ctx, item_id, force=force)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from e
    if r.status == "failed":
        typer.echo(f"No full text: {r.error}")
        raise typer.Exit(code=1)
    note = " (already stored; --force fetches again)" if r.status == "cached" else ""
    typer.echo(f"{item_id}: {r.source}, {r.sections} sections, {r.chunks} chunks{note}")
    for h in r.headings:
        typer.secho(f"  - {h}", fg=typer.colors.BRIGHT_BLACK)
    if query:
        typer.echo()
        for p in services.search_full_text(ctx, item_id, query):
            typer.echo(f"[{p.section}]  score={p.score}")
            body = p.text.split("\n\n", 1)[-1]
            typer.secho(f"  {body[:400]}...", fg=typer.colors.BRIGHT_BLACK)


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
    scheduled: bool = typer.Option(
        False,
        "--scheduled",
        help="How the launchd job runs it: wait for Ollama, skip if today is done, notify.",
    ),
) -> None:
    """Run the daily pipeline and write today's digest."""
    from datetime import date, datetime

    from radar import services

    if mode not in (None, "single", "multi"):
        typer.echo("--mode must be single or multi.")
        raise typer.Exit(code=1)
    if scheduled:
        from radar.schedule import notify

        typer.echo(f"=== {datetime.now():%Y-%m-%d %H:%M:%S} scheduled run ===")
        out = services.scheduled_daily(_ctx(), mode=mode)
        typer.echo(f"{out['status']}: {out['message']}")
        # Clicking an osascript banner opens Script Editor (the app that "sent" it), not the
        # digest, so the banner says how to read the digest instead.
        if out["status"] == "done":
            _print_daily(out["result"])
            title = "AI Research Radar: needs review" if out.get("review") else "AI Research Radar"
            notify(title, f"Today's digest is ready: {out['message'][:140]}. Read it: radar digest")
        elif out["status"] == "empty":
            _print_daily(out["result"])
            notify(
                "AI Research Radar: EMPTY digest",
                f"{out['message'][:120]}. See data/logs/daily.log",
            )
            raise typer.Exit(code=2)  # shows up as "last exit code = 2" in `radar schedule status`
        elif out["status"] == "failed":
            notify("AI Research Radar", f"The morning digest failed: {out['message'][:120]}")
            raise typer.Exit(code=1)
        return
    if not no_collect:
        typer.echo("Collecting from all sources (this takes a few minutes)...")
    result = services.daily(
        _ctx(),
        on=date.fromisoformat(on) if on else None,
        collect=not no_collect,
        mode=mode,
    )
    _print_daily(result)


@app.command()
def digest(
    on: str | None = typer.Option(None, "--date", help="YYYY-MM-DD (default: the latest one)."),
    open_: bool = typer.Option(False, "--open", help="Open the file in its default Mac app."),
) -> None:
    """Print a digest (the latest by default), or open it with --open."""
    import subprocess

    folder = _ctx().settings.digest_dir
    if on:
        path = folder / f"{on}.md"
    else:  # file names are YYYY-MM-DD.md, so sorting by name sorts by date
        found = sorted(folder.glob("????-??-??.md"))
        path = found[-1] if found else folder / "none.md"
    if not path.exists():
        typer.echo(f"No digest at {path}. Run `radar daily` first.")
        raise typer.Exit(code=1)
    if open_:
        subprocess.run(["open", str(path)], check=False)
        typer.echo(f"Opened {path}")
    else:
        typer.echo(path.read_text())


def _print_daily(result) -> None:
    for line in result.collect_log:
        typer.echo(f"  {line}")
    for line in result.profile_log:
        typer.secho(f"  {line}", fg=typer.colors.CYAN)
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
    for line in result.watch_log:
        typer.secho(f"  {line}", fg=typer.colors.CYAN)
    if result.review:
        typer.secho("\nNEEDS REVIEW: read the sources before relying on this digest.", fg="red")
        for r in result.review:
            typer.secho(f"  - {r}", fg=typer.colors.RED)
    timing = ", ".join(f"{k} {v}s" for k, v in result.timings.items())
    typer.secho(f"\nTimings: {timing}", fg=typer.colors.BRIGHT_BLACK)
    typer.echo(f"Digest written to {result.path}")


schedule_app = typer.Typer(help="M6: the morning digest job (a macOS launchd agent).")
app.add_typer(schedule_app, name="schedule")


@schedule_app.command("install")
def schedule_install(
    at: str = typer.Option("07:30", help="Time of day, HH:MM (24-hour)."),
    mode: str | None = typer.Option(None, help="single or multi (default: DIGEST_MODE)."),
    weekdays: bool = typer.Option(False, "--weekdays", help="Monday to Friday only."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plist, change nothing."),
) -> None:
    """Install (or update) the launch agent that runs `radar daily` every morning."""
    import plistlib
    import sys

    from radar import schedule

    hour, _, minute = at.partition(":")
    repo = Path.cwd().resolve()
    if not (repo / "pyproject.toml").exists():
        typer.echo("Run this from the project folder (where pyproject.toml is).")
        raise typer.Exit(code=1)
    try:
        uv = schedule.find_uv()
        plist = schedule.build_plist(repo, uv, int(hour), int(minute or 0), mode, weekdays)
    except (ValueError, FileNotFoundError) as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from e
    if dry_run or sys.platform != "darwin":
        if sys.platform != "darwin":
            typer.echo("(Not macOS: printing the plist instead of installing it.)")
        typer.echo(plistlib.dumps(plist).decode())
        return
    path = schedule.install(plist)
    days = "Monday to Friday" if weekdays else "every day"
    typer.echo(f"Installed {path}")
    typer.echo(
        f"`radar daily --scheduled` will run {days} at {int(hour):02d}:{int(minute or 0):02d}"
    )
    typer.echo(f"Log: {plist['StandardOutPath']}. Try it now with `radar schedule run`.")


@schedule_app.command("status")
def schedule_status() -> None:
    """Is the job installed and loaded, when does it run, and how did the last run end?"""
    from radar import schedule

    st = schedule.status()
    if not st["installed"]:
        typer.echo("Not installed. Run `radar schedule install`.")
        return
    typer.echo(f"plist:   {st['plist']}")
    typer.echo(f"loaded:  {st['loaded']}  state: {st.get('state', '-')}  runs: {st.get('runs', 0)}")
    typer.echo(f"last exit code: {st.get('last_exit', '-')}")
    typer.echo(f"when:    {st['when']}")
    typer.echo(f"command: {st['command']}")
    typer.echo(f"log:     {st['log']}")


@schedule_app.command("run")
def schedule_run() -> None:
    """Start the job now, through launchd (tests the real setup; output goes to the log)."""
    from radar import schedule

    try:
        schedule.run_now()
    except RuntimeError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from e
    typer.echo(f"Started. Follow it with: tail -f data/logs/{schedule.LOG_NAME}")


@schedule_app.command("remove")
def schedule_remove() -> None:
    """Unload the job and delete its plist."""
    from radar import schedule

    typer.echo("Removed." if schedule.remove() else "Nothing was installed.")


@app.command()
def health() -> None:
    """Check the database, Ollama and both models (what the morning job checks first)."""
    from radar import services

    ctx = _ctx()
    h = services.health(ctx)
    typer.echo(f"database: {'ok' if h['db'] else 'NOT OK'}")
    typer.echo(f"ollama:   {'ok' if h['ollama'] else 'NOT REACHABLE'} ({ctx.settings.ollama_host})")
    for m, ok in h["models"].items():
        typer.echo(f"model {m}: {'ok' if ok else f'missing (ollama pull {m})'}")
    if h["error"]:
        typer.echo(h["error"])
    quiet = services.quiet_sources(ctx)
    if quiet:  # a warning, not a failure: the digest can still run on the other sources
        typer.secho(
            f"quiet sources: {', '.join(quiet)} (nothing on the last two runs)", fg="yellow"
        )
    if not h["ok"]:
        raise typer.Exit(code=1)


@app.command()
def rate(
    item_id: str,
    value: str = typer.Argument(..., help="useful, not_relevant or skipped."),
) -> None:
    """Rate a digest item. Applied to your profile at the next `radar daily`."""
    from radar import services

    try:
        r = services.rate(_ctx(), item_id, value)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from None
    where = f" (shown in the {r['slot']} slot)" if r["slot"] else " (not from a digest)"
    typer.echo(f"Rated {item_id} {value}{where}.")


@app.command()
def watchlist(
    status: str | None = typer.Option(None, help="watching, established or dropped."),
) -> None:
    """Show the early-buzz watchlist: upvoted papers waiting to prove themselves in citations."""
    from radar import services

    rows = services.watchlist(_ctx(), status)
    if not rows:
        typer.echo("The watchlist is empty. Early-buzz picks are added by `radar daily`.")
        return
    for w in rows:
        typer.echo(f"[{w['status']}] {w['item_id']}  {w['title'][:70]}")
        detail = f"added {w['added_at']}, citations then {w['citations_at_add']}"
        if w["status"] == "watching":
            detail += f", re-check after {w['recheck_after']}"
        if w["note"]:
            detail += f"; last check: {w['note']}"
        typer.secho(f"    {detail}", fg=typer.colors.BRIGHT_BLACK)


@app.command()
def profile(
    update: bool = typer.Option(False, "--update", help="Apply pending ratings now."),
    no_agent: bool = typer.Option(
        False, "--no-agent", help="With --update: skip the Profile agent (no model call)."
    ),
    reset: bool = typer.Option(False, "--reset", help="Back to the default profile."),
    approve: int | None = typer.Option(
        None, "--approve", help="Approve a pending topic removal by its number."
    ),
    reject: int | None = typer.Option(
        None, "--reject", help="Reject a pending topic removal by its number."
    ),
) -> None:
    """Show the interest profile: topics, scoring weights, reserved slots, recent changes."""
    from radar import services

    ctx = _ctx()
    for pid, ok in ((approve, True), (reject, False)):
        if pid is None:
            continue
        try:
            row = services.decide_profile_change(ctx, pid, approve=ok)
        except ValueError as e:
            typer.echo(str(e))
            raise typer.Exit(code=1) from e
        typer.echo(f"{row['status'].capitalize()}: remove {row['topic']!r}.")
    if reset:
        services.reset_profile(ctx)
        typer.echo("Profile reset to the defaults (your ratings are kept).")
    if update:
        for line in services.update_profile(ctx, agent=not no_agent).lines():
            typer.secho(line, fg=typer.colors.CYAN)
        typer.echo("")
    p = services.profile(ctx)
    typer.echo("Topics:")
    for t in p["topics"]:
        typer.echo(f"  - {t}")
    typer.echo("Weights: " + ", ".join(f"{k} {v:.3f}" for k, v in p["weights"].items()))
    typer.echo(f"Reserved slots: {p['reserved_slots']} -> {p['reserved_plan']}")
    for slot, st in p["slot_stats"].items():
        typer.echo(f"  {slot}: {st['useful']} useful of {st['rated']} rated")
    typer.echo(f"Ratings applied: {p['ratings_applied']}, waiting: {p['pending_ratings']}")
    if p["awaiting_approval"]:
        typer.secho("Waiting for your approval (--approve N or --reject N):", fg="yellow")
        for c in p["awaiting_approval"]:
            typer.secho(f"  {c['id']}. {c['action']} {c['topic']!r}: {c['why']}", fg="yellow")
    if p["changes"]:
        typer.echo("Recent changes:")
        for c in p["changes"]:
            typer.secho(f"  {c['at']} {c['kind']}: {c['detail']}", fg=typer.colors.BRIGHT_BLACK)


@app.command()
def gaps() -> None:
    """Questions the agent couldn't answer. Each run searches arXiv for them."""
    from radar import services

    rows = services.gaps(_ctx())
    if not rows:
        typer.echo("No open gaps.")
        return
    for g in rows:
        typer.echo(f"{g['id']}. {g['question'] or g['topic']}")
        typer.secho(
            f"    since {g['created_at']}, searched {g['attempts']} time(s)",
            fg=typer.colors.BRIGHT_BLACK,
        )


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


def _local_time(iso: str) -> str:
    from datetime import datetime

    return datetime.fromisoformat(iso).astimezone().strftime("%m-%d %H:%M:%S")


def _duration(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


@app.command()
def calls(
    limit: int = typer.Option(30, "--limit", "-n", help="How many calls to show."),
    name: str = typer.Option("", help="Only calls whose name contains this, e.g. arxiv."),
    kind: str = typer.Option("", help="Only one kind: service, source, mcp or agent_tool."),
    errors: bool = typer.Option(False, "--errors", help="Only calls that failed."),
    top: bool = typer.Option(False, "--top", help="Only top-level calls (one per run)."),
    tree: str = typer.Option(
        "", help="Show one run as a tree: a call ID from the list, or 'last'."
    ),
    stats: bool = typer.Option(False, "--stats", help="Counts, errors and timings per name."),
    days: int = typer.Option(7, help="--stats only: how many days back."),
    show_args: bool = typer.Option(False, "--args", help="Also print each call's arguments."),
) -> None:
    """The local tool-call log: what ran, with what arguments, how long it took, what failed.

    Every traced call is logged here (TOOL_LOG=true), with or without LangSmith. When LangSmith
    is on, `--tree` also prints the trace ID to paste into LangSmith's search.
    """
    from radar import services

    ctx = _ctx()
    if stats:
        rows = services.call_stats(ctx, days)
        if not rows:
            typer.echo("No calls logged yet.")
        for r in rows:
            err = f"  {r['errors']} failed" if r["errors"] else ""
            typer.echo(
                f"{r['calls']:>5}x  {r['name']:<34} avg {_duration(r['avg_ms']):>8}"
                f"  max {_duration(r['max_ms']):>8}{err}"
            )
        return
    if tree:
        rows = services.call_tree(ctx, tree)
        if not rows:
            typer.echo(f"No call {tree!r} in the log.")
            raise typer.Exit(code=1)
        for r in rows:
            _print_call(r, indent="  " * r["depth"], show_args=show_args)
        trace = rows[0]["langsmith_trace_id"]
        typer.secho(
            f"\nLangSmith trace: {trace}"
            if trace
            else "\n(not traced to LangSmith: tracing was off)",
            fg=typer.colors.BRIGHT_BLACK,
        )
        return
    rows = services.calls(ctx, limit, name or None, kind or None, errors, top)
    if not rows:
        typer.echo("No calls logged yet. Run something (e.g. `radar search ...`), then try again.")
    for r in reversed(rows):  # oldest first, so the newest ends up next to the prompt
        nested = "↳ " if r["parent_id"] else ""  # ran inside another call; see --tree
        _print_call(r, indent=nested, show_args=show_args, show_id=True)


def _print_call(r: dict, indent: str = "", show_args: bool = False, show_id: bool = False) -> None:
    status = (
        typer.style("ERROR", fg=typer.colors.RED) if r["error"] else typer.style("ok", fg="green")
    )
    head = f"{_local_time(r['started_at'])}  {indent}{r['name']}  {_duration(r['duration_ms'])}"
    ident = typer.style(f"  [{r['id']}]", fg=typer.colors.BRIGHT_BLACK) if show_id else ""
    typer.echo(f"{head}  {status}{ident}")
    detail = r["error"] or r["summary"] or ""
    pad = " " * (16 + len(indent))
    if detail:
        typer.secho(f"{pad}-> {detail}", fg=typer.colors.BRIGHT_BLACK)
    if show_args:
        typer.secho(f"{pad}args {r['args']}", fg=typer.colors.BRIGHT_BLACK)


@app.command("eval")
def run_eval(
    file: Path = typer.Option(Path("evals/questions.jsonl"), help="Questions with expected IDs."),
    target: list[str] = typer.Option(
        ["retrieval"],
        "--target",
        "-t",
        help="retrieval (fast, no LLM), agent, no_retrieval, or all. Repeat to run several.",
    ),
    model: list[str] = typer.Option(
        [], "--model", "-m", help="Chat model(s) to compare, e.g. -m qwen3.5:9b -m qwen3.5:4b."
    ),
    backend: str | None = typer.Option(None, help="Vector backend: sqlite or lance."),
    k: int = typer.Option(5, help="Count a retrieval hit if an expected item is in the top k."),
    judge: bool = typer.Option(False, "--judge", help="Also score claim support (LLM judge)."),
    langsmith: bool | None = typer.Option(
        None,
        "--langsmith/--local",
        help="Upload as LangSmith experiments (default: when LANGSMITH_API_KEY is set).",
    ),
    limit: int | None = typer.Option(None, help="Only the first N questions (a quick check)."),
) -> None:
    """M6 eval suite: retrieval, the agent and the no-retrieval baseline, scored the same way
    locally and as LangSmith experiments. Writes evals/results/eval-*.md."""
    import os

    from radar import evals
    from radar.context import AppContext, make_vector_index

    if not file.exists():
        typer.echo(f"{file} not found. Copy evals/questions.example.jsonl and fill it in.")
        raise typer.Exit(code=1)
    targets = list(evals.TARGETS) if "all" in target else target
    bad = [t for t in targets if t not in evals.TARGETS]
    if bad:
        typer.echo(f"Unknown target(s) {bad}; use {', '.join(evals.TARGETS)} or all.")
        raise typer.Exit(code=1)
    ctx = _ctx()
    if backend:
        settings = ctx.settings.model_copy(update={"vector_backend": backend})
        ctx = AppContext(settings, ctx.conn, ctx.embeddings, make_vector_index(settings, ctx.conn))
    if langsmith is None:
        langsmith = bool(os.environ.get("LANGSMITH_API_KEY"))
    if langsmith and not os.environ.get("LANGSMITH_API_KEY"):
        typer.echo("--langsmith needs LANGSMITH_API_KEY in .env.")
        raise typer.Exit(code=1)
    slow = [t for t in targets if t != "retrieval"]
    if slow:
        typer.echo(f"Running {', '.join(slow)} on local Qwen: expect about a minute per question.")
    report = evals.run_suite(
        ctx,
        file,
        targets,
        models=model or None,
        k=k,
        judge=judge,
        use_langsmith=langsmith,
        limit=limit,
    )
    if langsmith:
        d = report["dataset"]
        typer.echo(
            f"LangSmith dataset {d['dataset']}: {d['added']} added, {d['updated']} updated,"
            f" {d['removed']} removed."
        )
    typer.echo()
    typer.echo(report["markdown"].split("\n## ")[0])  # the summary table
    typer.echo(f"Full report: {report['path']}")


if __name__ == "__main__":
    app()


# ---- safety --------------------------------------------------------------------------------


@app.command()
def flagged(
    release: str | None = typer.Option(
        None, "--release", help="You read this item and it's fine: let it into digests."
    ),
    all_: bool = typer.Option(False, "--all", help="Include items already released."),
) -> None:
    """Items held back from the digest because their text reads like instructions."""
    from radar import services

    ctx = _ctx()
    if release:
        ok = services.release_flagged(ctx, release)
        typer.echo(f"Released {release}." if ok else f"{release} isn't held.")
        return
    rows = services.flagged(ctx, include_released=all_)
    if not rows:
        typer.echo("Nothing flagged.")
        return
    for r in rows:
        state = (
            "released"
            if r["released_at"]
            else "flagged (about attacks)"
            if r["about_attacks"]
            else "HELD"
        )
        typer.echo(f"[{r['item_id']}] {r['title'][:70]}  ({r['source']}, {state})")
        typer.secho(f'    "{r["snippet"]}"', fg=typer.colors.BRIGHT_BLACK)
        typer.secho(f"    {r['url']}", fg=typer.colors.BRIGHT_BLACK)


@app.command()
def sources(days: int = typer.Option(14, help="How many days of runs to show.")) -> None:
    """What each source returned on recent collection runs, and which have gone quiet."""
    from radar import services

    h = services.source_health(_ctx(), days)
    if not h["runs"]:
        typer.echo("No collection runs recorded yet (they start with the next `radar daily`).")
        return
    for r in h["runs"]:
        what = f"error: {r['error'][:60]}" if r["error"] else f"fetched {r['fetched']}"
        typer.echo(f"{r['date']}  {r['source']:<6} {what}, added {r['added']}")
    if h["quiet"]:
        typer.secho(f"Quiet on the last two runs: {', '.join(h['quiet'])}", fg="yellow")


@app.command()
def critic() -> None:
    """How often the Critic's keep/drop matches your ratings of the same items."""
    from radar import services

    out = services.critic_agreement(_ctx())
    if not out["n"]:
        typer.echo(
            "No overlap yet: rate items from multi-mode digests, including the ones the Critic"
            " dropped (listed in the digest's notes)."
        )
        return

    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"

    typer.echo(f"Agreement: {pct(out['agreement'])} of {out['n']} rated decisions")
    typer.echo(
        f"  kept, and you found useful:        {pct(out['keep_agreement'])} of {out['keep_n']}"
    )
    typer.echo(
        f"  dropped, and you found not relevant: {pct(out['drop_agreement'])} of {out['drop_n']}"
    )
    for r in out["rows"]:
        ok = (r["decision"], r["rating"]) in {("keep", "useful"), ("drop", "not_relevant")}
        mark = "agree   " if ok else "DISAGREE"
        typer.echo(
            f"  {mark} {r['decision']:<4} / {r['rating']:<12} [{r['item_id']}] {r['title'][:50]}"
        )
