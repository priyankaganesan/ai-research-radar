"""What's going on under the hood: health, sources, flagged items, the call log, the Critic."""

from fastapi import APIRouter, Depends, HTTPException, Request

from radar import services
from radar.web.deps import WebState, fragment, get_state, page

router = APIRouter(prefix="/system")


@router.get("/health")
def health(request: Request, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        h = services.health(ctx)
        stats = services.stats(ctx)
        daily = services.daily_running(ctx)
    try:
        from radar import schedule

        sched = schedule.status()
    except Exception as e:  # launchctl only exists on macOS
        sched = {"installed": False, "error": f"{type(e).__name__}: {e}"}
    return page(
        request,
        "system/health.html",
        h=h,
        stats=stats,
        daily=daily,
        sched=sched,
        settings=state.settings,
        jobs=state.runner.recent(10),
    )


@router.get("/sources")
def sources(request: Request, days: int = 14, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        h = services.source_health(ctx, days)
    by_date: dict[str, dict] = {}
    names: list[str] = []
    for r in h["runs"]:
        by_date.setdefault(r["date"], {})[r["source"]] = r
        if r["source"] not in names:
            names.append(r["source"])
    return page(
        request,
        "system/sources.html",
        by_date=by_date,
        names=sorted(names),
        quiet=h["quiet"],
        days=days,
    )


@router.get("/flagged")
def flagged(request: Request, all: bool = False, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        rows = services.flagged(ctx, include_released=all)
    return page(request, "system/flagged.html", rows=rows, show_all=all)


@router.post("/flagged/{item_id}/release")
def release(item_id: str, request: Request, state: WebState = Depends(get_state)):
    """You read the held item and it's fine: let it into future digests."""
    with state.write() as ctx:
        ok = services.release_flagged(ctx, item_id)
    text = "Released: it can appear in future digests." if ok else "It was already released."
    return fragment(request, "partials/_message.html", kind="ok", text=text)


@router.get("/calls")
def calls(
    request: Request,
    name: str = "",
    kind: str = "",
    errors: bool = False,
    roots: bool = True,
    limit: int = 50,
    state: WebState = Depends(get_state),
):
    with state.read() as ctx:
        rows = services.calls(ctx, limit, name or None, kind or None, errors, roots)
        stats = services.call_stats(ctx, days=7)
    return page(
        request,
        "system/calls.html",
        rows=rows,
        stats=stats,
        f={"name": name, "kind": kind, "errors": errors, "roots": roots},
    )


@router.get("/calls/{call_id}")
def call_tree(call_id: str, request: Request, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        rows = services.call_tree(ctx, call_id)
    if not rows:
        raise HTTPException(404, f"No call {call_id!r} in the local log.")
    longest = max(r["duration_ms"] for r in rows) or 1
    return page(
        request,
        "system/call_tree.html",
        rows=rows,
        root=rows[0],
        longest=longest,
        langsmith_project=state.settings.langsmith_project,
    )


@router.get("/critic")
def critic(request: Request, state: WebState = Depends(get_state)):
    from radar.store import memory

    with state.read() as ctx:
        out = services.critic_agreement(ctx)
        unrated = services.critic_unrated(ctx)
        ratings = memory.latest_ratings(ctx.conn)
    return page(request, "system/critic.html", out=out, unrated=unrated, ratings=ratings)
