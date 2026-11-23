"""Your side of the agent: search, the interest profile, the watchlist, gaps and the library."""

from fastapi import APIRouter, Depends, Form, HTTPException, Request

from radar import services
from radar.web.deps import WebState, fragment, get_state, page
from radar.web.jobs import JobHandle
from radar.web.routes.items import submit_full_text

router = APIRouter()


# ---- search ---------------------------------------------------------------------------------


@router.get("/search")
def search(
    request: Request,
    q: str = "",
    kind: str = "",
    since: str = "",
    state: WebState = Depends(get_state),
):
    """The same hybrid search the agent uses (vectors + keywords, fused). The box asks again
    400 ms after you stop typing (hx-trigger="input changed delay:400ms"), not on every key."""
    passages, error = [], ""
    if q.strip():
        try:
            with state.read() as ctx:
                passages = services.search(
                    ctx, q.strip(), k=10, kind=kind or None, since=since or None
                )
        except Exception as e:  # the query is embedded by Ollama; it may not be running
            error = f"Search failed: {e}"
    if request.headers.get("HX-Target") == "results":
        return fragment(request, "partials/_passages.html", passages=passages, q=q, error=error)
    return page(request, "search.html", passages=passages, q=q, kind=kind, since=since, error=error)


# ---- profile --------------------------------------------------------------------------------


def _profile(state: WebState) -> dict:
    with state.read() as ctx:
        return services.profile(ctx)


@router.get("/profile")
def profile(request: Request, state: WebState = Depends(get_state)):
    return page(request, "profile.html", p=_profile(state), running=_profile_job(state))


def _profile_job(state: WebState):
    return next((j for j in state.runner.recent(10, kind="profile") if not j.done), None)


@router.post("/profile/pending/{pending_id}/{decision}")
def decide(pending_id: int, decision: str, request: Request, state: WebState = Depends(get_state)):
    """Approve or reject a topic removal the Profile agent proposed. A person's decision: it's a
    button here and a flag in the CLI, and deliberately not an MCP tool."""
    if decision not in ("approve", "reject"):
        raise HTTPException(404, "Unknown decision.")
    try:
        with state.write() as ctx:
            row = services.decide_profile_change(ctx, pending_id, approve=decision == "approve")
    except ValueError as e:
        return fragment(request, "partials/_message.html", kind="error", text=str(e))
    p = _profile(state)
    return fragment(
        request,
        "partials/_pending.html",
        p=p,
        decided=row,
        count=len(p["awaiting_approval"]),
        oob=True,  # also refresh the nav badge, wherever it is on the page
    )


@router.post("/profile/update")
def update(request: Request, agent: bool = Form(False), state: WebState = Depends(get_state)):
    """Apply waiting ratings now. Without the agent it's arithmetic (instant); with it, the
    Profile agent asks Qwen whether the topics should change, so it runs as a job."""
    if not agent:
        with state.write() as ctx:
            report = services.update_profile(ctx, agent=False)
        return page(request, "profile.html", p=_profile(state), report=report.lines())

    def job(h: JobHandle):
        h.step("Applying ratings to the weights and reserved slots")
        report = services.update_profile(state.job_ctx, agent=True)
        return report.lines()

    j = state.runner.submit("profile", "Profile update", job)
    return fragment(request, "partials/_job.html", job=j, runner=state.runner)


@router.post("/profile/reset")
def reset(request: Request, state: WebState = Depends(get_state)):
    with state.write() as ctx:
        services.reset_profile(ctx)
    return page(
        request, "profile.html", p=_profile(state), report=["Profile reset to the defaults."]
    )


# ---- watchlist, gaps, library ------------------------------------------------------------------

WATCH_STATUSES = ("watching", "established", "dropped")


@router.get("/watchlist")
def watchlist(request: Request, status: str = "", state: WebState = Depends(get_state)):
    if status and status not in WATCH_STATUSES:
        raise HTTPException(404, f"Unknown status {status!r}.")
    with state.read() as ctx:
        rows = services.watchlist(ctx, status or None)
    return page(request, "watchlist.html", rows=rows, status=status, statuses=WATCH_STATUSES)


@router.get("/gaps")
def gaps(request: Request, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        rows = services.gaps(ctx)
    return page(request, "gaps.html", rows=rows)


@router.get("/library")
def library(request: Request, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        rows = services.full_text_list(ctx, limit=200)
    jobs = [j for j in state.runner.recent(20, kind="full_text")]
    return page(request, "library.html", rows=rows, jobs=jobs)


@router.post("/library/fetch")
def library_fetch(request: Request, item_id: str = Form(...), state: WebState = Depends(get_state)):
    item_id = item_id.strip()
    with state.read() as ctx:
        if not item_id or services.get_item(ctx, item_id) is None:
            return fragment(
                request,
                "partials/_message.html",
                kind="error",
                text=f"No item {item_id!r} in the store. Search for it first.",
            )
    j = submit_full_text(state, item_id)
    return fragment(request, "partials/_job.html", job=j, runner=state.runner)
