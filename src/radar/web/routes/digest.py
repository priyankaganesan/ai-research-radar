"""Today's digest, past digests, and starting a digest run from the browser."""

from fastapi import APIRouter, Depends, Form, HTTPException, Request

from radar import services
from radar.web.deps import WebState, fragment, get_state, page
from radar.web.jobs import JobHandle

router = APIRouter()

DAILY_STEPS = {
    "collect": "Collected new items from the sources",
    "learn": "Applied your ratings to the profile",
    "candidates": "Picked candidates (new, not shown before, safety-screened)",
    "score": "Scored the candidates",
    "pick": "Chose the picks for each slot",
    "summarize": "Summarized the picks with Qwen",
    "render": "Wrote the digest",
    "recheck_watchlist": "Re-checked the early-buzz watchlist",
}


@router.get("/")
def today(request: Request, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        d = services.digest_view(ctx)
    return page(request, "digest.html", d=d)


@router.get("/digest/{on}")
def past(on: str, request: Request, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        d = services.digest_view(ctx, on)
    if d is None:
        raise HTTPException(404, f"No digest for {on}.")
    return page(request, "digest.html", d=d)


@router.post("/digest/run")
def run(
    request: Request,
    mode: str = Form(""),
    collect: bool = Form(False),
    state: WebState = Depends(get_state),
):
    """Queue a digest run. Refused while one is already going (here or the launchd job)."""
    with state.read() as ctx:
        other = services.daily_running(ctx)
    if other or any(j.kind == "daily" and not j.done for j in state.runner.recent(20)):
        return fragment(
            request,
            "partials/_message.html",
            kind="error",
            text="A digest run is already going. Try again when it finishes.",
        )

    def job(h: JobHandle):
        def on_step(node: str, _trace: list[dict]) -> None:
            h.step(DAILY_STEPS.get(node, f"Finished {node}"))

        h.step("Started" + (" (collecting first, this takes a while)" if collect else ""))
        return services.daily(state.job_ctx, mode=mode or None, collect=collect, on_step=on_step)

    j = state.runner.submit("daily", "Digest run", job)
    return fragment(request, "partials/_job.html", job=j, runner=state.runner)
