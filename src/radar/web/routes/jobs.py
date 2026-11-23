"""Following a background job: its page, its live event stream, and cancelling it.

Server-Sent Events (SSE) in one paragraph: the browser opens one long GET request with
`Accept: text/event-stream`, and the server keeps it open, writing small text messages:

    id: 3
    event: step
    data: <li>12 s · Searching the local store: “test-time compute”</li>

followed by a blank line. The browser's EventSource API reads them as they arrive, and htmx's
SSE extension swaps each one into the element whose `sse-swap` names that event. If the
connection drops, the browser reconnects by itself and sends the last `id` it saw
(`Last-Event-ID`), so the stream resumes where it left off instead of repeating steps.

This endpoint is `async`: it spends its time waiting, and an async function can wait without
holding one of the thread pool's threads. It only reads the job (in memory), never the database.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from radar.web.deps import WebState, fragment, get_state, page

router = APIRouter()

TICK_S = 1.0  # how often the stream checks for new steps and refreshes the elapsed time


def _job_or_404(state: WebState, job_id: str):
    job = state.runner.get(job_id)
    if job is None:
        raise HTTPException(404, "That job isn't known (the server may have restarted).")
    return job


@router.get("/jobs/{job_id}")
def job_page(job_id: str, request: Request, state: WebState = Depends(get_state)):
    job = _job_or_404(state, job_id)
    return page(request, "job.html", job=job, runner=state.runner)


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, request: Request, state: WebState = Depends(get_state)):
    job = state.runner.cancel(job_id) or _job_or_404(state, job_id)
    text = "Cancelled." if job.done else "Stopping after the current step…"
    return fragment(request, "partials/_message.html", kind="muted", text=text)


def sse(event: str, html: str, event_id: int | None = None) -> str:
    """One SSE message. `data:` can't hold a newline, so a multi-line fragment becomes several
    `data:` lines, which the browser joins back together."""
    lines = [f"event: {event}"]
    if event_id is not None:
        lines.insert(0, f"id: {event_id}")
    lines += [f"data: {line}" for line in html.splitlines() or [""]]
    return "\n".join(lines) + "\n\n"


@router.get("/jobs/{job_id}/events")
async def events(
    job_id: str, request: Request, after: int = 0, state: WebState = Depends(get_state)
):
    job = _job_or_404(state, job_id)
    env = request.app.state.templates.env
    step_tpl = env.get_template("partials/_step.html")
    status_tpl = env.get_template("partials/_job_status.html")
    job_tpl = env.get_template("partials/_job.html")
    last_seen = request.headers.get("Last-Event-ID")
    start = int(last_seen) + 1 if last_seen and last_seen.isdigit() else after

    async def stream():
        sent = start
        while True:
            events = job.events[sent:]  # a copy: the job thread may append meanwhile
            for e in events:
                yield sse("step", step_tpl.render(e=e), e["n"])
            sent += len(events)
            if job.done:
                yield sse("done", job_tpl.render(job=job, runner=state.runner))
                yield sse("close", "")
                return
            yield sse("status", status_tpl.render(job=job, runner=state.runner))
            if await request.is_disconnected():
                return
            await asyncio.sleep(TICK_S)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
