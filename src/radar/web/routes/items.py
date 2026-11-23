"""One item: its details, rating it, and its full text (fetch, then search inside it)."""

from dataclasses import asdict

from fastapi import APIRouter, Depends, Form, HTTPException, Request

from radar import services
from radar.web.deps import WebState, fragment, get_state, page
from radar.web.jobs import JobHandle

router = APIRouter()


@router.get("/items/{item_id}")
def item_page(item_id: str, request: Request, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        found = services.show(ctx, item_id)
        if found is None:
            raise HTTPException(404, f"No item {item_id!r} in the store.")
        item, signals = found
        info = {
            "rating": services.item_rating(ctx, item_id),
            "watch": services.watch_status(ctx, item_id),
            "full_text": services.full_text_record(ctx, item_id),
        }
    return page(request, "item.html", item=item, signals=signals, **info)


@router.post("/items/{item_id}/rate")
def rate(
    item_id: str, request: Request, value: str = Form(...), state: WebState = Depends(get_state)
):
    """Record a rating and send back the rating buttons with the new choice marked."""
    try:
        with state.write() as ctx:
            services.rate(ctx, item_id, value)
    except ValueError as e:
        return fragment(
            request, "partials/_rating.html", item_id=item_id, rating=None, error=str(e)
        )
    except Exception as e:  # e.g. "database is locked" while the morning run writes
        return fragment(
            request,
            "partials/_rating.html",
            item_id=item_id,
            rating=None,
            error=f"Not saved ({e}). Try again in a few seconds.",
        )
    return fragment(request, "partials/_rating.html", item_id=item_id, rating=value, saved=True)


def submit_full_text(state: WebState, item_id: str, force: bool = False):
    def job(h: JobHandle):
        h.step(f"Fetching {item_id} (arXiv HTML first, then the PDF)")
        r = services.full_text(state.job_ctx, item_id, reason="web", force=force)
        if r.status == "ok":
            h.step(f"Split into {r.sections} sections and {r.chunks} chunks, embedded and indexed")
        return asdict(r)

    return state.runner.submit("full_text", f"Full text of {item_id}", job, item_id=item_id)


@router.post("/items/{item_id}/fulltext")
def fetch_full_text(
    item_id: str, request: Request, force: bool = Form(False), state: WebState = Depends(get_state)
):
    with state.read() as ctx:
        if services.get_item(ctx, item_id) is None:
            raise HTTPException(404, f"No item {item_id!r} in the store.")
    j = submit_full_text(state, item_id, force)
    return fragment(request, "partials/_job.html", job=j, runner=state.runner)


@router.get("/items/{item_id}/fulltext/search")
def search_full_text(
    item_id: str, request: Request, q: str = "", state: WebState = Depends(get_state)
):
    passages, error = [], ""
    if q.strip():
        try:
            with state.read() as ctx:
                passages = services.search_full_text(ctx, item_id, q.strip())
        except Exception as e:  # the query is embedded by Ollama; it may not be running
            error = f"Search failed: {e}"
    return fragment(request, "partials/_passages.html", passages=passages, q=q, error=error)
