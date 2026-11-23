"""Asking questions: the form, the job that runs the Q&A graph, and past conversations.

A question becomes a background job (see jobs.py): the POST returns at once with a progress
panel, and the graph's steps stream into it while Qwen works. Follow-ups reuse the same
`thread_id`, which is how the LangGraph checkpointer finds the earlier messages.
"""

import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, Request

from radar import services
from radar.web.deps import WebState, fragment, get_state, page
from radar.web.jobs import JobHandle
from radar.web.render import describe_step

router = APIRouter()

QUIET_NODES = {"prepare", "rewrite_query"}  # say nothing when these add no trace entry


@router.get("/ask")
def ask_page(
    request: Request,
    about: str = "",
    thread: str = "",
    q: str = "",
    state: WebState = Depends(get_state),
):
    prefill, turns = q, []
    with state.read() as ctx:
        if about and (item := services.get_item(ctx, about)):
            prefill = f"About {item.title} [{item.id}]: "
        if thread:
            turns = services.conversation(ctx, thread)
        recent = services.conversations(ctx, limit=5)
    return page(
        request,
        "ask.html",
        prefill=prefill,
        thread_id=thread,
        turns=turns,
        recent=recent,
        verify_default=state.settings.verify_answers,
        running=[j for j in state.runner.recent(10, kind="ask") if not j.done],
    )


@router.post("/ask")
def ask(
    request: Request,
    question: str = Form(...),
    thread_id: str = Form(""),
    verify: bool = Form(False),
    state: WebState = Depends(get_state),
):
    question = question.strip()
    if not question:
        return fragment(request, "partials/_message.html", kind="error", text="Type a question.")
    thread_id = thread_id or uuid.uuid4().hex[:8]

    def job(h: JobHandle):
        def on_step(node: str, entries: list[dict]) -> None:
            if not entries and node in QUIET_NODES:
                return
            for entry in entries or [None]:
                h.step(*describe_step(node, entry))

        h.step("Started: Qwen is deciding where to look first")
        agent = state.agent(verify)
        result = agent.ask(
            question, thread_id=thread_id, on_step=on_step, should_stop=h.should_stop
        )
        # The top-level traced call for this question, so the answer can link to its call tree.
        root = services.calls(state.job_ctx, limit=1, name="services.Agent.ask", roots_only=True)
        h.job.meta["call_id"] = root[0]["id"] if root else None
        return result

    title = f"Question: {question[:60]}{'…' if len(question) > 60 else ''}"
    j = state.runner.submit("ask", title, job, question=question, thread_id=thread_id)
    return fragment(request, "partials/_turn.html", job=j, runner=state.runner, oob=True)


@router.get("/ask/history")
def history(request: Request, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        convs = services.conversations(ctx, limit=100)
    return page(request, "history.html", convs=convs)


@router.get("/ask/t/{thread_id}")
def conversation(thread_id: str, request: Request, state: WebState = Depends(get_state)):
    with state.read() as ctx:
        turns = services.conversation(ctx, thread_id)
    if not turns:
        raise HTTPException(404, f"No conversation {thread_id!r}.")
    return page(
        request,
        "ask.html",
        prefill="",
        thread_id=thread_id,
        turns=turns,
        recent=[],
        verify_default=state.settings.verify_answers,
        running=[
            j
            for j in state.runner.recent(10, kind="ask")
            if not j.done and j.meta.get("thread_id") == thread_id
        ],
    )
