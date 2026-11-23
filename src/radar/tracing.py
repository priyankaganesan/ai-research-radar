"""Tracing: every important call is recorded twice, in two complementary places.

1. **LangSmith** (cloud). `@traced` wraps the function with LangSmith's `@traceable`, so the call
   becomes a *run* in a trace tree, nested under whatever run called it (a graph node, another
   traced function, an MCP request). It only sends anything when LANGSMITH_TRACING=true and an
   API key is set; otherwise `@traceable` is a cheap no-op.
2. **The local tool-call log** (the `tool_calls` table in radar.db). One row per call: name,
   arguments, a short result summary, duration, error and timestamp. It works with LangSmith off
   and offline, and `radar calls` reads it. When LangSmith is on, each row also stores the
   LangSmith trace and run IDs, so a row leads to its trace in the web UI.

LangChain and LangGraph already trace LLM calls, graph nodes and `@tool` calls to LangSmith on
their own. So the agent tools (store_search, ...) use `traced(..., langsmith=False)`: LangChain
traces them, and the decorator only adds the local row. Adding `@traceable` there too would put
each tool in the trace twice.

The decorated functions take things that must never be sent anywhere (`ctx` holds the database
connection and the settings, including the Semantic Scholar API key), so `_clean_inputs` drops
them before LangSmith or the log sees the arguments.
"""

import contextlib
import contextvars
import dataclasses
import functools
import inspect
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

log = logging.getLogger("radar.tracing")

RunType = Literal["tool", "chain", "retriever", "llm", "embedding", "parser", "prompt"]

# Arguments that are plumbing, not inputs: never logged or sent to LangSmith.
HIDDEN_ARGS = {"self", "ctx", "client", "s2_client", "llm", "judge", "checkpointer", "search"}
MAX_ARGS_CHARS = 2000
MAX_SUMMARY_CHARS = 300


# ---- where local rows go ----------------------------------------------------------------------


class _Sink:
    """The connection the log writes to, shared by every thread of the process.

    It's the store's own connection (so tests and `:memory:` stores work), guarded by a lock
    because LangGraph can run tool calls on worker threads.
    """

    def __init__(self) -> None:
        self.conn: sqlite3.Connection | None = None
        self.lock = threading.Lock()


_sink = _Sink()

# A connection that overrides the shared one for the current context (see `use_connection`),
# with its own lock, and a flag that turns logging off for reads (see `quiet`).
_override: contextvars.ContextVar[tuple[sqlite3.Connection, threading.Lock] | None] = (
    contextvars.ContextVar("radar_log_conn", default=None)
)
_quiet: contextvars.ContextVar[bool] = contextvars.ContextVar("radar_log_quiet", default=False)

# The local row ID of the traced call we're inside, so nested calls record their parent.
# A ContextVar (not a global) keeps concurrent calls on different threads apart.
_current: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "radar_call", default=None
)


def configure(conn: sqlite3.Connection | None, keep_days: int | None = 30) -> None:
    """Send local rows to `conn` (None turns the local log off). Rows older than `keep_days`
    are deleted now, so the table doesn't grow forever. Called by `build_context`."""
    with _sink.lock:
        _sink.conn = conn
        if conn is not None and keep_days:
            cutoff = (datetime.now(UTC) - timedelta(days=keep_days)).isoformat(timespec="seconds")
            try:
                with conn:
                    conn.execute("DELETE FROM tool_calls WHERE started_at < ?", (cutoff,))
            except sqlite3.Error as e:  # e.g. the table isn't there yet; logging must not break
                log.warning("tool-call log: prune failed: %s", e)


def enabled() -> bool:
    return _sink.conn is not None


@contextlib.contextmanager
def use_connection(conn: sqlite3.Connection):
    """Write this context's log rows through `conn` instead of the shared connection.

    The web app needs this: its background job has its own database connection, and a log row
    written there joins the job's open transaction, exactly as the CLI's single connection
    does. LangGraph copies the context into its worker threads, so the override follows the
    job's tool calls too. Does nothing while the local log is off."""
    token = _override.set((conn, threading.Lock()))
    try:
        yield
    finally:
        _override.reset(token)


@contextlib.contextmanager
def quiet():
    """Don't record traced calls made inside this block, locally or in LangSmith.

    The web app wraps page views in it: opening the Profile page calls the traced
    `services.profile`, and a trace for every page view would bury the calls that matter
    (asks, ratings, fetches). Actions are still traced."""
    from langsmith import tracing_context

    token = _quiet.set(True)
    try:
        with tracing_context(enabled=False):
            yield
    finally:
        _quiet.reset(token)


# ---- turning arguments and results into short text ------------------------------------------


def _clean_inputs(inputs: dict) -> dict:
    return {k: v for k, v in inputs.items() if k not in HIDDEN_ARGS}


def _to_json(value: Any, limit: int) -> str:
    text = json.dumps(value, default=_jsonable, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, set | frozenset):
        return sorted(value, key=str)
    return repr(value)


def summarize(result: Any) -> str:
    """A one-line description of a result: counts for collections, the start of anything else."""
    if result is None:
        return "None"
    if isinstance(result, list | tuple) and not (
        isinstance(result, tuple) and len(result) <= 4 and all(_scalar(x) for x in result)
    ):
        first = ""
        if result and isinstance(result[0], dict):
            first = str(result[0].get("item_id") or result[0].get("id") or "")
        elif result and hasattr(result[0], "item_id"):
            first = result[0].item_id
        elif result and hasattr(result[0], "id"):
            first = str(result[0].id)
        return f"{len(result)} result(s)" + (f", first {first}" if first else "")
    if isinstance(result, str):
        text = result.replace("\n", " ")
    elif dataclasses.is_dataclass(result) and not isinstance(result, type):
        fields = dataclasses.asdict(result)
        short = {k: _short(v) for k, v in fields.items()}
        text = f"{type(result).__name__}({_to_json(short, MAX_SUMMARY_CHARS)})"
    else:
        text = _to_json(result, MAX_SUMMARY_CHARS)
    return text if len(text) <= MAX_SUMMARY_CHARS else text[: MAX_SUMMARY_CHARS - 1] + "…"


def _short(v: Any) -> Any:
    if isinstance(v, list | dict):
        return f"{len(v)} item(s)"
    if isinstance(v, str) and len(v) > 80:
        return v[:79].replace("\n", " ") + "…"
    return v


def _scalar(x: Any) -> bool:
    return isinstance(x, str | int | float | bool) or x is None


# ---- the decorator ------------------------------------------------------------------------------


def traced(
    name: str,
    run_type: RunType = "chain",
    *,
    kind: str = "service",
    langsmith: bool = True,
    summary: Callable[[Any], str] = summarize,
    process_outputs: Callable[[Any], dict] | None = None,
):
    """Trace a function to LangSmith (unless langsmith=False) and the local tool-call log.

    name:      the run name in LangSmith and in `radar calls`, e.g. "services.ingest_hf"
    run_type:  how LangSmith shows the run: "tool" (calls the outside world), "retriever"
               (returns passages), "chain" (orchestrates other steps)
    kind:      a tag for grouping: service, source, mcp or agent_tool
    """

    def decorate(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def logged(*args, **kwargs):
                if _sink.conn is None or _quiet.get():
                    return await fn(*args, **kwargs)
                call = _Call(name, kind, summary, fn, args, kwargs)
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as e:
                    call.finish(error=e)
                    raise
                call.finish(result=result)
                return result

        else:

            @functools.wraps(fn)
            def logged(*args, **kwargs):
                if _sink.conn is None or _quiet.get():
                    return fn(*args, **kwargs)
                call = _Call(name, kind, summary, fn, args, kwargs)
                try:
                    result = fn(*args, **kwargs)
                except BaseException as e:
                    call.finish(error=e)
                    raise
                call.finish(result=result)
                return result

        if not langsmith:
            return logged
        wrapped = traceable(
            run_type=run_type,
            name=name,
            tags=["radar", kind],
            process_inputs=_clean_inputs,
            process_outputs=process_outputs,
        )(logged)
        # @traceable advertises an extra `config` parameter; MCP and LangChain build their
        # argument schemas from the signature, so show them the original one.
        wrapped.__signature__ = inspect.signature(fn)
        return wrapped

    return decorate


class _Call:
    """One call in progress: remembers when it started and where it sits in the call tree."""

    def __init__(self, name, kind, summary, fn, args, kwargs):
        self.name, self.kind, self.summary = name, kind, summary
        self.args = _bind_args(fn, args, kwargs)
        self.id = uuid.uuid4().hex[:12]
        parent = _current.get()
        self.parent_id = parent[0] if parent else None
        self.root_id = parent[1] if parent else self.id
        self.token = _current.set((self.id, self.root_id))
        self.started = datetime.now(UTC)
        self.t0 = time.perf_counter()

    def finish(self, result: Any = None, error: BaseException | None = None) -> None:
        ms = (time.perf_counter() - self.t0) * 1000
        try:
            _current.reset(self.token)
        except ValueError:  # finished in a different context than it started (rare, async)
            pass
        _record(
            call_id=self.id,
            parent_id=self.parent_id,
            root_id=self.root_id,
            name=self.name,
            kind=self.kind,
            args=self.args,
            summary=None if error else _safe(self.summary, result),
            error=f"{type(error).__name__}: {error}" if error else None,
            started=self.started,
            ms=ms,
        )


def _bind_args(fn: Callable, args: tuple, kwargs: dict) -> dict:
    """The call's arguments by parameter name, with **kwargs spread out."""
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
    except (TypeError, ValueError):
        return _clean_inputs({"args": list(args), **kwargs})
    out = {}
    for key, value in bound.arguments.items():
        if sig.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
            out.update(value)
        else:
            out[key] = value
    return _clean_inputs(out)


def _safe(summary: Callable[[Any], str], result: Any) -> str:
    try:
        return summary(result)
    except Exception as e:  # a bad summary must never break the call it describes
        return f"(summary failed: {e})"


def _record(*, call_id, parent_id, root_id, name, kind, args, summary, error, started, ms):
    run = get_current_run_tree()  # set only when LangSmith is tracing this call
    row = (
        call_id,
        parent_id,
        root_id,
        name,
        kind,
        _safe(lambda a: _to_json(a, MAX_ARGS_CHARS), args),
        summary,
        error,
        started.isoformat(timespec="microseconds"),
        round(ms, 1),
        str(run.trace_id) if run else None,
        str(run.id) if run else None,
    )
    override = _override.get()
    if _sink.conn is None:
        return
    conn, lock = override if override else (_sink.conn, _sink.lock)
    with lock:
        try:
            # Join the caller's open transaction if there is one; otherwise commit right away.
            was_open = conn.in_transaction
            conn.execute(
                "INSERT INTO tool_calls (id, parent_id, root_id, name, kind, args, summary,"
                " error, started_at, duration_ms, langsmith_trace_id, langsmith_run_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            if not was_open:
                conn.commit()
        except sqlite3.Error as e:
            log.warning("tool-call log: could not record %s: %s", name, e)


def passages_as_documents(passages: Any) -> dict:
    """LangSmith shows a retriever run's results as a list of documents when they look like
    LangChain Documents ({"page_content", "metadata"}), so reshape our Passages that way."""
    docs = []
    for p in passages or []:
        d = dataclasses.asdict(p) if dataclasses.is_dataclass(p) else dict(p)
        text = d.pop("text", "")
        docs.append({"type": "Document", "page_content": text, "metadata": d})
    return {"documents": docs}
