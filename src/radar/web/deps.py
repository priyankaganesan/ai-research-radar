"""Everything the routes share: the store, the job runner, the Q&A agents and the templates.

Two database connections, on purpose
------------------------------------
The CLI opens one connection per process and is single-threaded, so it never thinks about this.
The web server has several threads: FastAPI runs each page request in a thread pool, and the job
runner has its own thread. SQLite transactions on one shared connection can interleave between
threads (one thread's `with conn:` could commit another thread's half-done work), so:

- **Requests** use `WebState.ctx` and take `WebState.lock` while they touch it (`state.read()`
  and `state.write()`). Page requests are quick (milliseconds), and there's one user, so
  taking turns costs nothing noticeable.
- **Jobs** use `WebState.job_ctx`, a second context with its own connection, used only by the
  worker thread. Its tool-call log rows go through that connection too (tracing.use_connection).

WAL mode lets the two connections (and the 7 AM launchd run, a third process) read while one of
them writes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fastapi import Request
from fastapi.templating import Jinja2Templates

from radar import tracing
from radar.config import Settings
from radar.context import AppContext
from radar.web.jobs import JobRunner

TEMPLATES = Path(__file__).parent / "templates"


class WebState:
    def __init__(
        self,
        settings: Settings,
        ctx: AppContext,
        job_ctx: AppContext,
        runner: JobRunner,
        agent_factory=None,
    ):
        import threading

        self.settings = settings
        self.ctx = ctx
        self.job_ctx = job_ctx
        self.runner = runner
        self.lock = threading.Lock()
        self._agent_factory = agent_factory
        self._agents: dict[bool, object] = {}

    @contextmanager
    def read(self) -> Iterator[AppContext]:
        """For page views: take turns on the request connection, and don't trace (see
        tracing.quiet: a trace per page view would bury the calls that matter)."""
        with self.lock, tracing.quiet():
            yield self.ctx

    @contextmanager
    def write(self) -> Iterator[AppContext]:
        """For actions (rate, approve, release): take turns, and trace as usual."""
        with self.lock:
            yield self.ctx

    def agent(self, verify: bool):
        """The Q&A agent, built once per verify setting (building compiles the graph and opens
        the checkpointer). Only the job thread calls this, so it uses the job context."""
        if verify not in self._agents:
            if self._agent_factory is not None:
                self._agents[verify] = self._agent_factory(self.job_ctx, verify)
            else:
                from radar.services import Agent

                self._agents[verify] = Agent(self.job_ctx, verify=verify)
        return self._agents[verify]


def get_state(request: Request) -> WebState:
    return request.app.state.radar


def is_htmx(request: Request) -> bool:
    """htmx marks its requests with this header. A normal visit gets the whole page; an htmx
    request gets just the fragment that changes ("one route, two shapes").

    Two exceptions want the whole page anyway: boosted links (`hx-boost` swaps the whole
    <body>) and history restores (the back button, when htmx's cache is empty)."""
    h = request.headers
    return (
        h.get("HX-Request") == "true"
        and h.get("HX-Boosted") != "true"
        and h.get("HX-History-Restore-Request") != "true"
    )


def nav_info(state: "WebState") -> dict:
    """Badges for the nav bar: things waiting for you, and whether Qwen is busy."""
    from radar import services
    from radar.store import safety as safety_store

    with state.read() as ctx:
        pending = len(safety_store.pending(ctx.conn))
        flagged = len(safety_store.flagged(ctx.conn))
        daily = services.daily_running(ctx)
    return {
        "pending": pending,
        "flagged": flagged,
        "daily_running": daily,
        "busy_job": state.runner.busy(),
    }


def page(request: Request, template: str, status_code: int = 200, **context):
    """Render a template that `{% extends layout %}`: the full page for a normal visit, just its
    content block for an htmx request."""
    state = get_state(request)
    htmx = is_htmx(request)
    context.setdefault("nav", None if htmx else nav_info(state))
    return request.app.state.templates.TemplateResponse(
        request,
        template,
        {"layout": "bare.html" if htmx else "base.html", "htmx": htmx, **context},
        status_code=status_code,
    )


def fragment(request: Request, template: str, status_code: int = 200, **context):
    """Render a partial template on its own (the answer to an htmx action)."""
    return request.app.state.templates.TemplateResponse(
        request, template, context, status_code=status_code
    )


def make_templates() -> Jinja2Templates:
    from radar.web import render

    # Jinja2Templates turns autoescaping on for .html files: every {{ value }} is HTML-escaped,
    # so a paper title like "<script>..." shows as text instead of running.
    templates = Jinja2Templates(directory=str(TEMPLATES))
    render.install(templates.env)
    return templates
