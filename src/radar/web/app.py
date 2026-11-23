"""The FastAPI app: `create_app()` wires routes, middleware, static files and shared state.

`radar serve` calls `create_app()` and hands the result to uvicorn. Tests call it with a temp
store, fake models and an inline job runner.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from radar import tracing
from radar.config import Settings, get_settings
from radar.web import security
from radar.web.deps import WebState, make_templates, page
from radar.web.jobs import JobRunner

STATIC = Path(__file__).parent / "static"


def create_app(
    settings: Settings | None = None,
    ctx=None,
    job_ctx=None,
    runner: JobRunner | None = None,
    agent_factory=None,
) -> FastAPI:
    """Build the app. With no arguments it opens the real store (two connections, see deps.py)
    and starts a background job runner; tests pass their own contexts and an inline runner."""
    settings = settings or get_settings()
    if ctx is None or job_ctx is None:
        from radar.context import build_context

        # Build both without pointing the tool-call log anywhere, then decide: requests log
        # through the request connection; the job thread through its own (see runner below).
        job_ctx = job_ctx or build_context(settings, trace_log=False)
        ctx = ctx or build_context(settings, embeddings=job_ctx.embeddings, trace_log=False)
        tracing.configure(ctx.conn if settings.tool_log else None, settings.tool_log_keep_days)
    if runner is None:
        runner = JobRunner(thread_context=lambda: tracing.use_connection(job_ctx.conn))

    state = WebState(settings, ctx, job_ctx, runner, agent_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runner.start()
        yield
        runner.stop()

    app = FastAPI(title="AI Research Radar", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.radar = state
    app.state.templates = make_templates()
    app.add_middleware(security.SameOriginWrites)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=security.ALLOWED_HOSTS)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    from radar.web.routes import ask, digest, items, jobs, me, system

    for module in (digest, items, ask, jobs, me, system):
        app.include_router(module.router)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        # htmx doesn't swap error responses by default, so actions show a short message via
        # the response's own target (see base.html's htmx:responseError handler); pages get
        # a proper error page.
        if exc.status_code not in (404, 400, 409):
            return await http_exception_handler(request, exc)
        return page(request, "error.html", status_code=exc.status_code, message=exc.detail)

    return app
