"""Background jobs: work that takes too long to finish inside one web request.

An `ask` can take ten minutes on a laptop running Qwen locally. Instead of making the browser
wait on one request, the route *submits a job* and returns at once; the page then follows the
job's progress (Server-Sent Events, see routes/jobs.py) and shows the result when it's done.

Design choices, each for a reason:
- **One worker thread, one queue.** Ollama on a 16 GB Mac runs one Qwen generation at a time
  anyway; a queue makes that visible ("1 job ahead of you") instead of two jobs crawling.
- **A thread, not asyncio.** The services are ordinary blocking Python (sqlite3, LangChain's
  sync `invoke`), so they run as they are, in a thread.
- **In memory.** Jobs live in a dict while the server runs. The results worth keeping
  (questions, answers, digests, full texts) are saved by the services themselves, so a restart
  only loses progress of jobs still running.
- **Cooperative cancel.** Python can't safely kill a thread in the middle of an Ollama call. Cancel
  sets a flag; the job checks it between steps and stops there.

This module knows nothing about HTML: a job is a function, events are plain dicts. The routes
decide how to show them.
"""

import itertools
import logging
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("radar.web.jobs")

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
FINISHED = (DONE, FAILED, CANCELLED)


class JobCancelled(Exception):
    """Raised inside a job when it notices it was asked to stop."""


@dataclass
class Job:
    id: str
    kind: str  # "ask" | "full_text" | "daily" | "profile"
    title: str
    status: str = QUEUED
    events: list[dict] = field(default_factory=list)  # {"n", "t", "text", "detail"}
    result: Any = None
    error: str = ""
    meta: dict = field(default_factory=dict)  # anything the routes want to keep (thread_id...)
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    cancel_requested: bool = False

    @property
    def done(self) -> bool:
        return self.status in FINISHED

    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started


class JobHandle:
    """What a running job gets: a way to report progress and to notice a cancel."""

    def __init__(self, job: Job, lock: threading.Lock):
        self.job = job
        self._lock = lock

    def step(self, text: str, detail: str = "") -> None:
        with self._lock:
            self.job.events.append(
                {"n": len(self.job.events), "t": self.job.elapsed(), "text": text, "detail": detail}
            )

    def should_stop(self) -> bool:
        return self.job.cancel_requested

    def check(self) -> None:
        if self.job.cancel_requested:
            raise JobCancelled()


JobFn = Callable[[JobHandle], Any]


class JobRunner:
    """Runs submitted jobs one at a time on a background thread.

    inline=True runs each job right away in the caller's thread: tests use it so they can
    check results without waiting or sleeping.
    """

    def __init__(
        self,
        inline: bool = False,
        keep: int = 50,
        thread_context: Callable[[], AbstractContextManager] | None = None,
    ):
        self.inline = inline
        self.keep = keep
        self._jobs: dict[str, Job] = {}
        self._fns: dict[str, JobFn] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._order = itertools.count()
        self._thread: threading.Thread | None = None
        # Entered once by the worker thread, around all its jobs: the web app uses it to send the
        # worker's tool-call log rows through the worker's own database connection.
        self._thread_context = thread_context or nullcontext

    # ---- lifecycle --------------------------------------------------------------------------

    def start(self) -> None:
        if self.inline or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._work, name="radar-jobs", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=2)
            self._thread = None

    # ---- the API the routes use ---------------------------------------------------------------

    def submit(self, kind: str, title: str, fn: JobFn, **meta) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, title=title, meta=meta)
        with self._lock:
            self._jobs[job.id] = job
            self._fns[job.id] = fn
            self._prune()
        if self.inline:
            self._run(job.id)
        else:
            self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def recent(self, limit: int = 10, kind: str | None = None) -> list[Job]:
        jobs = [j for j in self._jobs.values() if kind is None or j.kind == kind]
        return sorted(jobs, key=lambda j: j.created, reverse=True)[:limit]

    def ahead_of(self, job: Job) -> int:
        """How many jobs will run before this queued one (including one running now)."""
        if job.status != QUEUED:
            return 0
        return sum(
            1
            for j in self._jobs.values()
            if j.created < job.created and j.status in (QUEUED, RUNNING)
        )

    def busy(self) -> Job | None:
        return next((j for j in self._jobs.values() if j.status == RUNNING), None)

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None or job.done:
            return job
        job.cancel_requested = True
        if job.status == QUEUED:  # never started: just mark it
            self._finish(job, CANCELLED)
        return job

    # ---- the worker ---------------------------------------------------------------------------

    def _work(self) -> None:
        with self._thread_context():
            while (job_id := self._queue.get()) is not None:
                self._run(job_id)

    def _run(self, job_id: str) -> None:
        job = self._jobs[job_id]
        fn = self._fns.pop(job_id, None)
        if job.done or fn is None:  # cancelled while queued
            return
        job.status, job.started = RUNNING, time.time()
        handle = JobHandle(job, self._lock)
        try:
            job.result = fn(handle)
        except JobCancelled:
            self._finish(job, CANCELLED)
            return
        except Exception as e:
            if type(e).__name__ == "Cancelled":  # services.Cancelled, raised between graph steps
                self._finish(job, CANCELLED)
                return
            job.error = f"{type(e).__name__}: {e}"
            job.meta["traceback"] = traceback.format_exc(limit=5)
            log.exception("job %s (%s) failed", job.id, job.title)
            self._finish(job, FAILED)
            return
        self._finish(job, DONE)

    def _finish(self, job: Job, status: str) -> None:
        job.status = status
        job.finished = time.time()

    def _prune(self) -> None:
        """Forget the oldest finished jobs beyond `keep`, so memory doesn't grow forever."""
        finished = sorted((j for j in self._jobs.values() if j.done), key=lambda j: j.created)
        for j in finished[: max(0, len(self._jobs) - self.keep)]:
            del self._jobs[j.id]
