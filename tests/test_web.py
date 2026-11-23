"""The web UI, driven in-process with FastAPI's TestClient: no browser, no Ollama, no network.

Jobs run inline (JobRunner(inline=True)), so a POST that starts a job returns after it's done.
"""

import json
import os
import threading
import time

import pytest
from conftest import fake_chat, make_item, tool_call
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from radar import services, tracing
from radar.rag.index import index_items
from radar.store import digests as digest_store
from radar.store import safety as safety_store
from radar.store import tool_calls
from radar.web.app import create_app
from radar.web.jobs import CANCELLED, DONE, FAILED, QUEUED, JobRunner
from radar.web.render import md

HX = {"HX-Request": "true"}
SUFFICIENT = '{"sufficient": true, "missing": ""}'


def fake_agent_factory(*replies, judge=(SUFFICIENT,)):
    def factory(ctx, verify):
        return services.Agent(
            ctx,
            llm=fake_chat(*replies),
            judge=fake_chat(*judge),
            checkpointer=InMemorySaver(),
            verify=False,
        )

    return factory


def client_for(ctx, agent_factory=None, runner=None) -> TestClient:
    app = create_app(
        ctx.settings,
        ctx=ctx,
        job_ctx=ctx,
        runner=runner or JobRunner(inline=True),
        agent_factory=agent_factory,
    )
    return TestClient(app, base_url="http://127.0.0.1:8000")


@pytest.fixture
def seeded(filled_ctx):
    """A store with a digest for 2026-11-20: two picks, one Critic drop, a review flag."""
    ctx = filled_ctx
    with ctx.conn:
        digest_store.save_digest(
            ctx.conn,
            "2026-11-20",
            [
                {"item_id": "2401.10774", "rank": 1, "slot": "main", "score": 0.7, "reason": "Top"},
                {
                    "item_id": "2410.00002",
                    "rank": 2,
                    "slot": "early_buzz",
                    "score": 0.5,
                    "reason": "Upvoted",
                },
            ],
        )
        digest_store.save_summary(
            ctx.conn, "2401.10774", "Medusa adds heads [2401.10774].", "fake", "multi"
        )
        safety_store.save_review(ctx.conn, "2026-11-20", ["Critic hit its drop cap."])
        safety_store.save_critic_decisions(
            ctx.conn,
            "2026-11-20",
            [
                {"item": "2401.10774", "critic": "keep"},
                {"item": "2005.11401", "critic": "drop", "concerns": "too old"},
            ],
        )
    return ctx


# ---- pages --------------------------------------------------------------------------------


def test_every_page_renders(seeded):
    c = client_for(seeded)
    for url in [
        "/",
        "/digest/2026-11-20",
        "/ask",
        "/search",
        "/profile",
        "/watchlist",
        "/gaps",
        "/library",
        "/ask/history",
        "/items/2401.10774",
        "/system/health",
        "/system/sources",
        "/system/flagged",
        "/system/calls",
        "/system/critic",
    ]:
        r = c.get(url)
        assert r.status_code == 200, url
        assert "<nav>" in r.text, url  # a normal visit gets the whole page


def test_htmx_requests_get_only_the_fragment(seeded):
    r = client_for(seeded).get("/watchlist", headers=HX)
    assert r.status_code == 200
    assert "<nav>" not in r.text and "Early-buzz watchlist" in r.text


def test_digest_shows_picks_by_slot_review_and_critic_drops(seeded):
    r = client_for(seeded).get("/")
    assert "Digest for 2026-11-20" in r.text
    assert "Main picks" in r.text and "Early buzz" in r.text
    assert "Critic hit its drop cap." in r.text  # the safety review banner
    assert "too old" in r.text  # the dropped item, with its reason
    assert 'href="/items/2401.10774">[2401.10774]</a>' in r.text  # citation linked


def test_unknown_digest_and_item_are_404s(seeded):
    c = client_for(seeded)
    assert c.get("/digest/1999-01-01").status_code == 404
    assert c.get("/items/nope").status_code == 404


def test_no_digest_yet_offers_a_run(filled_ctx):
    r = client_for(filled_ctx).get("/")
    assert "No digest yet" in r.text and 'hx-post="/digest/run"' in r.text


# ---- small writes and the guardrails around them --------------------------------------------


def test_rating_saves_and_returns_the_buttons_with_the_choice(seeded):
    r = client_for(seeded).post("/items/2401.10774/rate", data={"value": "useful"}, headers=HX)
    assert r.status_code == 200
    assert 'class="chosen' in r.text and "Saved" in r.text
    assert services.item_rating(seeded, "2401.10774") == "useful"


def test_bad_rating_value_is_reported_not_crashed(seeded):
    r = client_for(seeded).post("/items/2401.10774/rate", data={"value": "meh"}, headers=HX)
    assert r.status_code == 200 and "must be one of" in r.text


def test_writes_without_htmx_header_are_refused(seeded):
    """A plain cross-site form can't add the HX-Request header (CSRF guard)."""
    r = client_for(seeded).post("/items/2401.10774/rate", data={"value": "useful"})
    assert r.status_code == 403
    assert services.item_rating(seeded, "2401.10774") is None


def test_writes_from_another_origin_are_refused(seeded):
    headers = {**HX, "Origin": "https://evil.example"}
    r = client_for(seeded).post("/items/2401.10774/rate", data={"value": "useful"}, headers=headers)
    assert r.status_code == 403


def test_requests_for_another_host_are_refused(seeded):
    """DNS rebinding: a page on evil.example re-pointed at 127.0.0.1 still says Host: evil."""
    r = client_for(seeded).get("/", headers={"Host": "evil.example"})
    assert r.status_code == 400


def test_source_text_is_escaped(filled_ctx):
    item = make_item("2410.09999", "<script>alert(1)</script> Paper", "Abstract <b>bold</b>")
    index_items(filled_ctx.conn, filled_ctx.embeddings, filled_ctx.vectors, [item])
    r = client_for(filled_ctx).get("/items/2410.09999")
    assert "<script>alert(1)</script>" not in r.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in r.text
    assert "<b>bold</b>" not in r.text


def test_markdown_drops_raw_html_and_unsafe_links():
    html = str(md("Hi <img src=x onerror=alert(1)> [x](javascript:alert(1)) [ok](https://a.b)"))
    assert "<img" not in html and 'href="javascript' not in html
    assert 'href="https://a.b"' in html and 'rel="noopener noreferrer"' in html


def test_only_kept_citations_are_linked():
    html = str(md("A [2401.10774] and B [2410.00002].", cited=["2401.10774"]))
    assert 'href="/items/2401.10774"' in html and 'href="/items/2410.00002"' not in html


def test_approving_a_topic_removal_updates_profile_and_badge(seeded):
    from radar.personalize import load_profile

    topic = load_profile(seeded).topics[0]
    with seeded.conn:
        safety_store.add_pending(seeded.conn, topic, "never rated useful")
    pid = safety_store.pending(seeded.conn)[0]["id"]
    c = client_for(seeded)
    assert 'class="badge">1<' in c.get("/profile").text
    r = c.post(f"/profile/pending/{pid}/approve", headers=HX)
    assert "Approved" in r.text
    assert 'id="badge-pending" hx-swap-oob="true"' in r.text  # the nav badge updates too
    assert topic not in load_profile(seeded).topics


def test_rejecting_keeps_the_topic(seeded):
    from radar.personalize import load_profile

    topic = load_profile(seeded).topics[0]
    with seeded.conn:
        safety_store.add_pending(seeded.conn, topic, "why")
    pid = safety_store.pending(seeded.conn)[0]["id"]
    client_for(seeded).post(f"/profile/pending/{pid}/reject", headers=HX)
    assert topic in load_profile(seeded).topics


def test_releasing_a_flagged_item(seeded):
    with seeded.conn:
        safety_store.flag_item(seeded.conn, "2410.00001", "Ignore previous instructions", False)
    c = client_for(seeded)
    assert "Ignore previous instructions" in c.get("/system/flagged").text
    r = c.post("/system/flagged/2410.00001/release", headers=HX)
    assert "Released" in r.text
    assert safety_store.held_ids(seeded.conn) == set()


def test_profile_update_without_the_agent_runs_inline(seeded):
    r = client_for(seeded).post("/profile/update", data={}, headers=HX)
    assert r.status_code == 200 and "Score weights" in r.text


# ---- asking ---------------------------------------------------------------------------------


def test_ask_runs_as_a_job_with_steps_and_a_cited_answer(filled_ctx):
    factory = fake_agent_factory(
        tool_call("store_search", query="Medusa"), "Medusa adds decoding heads [2401.10774]."
    )
    r = client_for(filled_ctx, factory).post(
        "/ask", data={"question": "What is Medusa?"}, headers=HX
    )
    assert r.status_code == 200
    assert "What is Medusa?" in r.text
    assert 'href="/items/2401.10774"' in r.text  # the finished answer, citation linked
    assert "Searching the local store" in r.text  # the steps, kept under the answer
    assert "Evidence check: enough to answer" in r.text
    convs = services.conversations(filled_ctx)
    assert convs and convs[0]["first_question"] == "What is Medusa?"
    turns = services.conversation(filled_ctx, convs[0]["thread_id"])
    assert "Medusa adds decoding heads" in turns[0]["answer"]


def test_ask_sends_the_thread_id_back_for_follow_ups(filled_ctx):
    factory = fake_agent_factory(tool_call("store_search", query="Medusa"), "Medusa [2401.10774].")
    r = client_for(filled_ctx, factory).post(
        "/ask", data={"question": "What is Medusa?", "thread_id": "abc123"}, headers=HX
    )
    assert 'id="thread-id" name="thread_id" value="abc123" hx-swap-oob="true"' in r.text
    assert services.conversation(filled_ctx, "abc123")


def test_conversation_page_shows_saved_answers(filled_ctx):
    factory = fake_agent_factory(tool_call("store_search", query="Medusa"), "Medusa [2401.10774].")
    c = client_for(filled_ctx, factory)
    c.post("/ask", data={"question": "What is Medusa?", "thread_id": "t1"}, headers=HX)
    r = c.get("/ask/t/t1")
    assert "What is Medusa?" in r.text and 'href="/items/2401.10774"' in r.text
    assert c.get("/ask/t/missing").status_code == 404


def test_a_failed_job_shows_its_error(filled_ctx):
    def broken(ctx, verify):
        raise RuntimeError("Ollama is not running")

    r = client_for(filled_ctx, broken).post("/ask", data={"question": "Q?"}, headers=HX)
    assert "failed" in r.text and "Ollama is not running" in r.text


def test_event_stream_replays_steps_then_done(filled_ctx):
    factory = fake_agent_factory(tool_call("store_search", query="Medusa"), "Medusa [2401.10774].")
    c = client_for(filled_ctx, factory)
    c.post("/ask", data={"question": "What is Medusa?"}, headers=HX)
    job = c.app.state.radar.runner.recent(1)[0]
    with c.stream("GET", f"/jobs/{job.id}/events") as r:
        body = "".join(r.iter_text())
    assert r.headers["content-type"].startswith("text/event-stream")
    assert body.count("event: step") == len(job.events)
    assert "id: 0\n" in body
    assert body.index("event: done") < body.index("event: close")
    # A reconnect says which step it saw last and gets only the rest
    with c.stream("GET", f"/jobs/{job.id}/events", headers={"Last-Event-ID": "1"}) as r:
        again = "".join(r.iter_text())
    assert again.count("event: step") == len(job.events) - 2


def test_job_page_and_unknown_job(filled_ctx):
    factory = fake_agent_factory(tool_call("store_search", query="Medusa"), "Medusa [2401.10774].")
    c = client_for(filled_ctx, factory)
    c.post("/ask", data={"question": "What is Medusa?"}, headers=HX)
    job = c.app.state.radar.runner.recent(1)[0]
    assert "Medusa" in c.get(f"/jobs/{job.id}").text
    assert c.get("/jobs/nope").status_code == 404


# ---- other jobs -----------------------------------------------------------------------------


def test_full_text_fetch_job(filled_ctx, monkeypatch):
    def fake_full_text(ctx, item_id, reason="manual", force=False, client=None):
        return services.FullTextResult(item_id, "ok", "arxiv_html", 3, 7)

    monkeypatch.setattr(services, "full_text", fake_full_text)
    r = client_for(filled_ctx).post("/items/2401.10774/fulltext", headers=HX)
    assert "3 sections, 7 chunks now searchable" in r.text


def test_digest_run_is_refused_while_another_runs(seeded):
    lock = seeded.settings.db_path.parent / "daily.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "started": "2026-11-21T07:00:00"}))
    c = client_for(seeded)
    assert "is using Qwen right now" in c.get("/").text  # the nav banner
    r = c.post("/digest/run", data={"collect": ""}, headers=HX)
    assert "already going" in r.text


# ---- the job runner on its own ----------------------------------------------------------------


def test_runner_runs_jobs_in_order_one_at_a_time():
    runner = JobRunner()
    runner.start()
    gate = threading.Event()
    first = runner.submit("t", "first", lambda h: gate.wait(2) and "one")
    second = runner.submit("t", "second", lambda h: "two")
    time.sleep(0.1)
    assert second.status == QUEUED and runner.ahead_of(second) == 1
    gate.set()
    for _ in range(50):
        if second.done:
            break
        time.sleep(0.05)
    runner.stop()
    assert (first.result, second.result) == ("one", "two")
    assert first.finished <= second.started


def test_cancel_queued_and_running_jobs():
    runner = JobRunner()
    runner.start()
    started = threading.Event()

    def slow(h):
        started.set()
        for _ in range(100):
            h.check()  # raises once a cancel is requested
            time.sleep(0.02)

    a = runner.submit("t", "a", slow)
    b = runner.submit("t", "b", lambda h: "never")
    started.wait(1)
    runner.cancel(b.id)
    runner.cancel(a.id)
    for _ in range(50):
        if a.done:
            break
        time.sleep(0.05)
    runner.stop()
    assert a.status == CANCELLED and b.status == CANCELLED and b.started is None


def test_failures_are_captured():
    runner = JobRunner(inline=True)
    job = runner.submit("t", "boom", lambda h: 1 / 0)
    assert job.status == FAILED and "ZeroDivisionError" in job.error
    ok = runner.submit("t", "fine", lambda h: h.step("hello") or 42)
    assert ok.status == DONE and ok.result == 42 and ok.events[0]["text"] == "hello"


# ---- the service pieces the web UI added ------------------------------------------------------


def test_ask_reports_steps_and_can_be_stopped(filled_ctx):
    def agent():
        return services.Agent(
            filled_ctx,
            llm=fake_chat(tool_call("store_search", query="Medusa"), "Medusa [2401.10774]."),
            judge=fake_chat(SUFFICIENT),
            checkpointer=InMemorySaver(),
            verify=False,
        )

    seen = []
    result = agent().ask("What is Medusa?", on_step=lambda n, e: seen.append((n, e)))
    nodes = [n for n, _ in seen]
    assert nodes[:3] == ["prepare", "rewrite_query", "reason"] and nodes[-1] == "answer"
    assert [e["step"] for _, es in seen for e in es] == [t["step"] for t in result.trace]
    with pytest.raises(services.Cancelled):
        agent().ask("What is Medusa?", should_stop=lambda: True)


def test_daily_lock_detects_live_and_stale_runs(ctx):
    lock = ctx.settings.db_path.parent / "daily.lock"
    assert services.daily_running(ctx) is None
    lock.write_text(json.dumps({"pid": os.getpid(), "started": "x"}))
    assert services.daily_running(ctx)["pid"] == os.getpid()
    with pytest.raises(services.DailyRunning):
        with services._daily_lock(ctx):
            pass
    lock.write_text(json.dumps({"pid": 2**22 + 12345, "started": "x"}))  # no such process
    assert services.daily_running(ctx) is None
    with services._daily_lock(ctx):
        assert services.daily_running(ctx) is not None
    assert not lock.exists()


def test_scheduled_run_skips_while_another_runs(ctx):
    def busy(ctx, mode=None):
        raise services.DailyRunning("started from the web UI")

    out = services.scheduled_daily(ctx, run_daily=busy, check=lambda: {"ok": True})
    assert out["status"] == "skipped"


def test_digest_view(seeded):
    d = services.digest_view(seeded)
    assert d["date"] == "2026-11-20" and d["older"] is None and d["newer"] is None
    assert [p["item_id"] for p in d["picks"]] == ["2401.10774", "2410.00002"]
    assert d["picks"][0]["summary"].startswith("Medusa")
    assert [x["item_id"] for x in d["dropped"]] == ["2005.11401"]
    assert services.digest_view(seeded, "2026-01-01") is None


def test_quiet_reads_are_not_logged_but_actions_are(seeded):
    tracing.configure(seeded.conn)
    c = client_for(seeded)
    c.get("/profile")
    assert not tool_calls.recent(seeded.conn, name="services.profile")
    c.post("/items/2401.10774/rate", data={"value": "useful"}, headers=HX)
    assert tool_calls.recent(seeded.conn, name="services.rate")


def test_use_connection_sends_rows_to_another_connection(ctx, settings):
    from radar.store.db import connect

    other = connect(settings.db_path)
    tracing.configure(ctx.conn)

    @tracing.traced("test.call", langsmith=False)
    def call():
        return 1

    with tracing.use_connection(other):
        other.execute("BEGIN IMMEDIATE")  # a transaction open on the other connection
        call()  # its row joins that transaction...
        assert not tool_calls.recent(ctx.conn, name="test.call")  # ...so it's not visible yet
        other.commit()
    assert tool_calls.recent(ctx.conn, name="test.call")
