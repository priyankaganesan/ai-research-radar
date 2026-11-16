"""M4 end to end on a temp store: ratings -> profile, Profile agent, watchlist, gaps.
Scripted models and fake embeddings only; no Ollama, no network."""

from datetime import date, timedelta

from conftest import fake_chat, make_item, tool_call
from langgraph.checkpoint.memory import InMemorySaver
from test_daily import TODAY, _fill

from radar import personalize, services
from radar.graphs.daily import load_candidates
from radar.rag.index import index_items
from radar.store import digests as digest_store
from radar.store import memory
from radar.store import signals as signal_store

REL_HEAVY = {"relevance": 0.4, "attention": 0.0, "citations": 0.0, "recency": 0.05}


def _digest(ctx, rows, on=TODAY):
    """Write a digest directly: [(item_id, slot)], each with a relevance-heavy score."""
    with ctx.conn:
        digest_store.save_digest(
            ctx.conn,
            on.isoformat(),
            [
                {"item_id": i, "slot": s, "rank": n, "score": 0.45, "reason": "r"}
                | {"parts": REL_HEAVY}
                for n, (i, s) in enumerate(rows, 1)
            ],
        )


def test_rating_remembers_the_slot(filled_ctx):
    _digest(filled_ctx, [("2401.10774", "early_buzz")])
    r = services.rate(filled_ctx, "2401.10774", "useful", on=TODAY)
    assert r["slot"] == "early_buzz"
    r = services.rate(filled_ctx, "2005.11401", "skipped", on=TODAY)
    assert r["slot"] is None


def test_update_applies_each_rating_once(filled_ctx):
    _digest(filled_ctx, [("2401.10774", "main"), ("2410.00001", "main")])
    services.rate(filled_ctx, "2401.10774", "useful", on=TODAY)
    services.rate(filled_ctx, "2410.00001", "useful", on=TODAY)

    first = personalize.update_profile(filled_ctx)
    w = personalize.load_profile(filled_ctx).weights
    assert first.new_ratings == 2
    assert w["relevance"] > 0.45  # both picks were mostly relevance
    second = personalize.update_profile(filled_ctx)
    assert second.new_ratings == 0
    assert personalize.load_profile(filled_ctx).weights == w
    assert memory.recent_changes(filled_ctx.conn)[0]["kind"] == "weights"


def test_reserved_slots_grow_when_reserved_items_are_useful(filled_ctx):
    ids = ["2401.10774", "2005.11401", "2410.00001", "2410.00002"]
    _digest(filled_ctx, [(i, "lab_launch" if n % 2 else "early_buzz") for n, i in enumerate(ids)])
    for i in ids:
        services.rate(filled_ctx, i, "useful", on=TODAY)
    report = personalize.update_profile(filled_ctx)
    assert report.reserved_change == {"from": 2, "to": 3, "useful_rate": 1.0}
    p = services.profile(filled_ctx)
    assert p["reserved_slots"] == 3 and len(p["reserved_plan"]) == 3


def test_useful_rating_raises_relevance_of_similar_items(ctx):
    ctx.settings.digest_min_text_chars = 0
    ctx.settings.interest_topics = "retrieval augmented generation"
    items = [
        make_item("2410.00011", "Protein folding", "Protein structure prediction model."),
        make_item("2410.00012", "Protein design", "Protein structure design with diffusion."),
    ]
    index_items(ctx.conn, ctx.embeddings, ctx.vectors, items)

    def relevance(item_id):
        feats = load_candidates(ctx, date(2026, 10, 22), 7)
        return next(f.relevance for f in feats if f.item_id == item_id)

    before = relevance("2410.00012")
    services.rate(ctx, "2410.00011", "useful", on=TODAY)
    assert relevance("2410.00012") > before  # a similar paper now counts as relevant


def test_profile_agent_edits_topics_through_the_guard(filled_ctx):
    _digest(filled_ctx, [("2401.10774", "main"), ("2410.00002", "main"), ("2410.00001", "main")])
    for i in ("2401.10774", "2410.00002"):
        services.rate(filled_ctx, i, "useful", on=TODAY)
    services.rate(filled_ctx, "2410.00001", "not_relevant", on=TODAY)
    llm = fake_chat(
        '{"add": ["speculative decoding", "a", "x y z"], "remove": ["nope"], "why": "decoding"}'
    )
    report = personalize.update_profile(filled_ctx, llm)
    assert report.agent_ran
    assert report.added == ["speculative decoding", "a"] and len(report.rejected) == 2
    topics = personalize.load_profile(filled_ctx).topics
    assert "speculative decoding" in topics
    assert memory.recent_changes(filled_ctx.conn)[0]["kind"] == "topics"


def test_profile_agent_waits_for_enough_evidence(filled_ctx):
    services.rate(filled_ctx, "2401.10774", "useful", on=TODAY)
    report = personalize.update_profile(filled_ctx, fake_chat())  # would fail if called
    assert not report.agent_ran


def test_reset_keeps_ratings_but_restores_defaults(filled_ctx):
    _digest(filled_ctx, [("2401.10774", "main")])
    services.rate(filled_ctx, "2401.10774", "useful", on=TODAY)
    personalize.update_profile(filled_ctx)
    services.reset_profile(filled_ctx)
    p = services.profile(filled_ctx)
    assert p["weights"]["relevance"] == 0.45 and p["pending_ratings"] == 0


# ---- questions and gaps ----


def _agent(ctx, llm, judge):
    return services.Agent(ctx, llm=llm, judge=judge, checkpointer=InMemorySaver())


def test_unanswered_question_becomes_a_gap(filled_ctx):
    insufficient = '{"sufficient": false, "missing": "x"}'
    filled_ctx.settings.max_tool_steps = 2
    llm = fake_chat(tool_call("store_search", query="a"), tool_call("store_search", query="b"))
    agent = _agent(filled_ctx, llm, fake_chat(insufficient, insufficient))
    result = agent.ask("What is mixture of depths?")
    assert result.cited == []
    gaps = services.gaps(filled_ctx)
    assert [g["question"] for g in gaps] == ["What is mixture of depths?"]

    answered = _agent(
        filled_ctx,
        fake_chat(tool_call("store_search", query="Medusa"), "Heads [2401.10774]."),
        fake_chat('{"sufficient": true, "missing": ""}'),
    )
    answered.ask("What is Medusa?")
    qs = memory.recent_questions(filled_ctx.conn)
    assert [q["answered"] for q in qs] == [1, 0]
    assert len(services.gaps(filled_ctx)) == 1  # an answered question opens no gap


def test_collect_gaps_searches_and_resolves(filled_ctx):
    with filled_ctx.conn:
        memory.log_question(filled_ctx.conn, "q1", False, [], gap_topic="mixture of depths")
        memory.log_question(filled_ctx.conn, "q2", False, [], gap_topic="nothing out there")
    found = make_item("2404.02258", "Mixture-of-Depths", "Dynamic compute allocation per token.")

    def fake_search(topic, max_results=5):
        return [found] if "depths" in topic else []

    log = services.collect_gaps(filled_ctx, search=fake_search)
    assert len(log) == 2
    left = services.gaps(filled_ctx)
    assert [g["topic"] for g in left] == ["nothing out there"]
    assert left[0]["attempts"] == 1
    for _ in range(5):
        services.collect_gaps(filled_ctx, search=fake_search)
    assert services.gaps(filled_ctx)[0]["attempts"] == personalize.GAP_MAX_ATTEMPTS


# ---- the daily run and the watchlist ----


def test_daily_learns_and_watches(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx.settings, "digest_dir", tmp_path / "digests")
    _fill(ctx)
    with ctx.conn:
        digest_store.set_profile(ctx.conn, "reserved_slots", 1)
    result = services.daily(ctx, on=TODAY, collect=False, llm=fake_chat(*["S."] * 8))

    assert len(result.picks) == 8  # still 8 items with only 1 reserved slot
    assert [p["slot"] for p in result.picks].count("early_buzz") == 1
    assert "lab_launch" not in {p["slot"] for p in result.picks}
    assert result.profile_log and "learn" in result.timings
    assert "recheck_watchlist" in result.timings
    watched = services.watchlist(ctx)
    assert [w["item_id"] for w in watched] == ["2410.10007"]
    row = digest_store.get_digest(ctx.conn, TODAY.isoformat())[0]
    assert row["parts"]  # the score parts are kept for ratings


def test_watchlist_recheck_outcomes(filled_ctx):
    conn = filled_ctx.conn
    start = date(2026, 10, 1)
    with conn:
        for i in ("2401.10774", "2410.00001", "2410.00002"):
            signal_store.record(conn, i, start, citations=1)
            memory.watch(conn, i, start, 1, personalize.WATCH_FIRST_CHECK_DAYS)
        signal_store.record(conn, "2401.10774", start + timedelta(days=14), citations=20)

    day14 = start + timedelta(days=14)
    log = personalize.recheck_watchlist(filled_ctx, day14)
    status = {w["item_id"]: w["status"] for w in services.watchlist(filled_ctx)}
    assert status == {
        "2401.10774": "established",
        "2410.00001": "watching",
        "2410.00002": "watching",
    }
    assert len(log) == 3

    day28 = start + timedelta(days=28)
    personalize.recheck_watchlist(filled_ctx, day28)
    status = {w["item_id"]: w["status"] for w in services.watchlist(filled_ctx)}
    assert status["2410.00001"] == "dropped"
    assert services.watchlist(filled_ctx, "dropped")[0]["note"].startswith("+0 citations")
