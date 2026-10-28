import json

from conftest import fake_chat, tool_call
from langgraph.checkpoint.memory import InMemorySaver

from radar import services


def test_ask_returns_answer_and_trace(filled_ctx):
    agent = services.Agent(
        filled_ctx,
        llm=fake_chat(tool_call("store_search", query="Medusa"), "Heads [2401.10774]."),
        judge=fake_chat('{"sufficient": true, "missing": ""}'),
        checkpointer=InMemorySaver(),
    )
    result = agent.ask("What is Medusa?")
    assert result.cited == ["2401.10774"]
    assert result.trace and result.thread_id


def test_no_retrieval_skips_tools(filled_ctx):
    agent = services.Agent(
        filled_ctx,
        llm=fake_chat("From memory: Medusa is..."),
        judge=fake_chat(),
        checkpointer=InMemorySaver(),
    )
    result = agent.ask("What is Medusa?", no_retrieval=True)
    assert result.answer.startswith("From memory") and result.cited == []


def test_agent_uses_sqlite_checkpointer_by_default(filled_ctx, settings):
    services.Agent(filled_ctx, llm=fake_chat(), judge=fake_chat())
    assert settings.checkpoint_db_path.exists()


def test_eval_retrieval(filled_ctx, tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(
        json.dumps({"question": "Medusa decoding heads", "expected_ids": ["2401.10774"]})
        + "\n"
        + json.dumps({"question": "Medusa", "expected_ids": ["0000.00000"]})
        + "\n"
    )
    report = services.eval_retrieval(filled_ctx, f, k=3)
    assert report["recall"] == 0.5
    assert [r["hit"] for r in report["results"]] == [True, False]


def test_ingest_falls_back_to_rss(ctx, monkeypatch):
    from conftest import SAMPLE_ITEMS

    from radar.sources import arxiv

    def blocked(**kwargs):
        raise arxiv.ArxivUnavailable("429")

    monkeypatch.setattr(arxiv, "fetch_recent", blocked)
    monkeypatch.setattr(arxiv, "fetch_latest_rss", lambda cats: SAMPLE_ITEMS[:2])
    fetched, added, used = services.ingest(ctx)
    assert (fetched, added, used) == (2, 2, "rss")
