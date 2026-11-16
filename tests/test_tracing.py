"""Tracing: the local tool-call log, and what @traced hands to LangSmith. No network."""

import asyncio
import json
from unittest.mock import MagicMock

import httpx
import pytest
from conftest import FakeEmbeddings, fake_chat, tool_call
from langgraph.checkpoint.memory import InMemorySaver
from langsmith.run_helpers import tracing_context
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from radar import services, tracing
from radar.mcp_server import build_server
from radar.sources import arxiv, hf_papers
from radar.store import tool_calls
from radar.tracing import traced


@pytest.fixture
def logged(ctx):
    """The store with the local tool-call log switched on."""
    tracing.configure(ctx.conn)
    return ctx


@pytest.fixture
def logged_filled(filled_ctx):
    tracing.configure(filled_ctx.conn)
    return filled_ctx


def rows(ctx) -> list[dict]:
    return list(reversed(tool_calls.recent(ctx.conn, limit=100)))  # oldest first


def test_a_service_call_is_logged_with_args_summary_and_timing(logged_filled):
    services.search(logged_filled, "Medusa decoding heads", k=3)

    [row] = rows(logged_filled)
    assert row["name"] == "services.search"
    assert row["kind"] == "service"
    assert json.loads(row["args"]) == {"query": "Medusa decoding heads", "k": 3}  # no ctx
    assert row["summary"].startswith("3 result(s), first 2401.10774")
    assert row["error"] is None
    assert row["duration_ms"] >= 0
    assert row["started_at"].endswith("+00:00")  # UTC
    assert row["parent_id"] is None and row["root_id"] == row["id"]
    assert row["langsmith_trace_id"] is None  # LangSmith is off in tests


def test_nothing_is_logged_when_the_log_is_off(filled_ctx):
    services.search(filled_ctx, "Medusa")
    assert rows(filled_ctx) == []


def test_errors_are_logged_and_still_raised(logged):
    @traced("demo.broken", "tool")
    def broken(x: int):
        raise ValueError(f"bad {x}")

    with pytest.raises(ValueError, match="bad 3"):
        broken(3)
    [row] = rows(logged)
    assert row["error"] == "ValueError: bad 3"
    assert row["summary"] is None


def test_nested_calls_form_a_tree(logged):
    @traced("demo.leaf", "tool", kind="source")
    def leaf(n):
        return list(range(n))

    @traced("demo.middle")
    def middle():
        return leaf(2) + leaf(3)

    @traced("demo.top")
    def top():
        return middle()

    top()
    by_name = {}
    for r in rows(logged):
        by_name.setdefault(r["name"], []).append(r)
    root = by_name["demo.top"][0]
    mid = by_name["demo.middle"][0]
    assert mid["parent_id"] == root["id"]
    assert all(r["parent_id"] == mid["id"] for r in by_name["demo.leaf"])
    assert {r["root_id"] for r in rows(logged)} == {root["id"]}

    tree = services.call_tree(logged, "last")
    assert [(r["name"], r["depth"]) for r in tree] == [
        ("demo.top", 0),
        ("demo.middle", 1),
        ("demo.leaf", 2),
        ("demo.leaf", 2),
    ]
    # Any call in the tree (or a prefix of its ID) finds the same tree.
    assert services.call_tree(logged, by_name["demo.leaf"][1]["id"][:6]) == tree


def test_async_functions_are_traced_too(logged):
    @traced("demo.async", "tool")
    async def fetch(q: str):
        return {"q": q}

    assert asyncio.run(fetch("x")) == {"q": "x"}
    [row] = rows(logged)
    assert row["name"] == "demo.async" and json.loads(row["args"]) == {"q": "x"}


def test_secrets_never_reach_the_log(monkeypatch, logged_filled):
    logged_filled.settings.semantic_scholar_api_key = "sk-very-secret"
    services.stats(logged_filled)
    services.search(logged_filled, "Medusa")
    dump = json.dumps(rows(logged_filled))
    assert "sk-very-secret" not in dump
    assert "AppContext" not in dump


def test_a_failing_source_fetch_is_logged(monkeypatch, logged):
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: None)  # skip the retry backoff

    def handler(request):
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(arxiv.ArxivUnavailable):
        arxiv.search("speculative decoding", max_results=3, client=client)
    [row] = rows(logged)
    assert row["name"] == "arxiv.search" and row["kind"] == "source"
    assert row["error"].startswith("ArxivUnavailable")
    assert json.loads(row["args"]) == {"query": "speculative decoding", "max_results": 3}


def test_hf_fetch_logs_each_day_inside_fetch_recent(logged):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    hf_papers.fetch_recent(days=2, client=client)
    names = [r["name"] for r in services.call_tree(logged, "last")]
    assert names == ["hf_papers.fetch_recent", "hf_papers.fetch_daily", "hf_papers.fetch_daily"]


def test_mcp_calls_are_logged_with_the_agent_tool_inside(logged_filled, tmp_path):
    server = build_server(lambda: logged_filled, digests_dir=tmp_path)

    async def go():
        async with create_connected_server_and_client_session(server._mcp_server) as s:
            await s.call_tool("store_search", {"query": "Medusa decoding heads"})
            await s.call_tool("get_item", {"item_id": "nope"})  # fails: no such item
            await s.read_resource("radar://stats")

    asyncio.run(go())
    got = [(r["name"], r["kind"], r["error"] is not None) for r in rows(logged_filled)]
    assert ("mcp.store_search", "mcp", False) in got
    assert ("tool.store_search", "agent_tool", False) in got
    assert ("mcp.get_item", "mcp", True) in got
    assert ("mcp.resource.stats", "mcp", False) in got
    outer = next(r for r in rows(logged_filled) if r["name"] == "mcp.store_search")
    inner = next(r for r in rows(logged_filled) if r["name"] == "tool.store_search")
    assert inner["parent_id"] == outer["id"]
    assert json.loads(outer["args"])["query"] == "Medusa decoding heads"


def test_tool_calls_inside_the_agent_graph_land_under_ask(logged_filled):
    """The graph runs tools on its own (worker threads included); they still nest under ask."""
    agent = services.Agent(
        logged_filled,
        llm=fake_chat(tool_call("store_search", query="Medusa"), "Heads [2401.10774]."),
        judge=fake_chat('{"sufficient": true, "missing": ""}'),
        checkpointer=InMemorySaver(),
    )
    agent.ask("What is Medusa?")
    tree = services.call_tree(logged_filled, "last")
    assert [(r["name"], r["depth"]) for r in tree] == [
        ("services.Agent.ask", 0),
        ("tool.store_search", 1),
    ]
    # Only the arguments actually passed are logged (defaults are left out).
    assert json.loads(tree[0]["args"]) == {"question": "What is Medusa?"}


def test_old_rows_are_pruned_when_the_log_starts(logged):
    with logged.conn:
        logged.conn.execute(
            "INSERT INTO tool_calls (id, root_id, name, kind, args, started_at, duration_ms)"
            " VALUES ('old', 'old', 'x', 'service', '{}', '2020-01-01T00:00:00+00:00', 1)"
        )
    tracing.configure(logged.conn, keep_days=30)
    assert rows(logged) == []


def test_langsmith_gets_the_run_with_clean_inputs_and_the_row_links_to_it(logged_filled):
    """With tracing on (a mock LangSmith client, so nothing leaves the machine), the run is sent
    with the right name, type and inputs, and the local row stores its trace and run IDs."""
    client = MagicMock()
    with tracing_context(enabled=True, client=client, project_name="test"):
        services.search(logged_filled, "Medusa decoding heads", k=2)

    [row] = rows(logged_filled)
    assert row["langsmith_trace_id"] and row["langsmith_run_id"]

    calls = client.method_calls  # everything the tracer asked the (mock) client to send
    payload = " ".join(str(c) for c in calls)
    assert "services.search" in payload and "retriever" in payload
    assert "'query': 'Medusa decoding heads'" in payload
    assert "AppContext" not in payload  # ctx was dropped by process_inputs
    assert "page_content" in payload  # Passages reshaped as documents for the retriever view


def test_traced_functions_keep_their_signature():
    import inspect

    assert list(inspect.signature(services.search).parameters) == [
        "ctx",
        "query",
        "k",
        "kind",
        "since",
    ]


def test_cli_calls_lists_tree_and_stats(monkeypatch, settings):
    import radar.models

    monkeypatch.setattr(radar.models, "get_embeddings", lambda s: FakeEmbeddings())
    from radar.cli import app
    from radar.context import build_context

    ctx = build_context(settings)  # turns the log on, like every CLI command
    services.stats(ctx)

    @traced("demo.fails", "tool")
    def fails():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fails()

    runner = CliRunner()
    out = runner.invoke(app, ["calls"])
    assert out.exit_code == 0, out.output
    assert "services.stats" in out.output and "ERROR" in out.output and "boom" in out.output

    out = runner.invoke(app, ["calls", "--errors"])
    assert "demo.fails" in out.output and "services.stats" not in out.output

    out = runner.invoke(app, ["calls", "--tree", "last"])
    assert "demo.fails" in out.output and "tracing was off" in out.output

    out = runner.invoke(app, ["calls", "--stats"])
    assert "services.stats" in out.output and "1 failed" in out.output
