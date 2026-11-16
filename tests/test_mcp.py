"""MCP server and client, over an in-memory MCP session: no subprocess, no Ollama, no network."""

import asyncio
import json

from conftest import fake_chat, tool_call
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from mcp.shared.memory import create_connected_server_and_client_session

from radar.graphs.qa import build_qa_graph
from radar.mcp_client import MCPToolset, decode
from radar.mcp_server import build_server


def _session(ctx, tmp_path=None):
    server = build_server(lambda: ctx, digests_dir=tmp_path)
    return create_connected_server_and_client_session(server._mcp_server)


def test_server_lists_tools_resources_and_prompts(filled_ctx, tmp_path):
    async def go():
        async with _session(filled_ctx, tmp_path) as s:
            tools = {t.name: t for t in (await s.list_tools()).tools}
            resources = [str(r.uri) for r in (await s.list_resources()).resources]
            prompts = [p.name for p in (await s.list_prompts()).prompts]
            return tools, resources, prompts

    tools, resources, prompts = asyncio.run(go())
    assert set(tools) == {
        "store_search",
        "fetch_full_text",
        "search_arxiv",
        "semantic_scholar_lookup",
        "get_item",
        "ask",
        "rate",
    }
    assert tools["store_search"].annotations.readOnlyHint is True
    assert tools["search_arxiv"].annotations.readOnlyHint is False  # it writes to the store
    assert "query" in tools["store_search"].inputSchema["properties"]
    assert "radar://digest/latest" in resources
    assert {"radar://watchlist", "radar://profile"} <= set(resources)
    assert tools["rate"].annotations.readOnlyHint is False
    assert prompts == ["explain_paper"]


def test_store_search_over_mcp(filled_ctx):
    async def go():
        async with _session(filled_ctx) as s:
            return await s.call_tool("store_search", {"query": "Medusa decoding heads"})

    result = asyncio.run(go())
    passages = [json.loads(block.text) for block in result.content]
    assert passages[0]["item_id"] == "2401.10774"


def test_get_item_over_mcp(filled_ctx):
    async def go():
        async with _session(filled_ctx) as s:
            ok = await s.call_tool("get_item", {"item_id": "2401.10774"})
            missing = await s.call_tool("get_item", {"item_id": "nope"})
            return ok, missing

    ok, missing = asyncio.run(go())
    item = json.loads(ok.content[0].text)
    assert item["title"] == "Medusa: multiple decoding heads"
    assert "signals" in item
    assert missing.isError


def test_digest_resource_reads_latest_file(filled_ctx, tmp_path):
    (tmp_path / "2026-11-01.md").write_text("old")
    (tmp_path / "2026-11-02.md").write_text("# Digest for 2026-11-02")

    async def go():
        async with _session(filled_ctx, tmp_path) as s:
            return await s.read_resource("radar://digest/latest")

    assert asyncio.run(go()).contents[0].text == "# Digest for 2026-11-02"


def test_qa_graph_runs_on_tools_loaded_over_mcp(filled_ctx):
    """Dogfooding: the graph gets its tools from the MCP server and behaves the same."""
    toolset = MCPToolset(open_session=lambda: _session(filled_ctx))
    try:
        tools = toolset.tools(["store_search", "search_arxiv", "semantic_scholar_lookup"])
        assert set(tools) == {"store_search", "search_arxiv", "semantic_scholar_lookup"}
        llm = fake_chat(
            tool_call("store_search", query="Medusa decoding heads"),
            "Medusa adds decoding heads [2401.10774].",
        )
        graph = build_qa_graph(
            llm,
            fake_chat('{"sufficient": true, "missing": ""}'),
            tools,
            checkpointer=InMemorySaver(),
        )
        state = graph.invoke(
            {"messages": [HumanMessage("What is Medusa?")]}, {"configurable": {"thread_id": "m"}}
        )
    finally:
        toolset.close()
    assert state["cited"] == ["2401.10774"]


def test_decode_handles_mcp_shapes():
    assert decode([]) == []  # empty list result: no content blocks
    assert decode([{"type": "text", "text": '{"a": 1}'}]) == [{"a": 1}]
    two = [{"type": "text", "text": '{"a": 1}'}, {"type": "text", "text": '{"b": 2}'}]
    assert decode(two) == [{"a": 1}, {"b": 2}]
    assert decode("plain words") == "plain words"
