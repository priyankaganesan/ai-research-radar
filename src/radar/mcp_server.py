"""MCP server (M1.6): Radar's tools, offered to any MCP client (Claude Code, Claude Desktop, ...).

MCP (Model Context Protocol) is a standard way for an AI app to discover and call tools. The
client starts `radar mcp` as a subprocess and talks JSON-RPC over its stdin/stdout ("stdio").

The search tools are not rewritten here: they are the same LangChain tools the Q&A graph uses
(`radar.tools`), converted to MCP tools with `langchain_mcp_adapters.tools.to_fastmcp`. So a tool
is defined once and reachable two ways. The rest (`get_item`, `ask`, the digest resource, the
prompt) are thin wrappers over `radar.services`, like the CLI.

On stdio, stdout *is* the protocol channel: a stray print() corrupts it. Log to stderr only.
"""

import logging
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from langchain_mcp_adapters.tools import to_fastmcp
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from radar.context import AppContext

log = logging.getLogger("radar.mcp")

INSTRUCTIONS = """AI Research Radar: a local store of recent AI papers and lab posts, searched \
with hybrid (vector + keyword) retrieval. Start with store_search. search_arxiv and \
semantic_scholar_lookup search live and save what they find into the store. ask runs the \
full local Q&A agent (slow: it calls a local model several times)."""

# Hints the client shows the user. The live-search tools write to the store, so they're not
# read-only; a careful client can ask before calling them.
ANNOTATIONS = {
    "store_search": ToolAnnotations(title="Search the Radar store", readOnlyHint=True),
    "search_arxiv": ToolAnnotations(
        title="Search arXiv (saves results to the store)", readOnlyHint=False, openWorldHint=True
    ),
    "semantic_scholar_lookup": ToolAnnotations(
        title="Search Semantic Scholar (saves results to the store)",
        readOnlyHint=False,
        openWorldHint=True,
    ),
}


def build_server(get_ctx: Callable[[], AppContext], digests_dir: Path | None = None) -> FastMCP:
    """Build the server. `get_ctx` is called once, so tests can pass a temp store."""
    from radar import services
    from radar.tools import make_tools

    ctx = get_ctx()
    digests_dir = digests_dir or Path(ctx.settings.digest_dir)

    tools = []
    for tool in make_tools(ctx).values():
        t = to_fastmcp(tool)
        t.annotations = ANNOTATIONS.get(tool.name)
        tools.append(t)
    mcp = FastMCP("radar", instructions=INSTRUCTIONS, tools=tools)

    @mcp.tool(annotations=ToolAnnotations(title="Get one item", readOnlyHint=True))
    def get_item(item_id: str) -> dict:
        """Get one paper or post from the store: title, abstract, authors, link, date, TLDR,
        and its latest attention signals (Hugging Face upvotes, citations, HN points).

        Args:
            item_id: the ID shown by store_search, e.g. "2401.10774" or "post:ab12..."
        """
        found = services.show(ctx, item_id)
        if found is None:
            raise ValueError(f"No item {item_id!r} in the store.")
        item, signals = found
        out = asdict(item)
        out["published"] = item.published.isoformat()
        out["signals"] = signals
        return out

    agent = None  # built on first use: it loads the chat model and the checkpointer

    @mcp.tool(annotations=ToolAnnotations(title="Ask the Radar agent", readOnlyHint=True))
    def ask(question: str, thread_id: str | None = None) -> dict:
        """Ask the Radar Q&A agent. It searches, checks its evidence and answers with
        citations, or says it can't answer. Runs on a local model, so it takes a while.

        Args:
            question: the question, in plain words
            thread_id: pass the thread_id from an earlier answer to ask a follow-up
        """
        nonlocal agent
        if agent is None:
            agent = services.Agent(ctx)
        result = agent.ask(question, thread_id=thread_id)
        return {"answer": result.answer, "cited": result.cited, "thread_id": result.thread_id}

    @mcp.resource("radar://digest/latest", mime_type="text/markdown")
    def latest_digest() -> str:
        """The most recent daily digest (Markdown)."""
        files = sorted(digests_dir.glob("*.md")) if digests_dir.exists() else []
        if not files:
            return "No digest yet. Run `radar daily` to make one."
        return files[-1].read_text()

    @mcp.resource("radar://stats", mime_type="application/json")
    def stats() -> dict:
        """How many items and vectors the store holds, and which vector backend is active."""
        return services.stats(ctx)

    @mcp.prompt()
    def explain_paper(item_id: str) -> str:
        """Explain one paper from the store for a practitioner."""
        return (
            f"Use get_item to fetch {item_id} from AI Research Radar, then explain it for an "
            "ML practitioner: the problem, the key idea, how it's evaluated, and when it would "
            "matter in practice. Use store_search to find one or two related papers in the "
            "store and say how they differ. Cite items by ID in square brackets."
        )

    return mcp


def run(http: bool = False, port: int = 8765) -> None:
    """Entry point for `radar mcp`."""
    from radar.config import get_settings
    from radar.context import build_context

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # never stdout
    server = build_server(lambda: build_context(get_settings()))
    if http:
        server.settings.host = "127.0.0.1"
        server.settings.port = port
        log.info("Radar MCP on http://127.0.0.1:%s/mcp", port)
        server.run(transport="streamable-http")
    else:
        server.run(transport="stdio")
