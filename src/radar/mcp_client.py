"""Load the agent's tools over MCP instead of importing them (TOOLS_VIA_MCP=true).

This is "dogfooding": the Q&A graph becomes an MCP client of Radar's own MCP server, so every
tool call travels the full protocol (JSON-RPC over the server's stdin/stdout) and shows up in
the trace. Behaviour is the same as importing the tools directly; only the path differs.

MCP clients are async, and the Q&A graph is sync. So the client session lives on an event loop
in a background thread, and each tool gets a small sync wrapper that hands the call to that loop
and waits for the answer.
"""

import asyncio
import json
import os
import sys
import threading
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SessionFactory = Callable[[], AbstractAsyncContextManager[ClientSession]]


@asynccontextmanager
async def stdio_session():
    """Start `radar mcp` as a subprocess and open an MCP session to it."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "radar.cli", "mcp"], env=dict(os.environ)
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        yield session


class MCPToolset:
    """One MCP session, kept open on a background event loop for the life of the process."""

    def __init__(self, open_session: SessionFactory = stdio_session, timeout: float = 120):
        self.timeout = timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._ready: asyncio.Future | None = None
        self._stop: asyncio.Event | None = None
        self._main = asyncio.run_coroutine_threadsafe(self._run(open_session), self._loop)
        self._mcp_tools: list[BaseTool] = asyncio.run_coroutine_threadsafe(
            self._wait_ready(), self._loop
        ).result(timeout)

    async def _run(self, open_session: SessionFactory) -> None:
        # Open and close the session in this one task: the MCP SDK (anyio) requires it.
        self._stop = asyncio.Event()
        self._ready = self._ready or self._loop.create_future()
        try:
            async with open_session() as session:
                self._ready.set_result(await load_mcp_tools(session))
                await self._stop.wait()
        except Exception as e:
            if not self._ready.done():
                self._ready.set_exception(e)
            raise

    async def _wait_ready(self) -> list[BaseTool]:
        self._ready = self._ready or self._loop.create_future()
        return await self._ready

    def tools(self, names: list[str] | None = None) -> dict[str, BaseTool]:
        """Sync LangChain tools that call the MCP server. `names` picks a subset."""
        chosen = [t for t in self._mcp_tools if names is None or t.name in names]
        return {t.name: self._sync_tool(t) for t in chosen}

    def _sync_tool(self, remote: BaseTool) -> BaseTool:
        def call(**kwargs) -> list[dict] | dict | str:
            future = asyncio.run_coroutine_threadsafe(remote.ainvoke(kwargs), self._loop)
            return decode(future.result(self.timeout))

        return StructuredTool.from_function(
            func=call,
            name=remote.name,
            description=remote.description,
            args_schema=remote.args_schema,  # the JSON schema the server advertised
        )

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        try:
            self._main.result(10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


def decode(content) -> list[dict] | dict | str:
    """Turn MCP text content back into Python data.

    The server sends a list result as one JSON text block per element, so a list of passages
    arrives as a list of blocks. An empty list arrives as no blocks at all.
    """
    if isinstance(content, str):
        blocks = [content]
    else:
        blocks = [b["text"] if isinstance(b, dict) else str(b) for b in content]
    parsed = []
    for text in blocks:
        try:
            parsed.append(json.loads(text))
        except json.JSONDecodeError:
            parsed.append(text)
    if len(parsed) == 1 and isinstance(parsed[0], list):
        return parsed[0]
    if len(parsed) == 1 and not isinstance(parsed[0], dict):
        return parsed[0]
    return parsed
