"""Tool: read a paper's full text, for when the abstract lacks method detail (retrieval, M5)."""

from dataclasses import asdict

from langchain_core.tools import BaseTool, tool

from radar.context import AppContext
from radar.tracing import traced


def make_fetch_full_text(ctx: AppContext) -> BaseTool:
    @tool
    @traced("tool.fetch_full_text", "tool", kind="agent_tool", langsmith=False)
    def fetch_full_text(item_id: str, query: str) -> list[dict]:
        """Read the full text of one paper (or lab post) already in the store, and return the
        sections most relevant to the query.

        Use this when a passage you already have is the right paper but its abstract doesn't
        explain what the question asks: how a method works, settings, numbers, ablations or
        limitations. The first call downloads the paper (slow); later calls are instant.

        Args:
            item_id: the ID of a paper from earlier results, e.g. "2401.10774"
            query: what to look for inside the paper, e.g. "how the draft heads are trained"
        """
        from radar import services

        got = services.full_text(ctx, item_id, reason="agent tool")
        if got.status == "failed":
            raise ValueError(f"No full text for {item_id}: {got.error}")
        return [asdict(p) for p in services.search_full_text(ctx, item_id, query, k=3)]

    return fetch_full_text
