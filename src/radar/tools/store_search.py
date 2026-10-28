"""Tool: search the local knowledge store (grounding)."""

from dataclasses import asdict

from langchain_core.tools import BaseTool, tool

from radar.context import AppContext
from radar.rag.retrieve import hybrid_search


def make_store_search(ctx: AppContext) -> BaseTool:
    @tool
    def store_search(query: str, since: str | None = None) -> list[dict]:
        """Search the local research store for passages relevant to the query.

        Use this first. Returns the best matching papers with their IDs, titles, dates and text.

        Args:
            query: what to look for, in plain words (e.g. "knowledge conflicts in RAG")
            since: optional ISO date (YYYY-MM-DD); only return items published on or after it
        """
        passages = hybrid_search(
            ctx.conn, ctx.embeddings, ctx.vectors, query, k=ctx.settings.retrieval_k, since=since
        )
        return [asdict(p) for p in passages]

    return store_search
