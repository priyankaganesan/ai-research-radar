"""Tool: live arXiv search, for when the store doesn't have enough (retrieval)."""

from dataclasses import asdict

from langchain_core.tools import BaseTool, tool

from radar.context import AppContext
from radar.rag.index import index_items
from radar.schema import Passage
from radar.sources import arxiv
from radar.tracing import traced


def make_search_arxiv(ctx: AppContext) -> BaseTool:
    @tool
    @traced("tool.search_arxiv", "tool", kind="agent_tool", langsmith=False)
    def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
        """Search arXiv directly for papers not yet in the local store, and save them to it.

        Use this when store_search returned nothing relevant or the evidence is missing a part
        of the question.

        Args:
            query: keywords, e.g. "Medusa decoding heads"
            max_results: how many papers to fetch (1-10)
        """
        found = arxiv.search(query, max_results=max(1, min(max_results, 10)))
        index_items(ctx.conn, ctx.embeddings, ctx.vectors, found)
        return [
            asdict(
                Passage(
                    item_id=i.id,
                    title=i.title,
                    url=i.url,
                    published=i.published.isoformat(),
                    text=f"{i.title}\n\n{i.abstract}",
                    score=0.0,
                )
            )
            for i in found
        ]

    return search_arxiv
