"""Tool: Semantic Scholar search, with citation counts and TLDRs (retrieval).

A second way to reach papers outside the store. Results come with citation counts, which help
the model tell an established method from a brand-new idea, and they work even when the arXiv
API is rate-limiting us. Papers found are saved to the store, like search_arxiv.
"""

from dataclasses import asdict
from datetime import date

from langchain_core.tools import BaseTool, tool

from radar.context import AppContext
from radar.rag.index import index_items
from radar.schema import Passage
from radar.sources import semantic_scholar
from radar.store import items as item_store
from radar.store import signals as signal_store
from radar.tracing import traced


def make_semantic_scholar_lookup(ctx: AppContext) -> BaseTool:
    @tool
    @traced("tool.semantic_scholar_lookup", "tool", kind="agent_tool", langsmith=False)
    def semantic_scholar_lookup(query: str, max_results: int = 5) -> list[dict]:
        """Search Semantic Scholar for papers, with citation counts and one-line summaries.

        Use this when the store lacks what the question needs, when you need to know how
        established a paper is (citations), or when search_arxiv fails.

        Args:
            query: keywords, e.g. "speculative decoding draft model"
            max_results: how many papers to fetch (1-10)
        """
        client = semantic_scholar.make_client(ctx.settings.semantic_scholar_api_key)
        papers = semantic_scholar.search(query, limit=max(1, min(max_results, 10)), client=client)
        pairs = [(p, i) for p in papers if (i := p.to_item())]
        index_items(ctx.conn, ctx.embeddings, ctx.vectors, [i for _, i in pairs])
        with ctx.conn:
            for p, i in pairs:
                if item_store.exists(ctx.conn, i.id):
                    signal_store.record(ctx.conn, i.id, date.today(), citations=p.citations)
                    if p.tldr:
                        signal_store.set_tldr(ctx.conn, i.id, p.tldr)
        out = []
        for p, i in pairs:
            extra = f"Cited by {p.citations}." + (f" TLDR: {p.tldr}" if p.tldr else "")
            text = f"{i.title}\n\n{i.abstract}\n\n{extra}"
            published = i.published.isoformat()
            out.append(asdict(Passage(i.id, i.title, i.url, published, text, score=0.0)))
        return out

    return semantic_scholar_lookup
