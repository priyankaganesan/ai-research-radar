"""Tools the agent can call. Each covers something the model can't do on its own."""

from langchain_core.tools import BaseTool

from radar.context import AppContext


def make_tools(ctx: AppContext) -> dict[str, BaseTool]:
    from radar.tools.fetch_full_text import make_fetch_full_text
    from radar.tools.search_arxiv import make_search_arxiv
    from radar.tools.semantic_scholar_lookup import make_semantic_scholar_lookup
    from radar.tools.store_search import make_store_search

    tools = [
        make_store_search(ctx),
        make_search_arxiv(ctx),
        make_semantic_scholar_lookup(ctx),
        make_fetch_full_text(ctx),
    ]
    return {t.name: t for t in tools}
