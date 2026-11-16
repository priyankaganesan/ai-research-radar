"""Semantic Scholar (S2) Graph API: citation counts, TLDRs and paper search.

Two uses:
- enrich: look up papers already in the store (by arXiv ID, up to 500 per request) to record
  their citation counts and S2's one-sentence TLDR.
- search: find papers by keywords, for the agent's `semantic_scholar_lookup` tool. Results
  come with abstracts, so this also works when the arXiv API is rate-limiting us.

Without an API key, requests share a public pool that is often rate-limited (429); the shared
retry helper waits and tries again. A free key (SEMANTIC_SCHOLAR_API_KEY) gives a steady
1 request per second. Docs: https://api.semanticscholar.org/api-docs/graph
"""

import time
from dataclasses import dataclass
from datetime import date

import httpx

from radar.schema import Item
from radar.sources import http
from radar.tracing import traced

API_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = (
    "title,abstract,externalIds,citationCount,influentialCitationCount,tldr,publicationDate,"
    "authors,url"
)
BATCH_SIZE = 500  # the batch endpoint's limit
POLITE_DELAY_S = 1.1  # keyed clients get 1 request per second


@dataclass
class S2Paper:
    paper_id: str
    arxiv_id: str | None
    title: str
    abstract: str
    tldr: str | None
    citations: int
    influential_citations: int
    published: date | None
    url: str
    authors: list[str]

    def to_item(self) -> Item | None:
        """An Item for the store, or None if there's nothing to read (no abstract or TLDR)."""
        text = self.abstract or self.tldr
        if not text:
            return None
        if self.arxiv_id:
            item_id, url = self.arxiv_id, f"https://arxiv.org/abs/{self.arxiv_id}"
        else:
            item_id, url = f"s2:{self.paper_id}", self.url
        return Item(
            id=item_id,
            source="s2",
            kind="paper",
            title=self.title,
            abstract=text,
            authors=self.authors,
            published=self.published or date.today(),
            url=url,
            tldr=self.tldr,
        )


def make_client(api_key: str = "") -> httpx.Client:
    return http.make_client(headers={"x-api-key": api_key} if api_key else None)


def parse_paper(p: dict) -> S2Paper:
    ext = p.get("externalIds") or {}
    pub = p.get("publicationDate")
    return S2Paper(
        paper_id=p["paperId"],
        arxiv_id=ext.get("ArXiv"),
        title=" ".join((p.get("title") or "").split()),
        abstract=" ".join((p.get("abstract") or "").split()),
        tldr=(p.get("tldr") or {}).get("text"),
        citations=p.get("citationCount") or 0,
        influential_citations=p.get("influentialCitationCount") or 0,
        published=date.fromisoformat(pub) if pub else None,
        url=p.get("url") or f"https://www.semanticscholar.org/paper/{p['paperId']}",
        authors=[a["name"] for a in p.get("authors") or [] if a.get("name")],
    )


@traced("semantic_scholar.lookup_arxiv", "tool", kind="source")
def lookup_arxiv(
    arxiv_ids: list[str], client: httpx.Client | None = None, delay_s: float = POLITE_DELAY_S
) -> dict[str, S2Paper]:
    """S2 records for these arXiv IDs, keyed by arXiv ID. IDs S2 doesn't know are left out."""
    client = client or make_client()
    found: dict[str, S2Paper] = {}
    for start in range(0, len(arxiv_ids), BATCH_SIZE):
        if start and delay_s:
            time.sleep(delay_s)
        batch = arxiv_ids[start : start + BATCH_SIZE]
        resp = http.request_with_retry(
            client,
            f"{API_URL}/paper/batch",
            name="Semantic Scholar",
            method="POST",
            params={"fields": FIELDS},
            json={"ids": [f"ARXIV:{i}" for i in batch]},
        )
        # The reply is a list in request order, with null for IDs it couldn't find. Key by
        # the arXiv ID S2 reports, falling back to the position if it reports none.
        for asked, p in zip(batch, resp.json(), strict=False):
            if p:
                paper = parse_paper(p)
                found[paper.arxiv_id or asked] = paper
    return found


@traced("semantic_scholar.search", "tool", kind="source")
def search(query: str, limit: int = 5, client: httpx.Client | None = None) -> list[S2Paper]:
    """Keyword search across Semantic Scholar, most relevant first."""
    client = client or make_client()
    resp = http.request_with_retry(
        client,
        f"{API_URL}/paper/search",
        name="Semantic Scholar",
        params={"query": query, "limit": limit, "fields": FIELDS},
    )
    return [parse_paper(p) for p in resp.json().get("data", [])]
