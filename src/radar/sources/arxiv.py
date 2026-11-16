"""arXiv API client: recent papers by category, and keyword search.

The API returns Atom XML, which feedparser turns into Python dicts. arXiv asks clients to wait
about 3 seconds between requests, so paging sleeps between calls. When arXiv is busy it answers
429 (too many requests) or 503; `sources/http.py` then waits and retries with exponential backoff.
API docs: https://info.arxiv.org/help/api/user-manual.html

There's also a fallback: arXiv's RSS feeds (rss.arxiv.org) are served separately from the API,
list each day's new announcements with abstracts, and are often reachable when the API is
rate-limited. They only cover the latest announcement day, though.
"""

import re
import time
from datetime import date, datetime, timedelta

import feedparser
import httpx

from radar.schema import Item
from radar.sources import http
from radar.tracing import traced

API_URL = "https://export.arxiv.org/api/query"
RSS_URL = "https://rss.arxiv.org/rss/{cats}"
DEFAULT_CATEGORIES = ("cs.AI", "cs.CL", "cs.LG")
PAGE_SIZE = 100
POLITE_DELAY_S = 3.0
BACKOFF_S = http.BACKOFF_S


class ArxivUnavailable(http.SourceUnavailable):
    """arXiv kept refusing requests (rate limit or outage) after all retries."""


_VERSION = re.compile(r"v\d+$")


def parse_feed(xml: str) -> list[Item]:
    """Atom XML from the arXiv API -> Items. Pure function, so it's easy to test."""
    feed = feedparser.parse(xml)
    items = []
    for e in feed.entries:
        # e.id looks like "http://arxiv.org/abs/2410.01234v2"; keep the ID without the version
        arxiv_id = _VERSION.sub("", e.id.rsplit("/abs/", 1)[-1])
        pdf = next(
            (link.href for link in e.get("links", []) if link.get("type") == "application/pdf"),
            None,
        )
        items.append(
            Item(
                id=arxiv_id,
                source="arxiv",
                kind="paper",
                title=" ".join(e.title.split()),  # titles contain line breaks
                abstract=" ".join(e.summary.split()),
                authors=[a.name for a in e.get("authors", [])],
                published=datetime.strptime(e.published[:10], "%Y-%m-%d").date(),
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf,
                categories=[t.term for t in e.get("tags", [])],
            )
        )
    return items


def make_client() -> httpx.Client:
    # arXiv asks API users to identify themselves; anonymous default user agents get throttled.
    return http.make_client()


def _get(client: httpx.Client, params: dict, backoff_s: float = BACKOFF_S) -> list[Item]:
    try:
        resp = http.request_with_retry(
            client, API_URL, name="arXiv API", params=params, backoff_s=backoff_s
        )
    except http.SourceUnavailable as e:
        raise ArxivUnavailable(
            f"{e} It rate-limits busy networks; try `--source rss`, wait a while, or fetch less"
            " with --max."
        ) from e
    return parse_feed(resp.text)


@traced("arxiv.fetch_recent", "tool", kind="source")
def fetch_recent(
    days: int = 7,
    categories: tuple[str, ...] | list[str] = DEFAULT_CATEGORIES,
    max_results: int = 500,
    client: httpx.Client | None = None,
    delay_s: float = POLITE_DELAY_S,
) -> list[Item]:
    """Papers submitted in the last `days` days in the given categories, newest first."""
    client = client or make_client()
    since = date.today() - timedelta(days=days)
    query = " OR ".join(f"cat:{c}" for c in categories)
    items: list[Item] = []
    for start in range(0, max_results, PAGE_SIZE):
        page = _get(
            client,
            {
                "search_query": query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "start": start,
                "max_results": min(PAGE_SIZE, max_results - start),
            },
        )
        fresh = [i for i in page if i.published >= since]
        items.extend(fresh)
        if len(fresh) < len(page) or len(page) == 0:
            break  # reached papers older than the window, or the end of results
        if delay_s:
            time.sleep(delay_s)
    return items


def parse_rss(xml: str) -> list[Item]:
    """RSS from rss.arxiv.org -> Items. Skips "replace" entries (revisions of old papers)."""
    feed = feedparser.parse(xml)
    items = []
    for e in feed.entries:
        if e.get("arxiv_announce_type", "new") not in ("new", "cross"):
            continue
        arxiv_id = _VERSION.sub("", e.link.rsplit("/abs/", 1)[-1])
        desc = e.get("summary", "")
        abstract = desc.split("Abstract:", 1)[-1]
        published = date(*e.published_parsed[:3]) if e.get("published_parsed") else date.today()
        items.append(
            Item(
                id=arxiv_id,
                source="arxiv",
                kind="paper",
                title=" ".join(e.title.split()),
                abstract=" ".join(abstract.split()),
                authors=[a.strip() for a in e.get("author", "").split(",") if a.strip()],
                published=published,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                categories=[t.term for t in e.get("tags", [])],
            )
        )
    return items


@traced("arxiv.fetch_latest_rss", "tool", kind="source")
def fetch_latest_rss(
    categories: tuple[str, ...] | list[str] = DEFAULT_CATEGORIES,
    client: httpx.Client | None = None,
) -> list[Item]:
    """The latest announcement day's new papers, from the RSS feed (one request for all cats)."""
    client = client or make_client()
    try:
        resp = client.get(RSS_URL.format(cats="+".join(categories)))
    except httpx.TransportError as e:
        raise ArxivUnavailable(f"arXiv RSS gave a network error ({type(e).__name__}).") from e
    if resp.status_code in (429, 503):
        raise ArxivUnavailable(f"arXiv RSS returned {resp.status_code} too.")
    resp.raise_for_status()
    return parse_rss(resp.text)


@traced("arxiv.search", "tool", kind="source")
def search(query: str, max_results: int = 10, client: httpx.Client | None = None) -> list[Item]:
    """Keyword search across all of arXiv, most relevant first."""
    client = client or make_client()
    terms = " AND ".join(f"all:{t}" for t in re.findall(r"[\w\-]+", query))
    if not terms:
        return []
    return _get(
        client,
        {"search_query": terms, "sortBy": "relevance", "max_results": max_results},
    )
