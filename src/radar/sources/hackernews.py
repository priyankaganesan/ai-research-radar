"""Hacker News: AI stories that practitioners are discussing, with their points.

Uses the public HN search API run by Algolia (no key). A story that links to an arXiv paper
adds an `hn_points` signal to that paper; any other story becomes an item of kind "post" whose
text is the story's title and, for Ask/Show HN posts, its body.

Endpoint: GET https://hn.algolia.com/api/v1/search_by_date?query=...&tags=story
  &numericFilters=created_at_i>UNIX_TIME,points>=N  -> {"hits": [{objectID, title, url,
  points, num_comments, created_at, story_text}, ...]}
"""

import html
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

from radar.sources import http

API_URL = "https://hn.algolia.com/api/v1/search_by_date"
DEFAULT_QUERIES = ("LLM", "language model", "AI model", "arxiv", "OpenAI", "Anthropic", "DeepMind")
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")
_TAG = re.compile(r"<[^>]+>")


@dataclass
class HNStory:
    id: str
    title: str
    url: str  # the linked page, or the HN thread for Ask/Show HN
    points: int
    comments: int
    created: date
    text: str

    @property
    def thread_url(self) -> str:
        return f"https://news.ycombinator.com/item?id={self.id}"

    @property
    def arxiv_id(self) -> str | None:
        m = _ARXIV_URL.search(self.url)
        return m.group(1) if m else None


def parse_hits(data: dict) -> list[HNStory]:
    out = []
    for h in data.get("hits", []):
        if not h.get("title"):
            continue
        story_id = str(h["objectID"])
        created = datetime.fromtimestamp(h.get("created_at_i", 0)).date()
        text = " ".join(html.unescape(_TAG.sub(" ", h.get("story_text") or "")).split())
        out.append(
            HNStory(
                id=story_id,
                title=h["title"],
                url=h.get("url") or f"https://news.ycombinator.com/item?id={story_id}",
                points=h.get("points") or 0,
                comments=h.get("num_comments") or 0,
                created=created,
                text=text,
            )
        )
    return out


def fetch_stories(
    queries: tuple[str, ...] | list[str] = DEFAULT_QUERIES,
    days: int = 2,
    min_points: int = 50,
    client: httpx.Client | None = None,
    delay_s: float = 0.5,
) -> list[HNStory]:
    """Stories from the last `days` days matching any query, with at least `min_points`."""
    client = client or http.make_client()
    since = int((datetime.now() - timedelta(days=days)).timestamp())
    seen: dict[str, HNStory] = {}
    for n, q in enumerate(queries):
        if n and delay_s:
            time.sleep(delay_s)
        resp = http.request_with_retry(
            client,
            API_URL,
            name="Hacker News search",
            params={
                "query": q,
                "tags": "story",
                "numericFilters": f"created_at_i>{since},points>={min_points}",
                "hitsPerPage": 50,
            },
        )
        for s in parse_hits(resp.json()):
            seen.setdefault(s.id, s)
    return sorted(seen.values(), key=lambda s: s.points, reverse=True)
