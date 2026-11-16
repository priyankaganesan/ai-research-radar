"""Hugging Face Daily Papers: the papers the community picked each day, with upvotes.

Every Daily Paper is an arXiv paper, so items use the arXiv ID and merge with papers already
fetched from arXiv. The upvote count goes into the `signals` table; it's the "early buzz"
signal the digest uses to spot papers that are popular before they have citations.

Endpoint (public, no key): GET https://huggingface.co/api/daily_papers?date=YYYY-MM-DD returns
a JSON list; each entry holds a `paper` object with `id`, `title`, `summary`, `upvotes`,
`authors` [{name}] and `publishedAt`.
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta

import httpx

from radar.schema import Item
from radar.sources import http
from radar.tracing import traced

API_URL = "https://huggingface.co/api/daily_papers"
_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")


@dataclass
class HFPaper:
    item: Item
    upvotes: int


def parse_daily(data: list[dict]) -> list[HFPaper]:
    """The API's JSON list -> papers with their upvotes. Pure function, easy to test."""
    out = []
    for entry in data:
        p = entry.get("paper") or {}
        arxiv_id = str(p.get("id", ""))
        if not _ARXIV_ID.match(arxiv_id) or not p.get("title"):
            continue  # not an arXiv paper, or a malformed entry
        published = (p.get("publishedAt") or entry.get("publishedAt") or "")[:10]
        item = Item(
            id=arxiv_id,
            source="hf",
            kind="paper",
            title=" ".join(p["title"].split()),
            abstract=" ".join((p.get("summary") or "").split()),
            authors=[a.get("name", "") for a in p.get("authors", []) if a.get("name")],
            published=date.fromisoformat(published) if published else date.today(),
            url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        )
        out.append(HFPaper(item=item, upvotes=int(p.get("upvotes") or 0)))
    return out


@traced("hf_papers.fetch_daily", "tool", kind="source")
def fetch_daily(day: date, client: httpx.Client | None = None) -> list[HFPaper]:
    """One day's Daily Papers."""
    client = client or http.make_client()
    resp = http.request_with_retry(
        client, API_URL, name="Hugging Face Daily Papers", params={"date": day.isoformat()}
    )
    return parse_daily(resp.json())


@traced("hf_papers.fetch_recent", "tool", kind="source")
def fetch_recent(days: int = 3, client: httpx.Client | None = None) -> list[HFPaper]:
    """Daily Papers for the last `days` days (today included). A paper listed twice keeps its
    highest upvote count."""
    client = client or http.make_client()
    best: dict[str, HFPaper] = {}
    for n in range(days):
        for p in fetch_daily(date.today() - timedelta(days=n), client):
            if p.item.id not in best or p.upvotes > best[p.item.id].upvotes:
                best[p.item.id] = p
    return sorted(best.values(), key=lambda p: p.upvotes, reverse=True)
