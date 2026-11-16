"""AI lab blogs and announcements, read from their RSS or Atom feeds.

Each post becomes an item of kind "post". Its text is the post's summary or content from the
feed, with HTML stripped and cut to a few thousand characters: enough to embed and cite, and
the link points to the full post. Feeds move and break, so each one is fetched on its own and
a failure is reported without stopping the others.

The feed list is a setting (LAB_FEEDS) so you can add or remove labs. Some labs (Anthropic,
Meta AI) publish no feed at all.
"""

import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import feedparser
import httpx

from radar.schema import Item
from radar.sources import http
from radar.tracing import traced

DEFAULT_FEEDS = {
    "OpenAI": "https://openai.com/news/rss.xml",
    "Google DeepMind": "https://deepmind.google/blog/rss.xml",
    "Google Research": "https://research.google/blog/rss/",
    "Hugging Face": "https://huggingface.co/blog/feed.xml",
    "Microsoft Research": "https://www.microsoft.com/en-us/research/feed/",
    "Berkeley AI Research": "https://bair.berkeley.edu/blog/feed.xml",
    "NVIDIA Developer": "https://developer.nvidia.com/blog/feed",
}
MAX_CHARS = 4000  # about 1,000 tokens

_TAG = re.compile(r"<[^>]+>")


@dataclass
class LabFetch:
    items: list[Item] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)  # lab -> what went wrong


def post_id(link: str) -> str:
    """A stable ID from the post's link, so re-reading the feed never duplicates a post."""
    return "post:" + hashlib.sha1(link.encode()).hexdigest()[:12]


def clean(text: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", text)).split())


def parse_feed(xml: str, lab: str, since: date | None = None) -> list[Item]:
    """RSS/Atom XML -> post Items newer than `since`. Pure function, easy to test."""
    feed = feedparser.parse(xml)
    items = []
    for e in feed.entries:
        link = e.get("link")
        if not link or not e.get("title"):
            continue
        when = e.get("published_parsed") or e.get("updated_parsed")
        published = date(*when[:3]) if when else date.today()
        if since and published < since:
            continue
        # Prefer the full content when the feed includes it, else the summary.
        body = e.content[0].value if e.get("content") else e.get("summary", "")
        text = clean(body)[:MAX_CHARS] or clean(e.title)
        items.append(
            Item(
                id=post_id(link),
                source=f"lab:{lab}",
                kind="post",
                title=clean(e.title),
                abstract=text,
                authors=[lab],
                published=published,
                url=link,
            )
        )
    return items


@traced("lab_blogs.fetch_all", "tool", kind="source")
def fetch_all(
    feeds: dict[str, str], days: int = 14, client: httpx.Client | None = None
) -> LabFetch:
    """Posts from the last `days` days across all feeds."""
    client = client or http.make_client()
    since = date.today() - timedelta(days=days)
    result = LabFetch()
    for lab, url in feeds.items():
        try:
            resp = http.request_with_retry(client, url, name=lab, retries=1, backoff_s=2)
            result.items.extend(parse_feed(resp.text, lab, since))
        except (http.SourceUnavailable, httpx.HTTPError) as e:
            result.errors[lab] = str(e).splitlines()[0]
    return result
