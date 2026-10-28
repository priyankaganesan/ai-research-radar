"""Parsers and clients for the M3 sources, against hand-written fixtures in the documented
formats (tests never touch the network)."""

import json
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from conftest import FIXTURES

from radar.sources import hackernews, hf_papers, http, lab_blogs, semantic_scholar


def fixture_json(name: str):
    return json.loads((FIXTURES / name).read_text())


def lab_xml(name: str) -> str:
    recent = date.today() - timedelta(days=1)
    rfc822 = format_datetime(datetime(recent.year, recent.month, recent.day, tzinfo=UTC))
    return (
        (FIXTURES / name)
        .read_text()
        .replace("{RECENT_RFC822}", rfc822)
        .replace("{RECENT_ISO}", recent.isoformat())
    )


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- shared HTTP helper ---


def test_retry_helper_retries_then_gives_up(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429)

    with pytest.raises(http.SourceUnavailable, match="Example gave status 429 after 3 tries"):
        http.request_with_retry(mock_client(handler), "https://x.test", name="Example", retries=2)
    assert calls["n"] == 3


def test_retry_helper_does_not_retry_a_404():
    with pytest.raises(httpx.HTTPStatusError):
        http.request_with_retry(
            mock_client(lambda r: httpx.Response(404)), "https://x.test", name="Example"
        )


# --- Hugging Face Daily Papers ---


def test_hf_parse_keeps_arxiv_papers_with_upvotes():
    papers = hf_papers.parse_daily(fixture_json("hf_daily.json"))
    assert [p.item.id for p in papers] == ["2610.31819", "2401.10774"]  # non-arXiv entry skipped
    top = papers[0]
    assert top.upvotes == 87
    assert top.item.title == "Speculative Diffusion is a Tale of Two Tradeoffs"
    assert top.item.source == "hf" and top.item.kind == "paper"
    assert top.item.published == date(2026, 10, 26)
    assert top.item.authors == ["Ada Lovelace", "Alan Turing"]
    assert top.item.url == "https://arxiv.org/abs/2610.31819"


def test_hf_fetch_recent_asks_for_each_day_and_keeps_best_upvotes():
    days = []

    def handler(request):
        days.append(request.url.params["date"])
        return httpx.Response(200, json=fixture_json("hf_daily.json"))

    papers = hf_papers.fetch_recent(days=2, client=mock_client(handler))
    assert days == [date.today().isoformat(), (date.today() - timedelta(days=1)).isoformat()]
    assert [p.item.id for p in papers] == ["2610.31819", "2401.10774"]  # deduplicated


# --- Semantic Scholar ---


def test_s2_batch_posts_arxiv_ids_and_skips_unknown():
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["body"] = json.loads(request.content)
        seen["fields"] = request.url.params["fields"]
        return httpx.Response(200, json=fixture_json("s2_batch.json"))

    found = semantic_scholar.lookup_arxiv(["2401.10774", "2699.99999"], mock_client(handler))
    assert seen["method"] == "POST"
    assert seen["body"] == {"ids": ["ARXIV:2401.10774", "ARXIV:2699.99999"]}
    assert "tldr" in seen["fields"] and "citationCount" in seen["fields"]
    assert list(found) == ["2401.10774"]  # the null entry (unknown paper) is dropped
    medusa = found["2401.10774"]
    assert medusa.citations == 312 and medusa.influential_citations == 41
    assert medusa.tldr.startswith("Medusa adds extra decoding heads")


def test_s2_search_and_to_item():
    client = mock_client(lambda r: httpx.Response(200, json=fixture_json("s2_search.json")))
    papers = semantic_scholar.search("speculative decoding", client=client)
    assert len(papers) == 2
    item = papers[0].to_item()
    assert item.id == "2211.17192" and item.url == "https://arxiv.org/abs/2211.17192"
    assert item.tldr.startswith("A small draft model")
    assert papers[1].to_item() is None  # nothing to read: no abstract, no TLDR


def test_s2_sends_api_key_when_set():
    assert semantic_scholar.make_client("k123").headers["x-api-key"] == "k123"
    assert "x-api-key" not in semantic_scholar.make_client("").headers


# --- Lab blogs ---


def test_lab_rss_strips_html_and_skips_old_posts():
    since = date.today() - timedelta(days=14)
    items = lab_blogs.parse_feed(lab_xml("lab_feed.xml"), "Example Lab", since)
    assert [i.title for i in items] == ["Introducing Model X"]
    post = items[0]
    assert post.kind == "post" and post.source == "lab:Example Lab"
    assert post.abstract == "Model X is our newest reasoning model & it runs 2x faster."
    assert post.id == lab_blogs.post_id("https://lab.example.com/blog/model-x")


def test_lab_atom_prefers_full_content():
    items = lab_blogs.parse_feed(lab_xml("lab_feed_atom.xml"), "Atom Lab")
    assert items[0].abstract == "Overview The full post body about sparse attention."


def test_lab_fetch_all_reports_a_broken_feed_and_keeps_going():
    def handler(request):
        if "broken" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, text=lab_xml("lab_feed.xml"))

    feeds = {"Good": "https://good.test/feed", "Broken": "https://broken.test/feed"}
    got = lab_blogs.fetch_all(feeds, client=mock_client(handler))
    assert len(got.items) == 1
    assert list(got.errors) == ["Broken"] and "404" in got.errors["Broken"]


# --- Hacker News ---


def test_hn_parse_finds_arxiv_links_and_ask_hn_text():
    stories = hackernews.parse_hits(fixture_json("hn_search.json"))
    paper, blog, ask = stories
    assert paper.arxiv_id == "2610.31819"  # version suffix ignored
    assert blog.arxiv_id is None and blog.points == 130
    assert ask.url == "https://news.ycombinator.com/item?id=45000003"  # no link: the thread
    assert ask.text == "I'm picking between nomic and qwen3-embedding ."


def test_hn_fetch_filters_by_time_and_points_and_dedupes():
    params = []

    def handler(request):
        params.append(dict(request.url.params))
        return httpx.Response(200, json=fixture_json("hn_search.json"))

    stories = hackernews.fetch_stories(
        ["LLM", "arxiv"], days=2, min_points=50, client=mock_client(handler), delay_s=0
    )
    assert [p["query"] for p in params] == ["LLM", "arxiv"]
    assert params[0]["tags"] == "story" and "points>=50" in params[0]["numericFilters"]
    assert [s.points for s in stories] == [240, 130, 75]  # same hits twice, kept once
