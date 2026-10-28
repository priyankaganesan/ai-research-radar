from datetime import date, timedelta

import httpx
import pytest
from conftest import FIXTURES

from radar.sources import arxiv


def _feed() -> str:
    recent = (date.today() - timedelta(days=1)).isoformat()
    return (FIXTURES / "arxiv_feed.xml").read_text().replace("{RECENT}", recent)


def test_parse_feed_normalizes_fields():
    items = arxiv.parse_feed(_feed())
    medusa = items[0]
    assert medusa.id == "2401.10774"  # version suffix removed
    assert medusa.title.startswith("Medusa: Simple LLM") and "\n" not in medusa.title
    assert medusa.abstract.startswith("Large language models") and "  " not in medusa.abstract
    assert medusa.authors == ["Tianle Cai", "Yuhong Li"]
    assert medusa.categories == ["cs.CL", "cs.LG"]
    assert medusa.url == "https://arxiv.org/abs/2401.10774"
    assert medusa.pdf_url.endswith("2401.10774v3")


def test_fetch_recent_stops_at_old_papers():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params)
        return httpx.Response(200, text=_feed())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    items = arxiv.fetch_recent(days=7, categories=["cs.CL"], client=client, delay_s=0)
    assert [i.id for i in items] == ["2401.10774"]  # the 2020 paper is outside the window
    assert len(calls) == 1  # stopped paging once it saw an old paper
    assert calls[0]["search_query"] == "cat:cs.CL"
    assert calls[0]["sortBy"] == "submittedDate"


def test_search_builds_keyword_query():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, text=_feed())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    arxiv.search("Medusa decoding-heads", client=client)
    assert seen["search_query"] == "all:Medusa AND all:decoding-heads"
    assert arxiv.search("!!!", client=client) == []


def test_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: None)
    responses = iter([httpx.Response(429), httpx.Response(503), httpx.Response(200, text=_feed())])
    client = httpx.Client(transport=httpx.MockTransport(lambda req: next(responses)))
    assert arxiv.search("Medusa", client=client)[0].id == "2401.10774"


def test_gives_up_with_clear_error(monkeypatch):
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: None)
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(429)))
    with pytest.raises(arxiv.ArxivUnavailable, match="status 429"):
        arxiv.search("Medusa", client=client)


def test_client_identifies_itself():
    assert "ai-research-radar" in arxiv.make_client().headers["User-Agent"]


def test_parse_rss_skips_revisions():
    items = arxiv.parse_rss((FIXTURES / "arxiv_rss.xml").read_text())
    assert [i.id for i in items] == ["2401.10774", "2410.00002"]  # "replace" entry skipped
    medusa = items[0]
    assert medusa.abstract.startswith("Large language models")
    assert "Announce Type" not in medusa.abstract
    assert medusa.authors == ["Tianle Cai", "Yuhong Li"]
    assert medusa.published.isoformat() == "2026-10-26"


def test_fetch_latest_rss_one_request_for_all_categories():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, text=(FIXTURES / "arxiv_rss.xml").read_text())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    arxiv.fetch_latest_rss(["cs.AI", "cs.CL"], client=client)
    assert urls == ["https://rss.arxiv.org/rss/cs.AI+cs.CL"]


def test_timeouts_are_retried(monkeypatch):
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, text=_feed())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert arxiv.search("Medusa", client=client)[0].id == "2401.10774"


def test_persistent_timeout_becomes_unavailable(monkeypatch):
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: None)

    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(arxiv.ArxivUnavailable, match="ReadTimeout"):
        arxiv.search("Medusa", client=client)
