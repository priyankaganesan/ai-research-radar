"""The M3 ingest paths end to end: fixture HTTP in, rows in the store out."""

import json
from datetime import date

import httpx
from conftest import FIXTURES, SAMPLE_ITEMS
from test_sources import lab_xml
from typer.testing import CliRunner

from radar import services
from radar.cli import app
from radar.rag.index import index_items
from radar.store import signals


def json_client(name: str) -> httpx.Client:
    data = json.loads((FIXTURES / name).read_text())
    return httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=data)))


def test_ingest_hf_adds_papers_and_upvotes(ctx):
    report = services.ingest_hf(ctx, days=1, client=json_client("hf_daily.json"))
    assert (report.fetched, report.added, report.signals) == (2, 2, 2)
    assert signals.latest(ctx.conn, "2610.31819")["hf_upvotes"] == 87
    hits = services.search(ctx, "speculative diffusion tradeoffs")
    assert hits[0].item_id == "2610.31819"  # searchable right away


def test_ingest_hf_merges_with_a_paper_already_from_arxiv(filled_ctx):
    services.ingest_hf(filled_ctx, days=1, client=json_client("hf_daily.json"))
    medusa = services.get_item(filled_ctx, "2401.10774")
    assert medusa.source == "arxiv"  # one item, not a duplicate
    assert signals.latest(filled_ctx.conn, "2401.10774")["hf_upvotes"] == 12


def test_ingest_labs_adds_posts_and_reports_broken_feeds(ctx, monkeypatch):
    monkeypatch.setattr(
        ctx.settings, "lab_feeds", {"Good": "https://good.test", "Gone": "https://gone.test"}
    )

    def handler(request):
        if "gone" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, text=lab_xml("lab_feed.xml"))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = services.ingest_labs(ctx, days=14, client=client)
    assert (report.fetched, report.added) == (1, 1)
    assert "Gone" in report.errors
    hit = services.search(ctx, "Model X reasoning")[0]
    assert hit.item_id.startswith("post:") and hit.url == "https://lab.example.com/blog/model-x"


def test_ingest_hn_links_papers_and_stores_other_stories_as_posts(ctx):
    s2_calls = []

    def s2(request):
        s2_calls.append(json.loads(request.content)["ids"])
        paper = json.loads((FIXTURES / "s2_search.json").read_text())["data"][0]
        paper = {**paper, "externalIds": {"ArXiv": "2610.31819"}}
        return httpx.Response(200, json=[paper])

    report = services.ingest_hn(
        ctx,
        days=2,
        min_points=50,
        client=json_client("hn_search.json"),
        s2_client=httpx.Client(transport=httpx.MockTransport(s2)),
    )
    # The arXiv-linked story's paper wasn't in the store, so it came from Semantic Scholar.
    assert s2_calls == [["ARXIV:2610.31819"]]
    paper = services.get_item(ctx, "2610.31819")
    assert paper.source == "s2" and paper.tldr.startswith("A small draft model")
    assert signals.latest(ctx.conn, "2610.31819")["hn_points"] == 240
    # The other two stories are posts with their HN numbers.
    post = services.get_item(ctx, "hn:45000002")
    assert post.kind == "post" and "130 points" in post.abstract
    assert report.added == 3 and report.signals == 3


def test_enrich_records_citations_and_tldr(filled_ctx, monkeypatch):
    # Make the sample papers recent enough to be enriched.
    filled_ctx.conn.execute("UPDATE items SET published = ?", (date.today().isoformat(),))
    report = services.enrich(filled_ctx, client=json_client("s2_batch.json"))
    assert report.fetched == len(SAMPLE_ITEMS)
    assert report.signals == 1  # the fixture only knows Medusa
    found = services.show(filled_ctx, "2401.10774")
    item, sig = found
    assert item.tldr.startswith("Medusa adds extra decoding heads")
    assert sig["citations"] == 312 and sig["influential_citations"] == 41


def test_semantic_scholar_tool_indexes_results(ctx, monkeypatch):
    from radar.sources import semantic_scholar
    from radar.tools import make_tools

    monkeypatch.setattr(
        semantic_scholar, "make_client", lambda key="": json_client("s2_search.json")
    )
    tool = make_tools(ctx)["semantic_scholar_lookup"]
    out = tool.invoke({"query": "speculative decoding"})
    assert [p["item_id"] for p in out] == ["2211.17192"]
    assert "Cited by 1200." in out[0]["text"] and "TLDR:" in out[0]["text"]
    assert services.get_item(ctx, "2211.17192") is not None  # saved for next time
    assert signals.latest(ctx.conn, "2211.17192")["citations"] == 1200


def test_cli_rejects_unknown_source():
    result = CliRunner().invoke(app, ["ingest", "reddit"])
    assert result.exit_code == 1 and "Choose from" in result.output


def test_cli_show(monkeypatch):
    from conftest import FakeEmbeddings

    from radar.config import get_settings
    from radar.context import build_context

    monkeypatch.setattr("radar.models.get_embeddings", lambda settings=None: FakeEmbeddings())
    ctx = build_context(get_settings())
    index_items(ctx.conn, ctx.embeddings, ctx.vectors, SAMPLE_ITEMS[:1])
    signals.record(ctx.conn, "2401.10774", date.today(), hf_upvotes=12)
    ctx.conn.commit()
    result = CliRunner().invoke(app, ["show", "2401.10774"])
    assert result.exit_code == 0, result.output
    assert "hf_upvotes 12" in result.output and "Medusa" in result.output
