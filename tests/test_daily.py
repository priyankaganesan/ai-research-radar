"""The daily graph end to end on a temp store, with a scripted model and no network."""

from datetime import date

from conftest import fake_chat, make_item

from radar import services
from radar.rag.index import index_items
from radar.schema import Item
from radar.store import digests as digest_store
from radar.store import signals as signal_store

TODAY = date(2026, 10, 22)


def _fill(ctx):
    papers = [
        make_item(f"2410.1000{i}", f"Agent paper {i}", f"LLM agents and tool use, idea {i}.")
        for i in range(8)
    ]
    papers.append(make_item("2410.20000", "Old paper", "From long ago.", "2026-01-01"))
    post = Item(
        id="post:abc",
        source="lab:OpenAI",
        kind="post",
        title="We launched a new model",
        abstract="A new reasoning model.",
        published=date(2026, 10, 21),
        url="https://openai.com/x",
    )
    index_items(ctx.conn, ctx.embeddings, ctx.vectors, papers + [post])
    with ctx.conn:
        signal_store.record(ctx.conn, "2410.10007", TODAY, hf_upvotes=150)
        signal_store.record(ctx.conn, "2410.10001", date(2026, 10, 20), citations=2)
        signal_store.record(ctx.conn, "2410.10001", TODAY, citations=12)


def test_daily_writes_ranked_digest(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx.settings, "digest_dir", tmp_path / "digests")
    _fill(ctx)
    llm = fake_chat(*[f"Summary {i}." for i in range(8)])
    result = services.daily(ctx, on=TODAY, collect=False, llm=llm)

    assert len(result.picks) == 8
    ids = [p["item_id"] for p in result.picks]
    assert "2410.20000" not in ids  # outside the 7-day window
    slots = {p["item_id"]: p["slot"] for p in result.picks}
    assert slots["2410.10007"] == "early_buzz"
    assert slots["post:abc"] == "lab_launch"
    # every summary ends up citing its own item
    assert all(f"[{p['item_id']}]" in p["summary"] for p in result.picks)

    text = (tmp_path / "digests" / "2026-10-22.md").read_text()
    assert "Early buzz" in text and "Lab launch" in text
    rows = digest_store.get_digest(ctx.conn, "2026-10-22")
    assert [r["rank"] for r in rows] == list(range(1, 9))
    assert set(result.timings) >= {"collect", "score", "summarize", "render"}


def test_items_in_an_earlier_digest_are_not_repeated(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx.settings, "digest_dir", tmp_path / "digests")
    _fill(ctx)
    services.daily(ctx, on=date(2026, 10, 21), collect=False, llm=fake_chat(*["s."] * 8))
    first = {r["item_id"] for r in digest_store.get_digest(ctx.conn, "2026-10-21")}
    result = services.daily(ctx, on=TODAY, collect=False, llm=fake_chat(*["s."] * 8))
    assert not first & {p["item_id"] for p in result.picks}


def test_citation_velocity_uses_change_between_readings(ctx):
    from radar.graphs.daily import citation_velocity

    _fill(ctx)
    assert citation_velocity(ctx, "2410.10001", age_days=2) == 5.0  # 10 more in 2 days
    assert citation_velocity(ctx, "2410.10000", age_days=2) == 0.0
