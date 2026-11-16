"""M5: full-text parsing, section chunking, storage, the repeated-question trigger and the
agent tool. No network: HTTP goes through httpx.MockTransport, PDFs are made on the fly."""

import httpx
import pytest
from conftest import FIXTURES, fake_chat, make_item, tool_call
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from radar import services
from radar.graphs.qa import build_qa_graph
from radar.rag.chunking import approx_tokens, section_chunks, split_sentences
from radar.rag.retrieve import hybrid_search
from radar.schema import Item
from radar.sources import full_text as ft
from radar.sources.full_text import Section
from radar.store import full_texts, memory
from radar.tools import make_tools

HTML = (FIXTURES / "arxiv_full.html").read_text()


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def make_pdf(lines: list[str]) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=10)
        y += 14
    data = doc.tobytes()
    doc.close()
    return data


# ---- parsing ----------------------------------------------------------------------------------


def test_arxiv_html_splits_into_sections():
    sections = ft.parse_html(HTML)
    assert [s.heading for s in sections] == [
        "1 Introduction",
        "2 Method",
        "2 Method > 2.1 Training the heads",
        "3 Results",
        "Appendix A Hyperparameters",
    ]
    by = {s.heading: s.text for s in sections}
    assert "$K$ heads" in by["2 Method"]  # formula as LaTeX, not MathML symbols
    assert "Table 1: speedups." not in by["2 Method > 2.1 Training the heads"]  # figures skipped
    assert "junk cell" not in " ".join(by.values())
    heads = by["2 Method > 2.1 Training the heads"]
    assert "$L=\\sum_{k}\\lambda_{k}L_{k}$" in heads  # display equation kept (it's a <table>)
    assert "$k$, the head index" in heads  # no stray space before the comma
    everything = " ".join(by.values())
    for gone in ("Back to arXiv", "We thank", "Leviathan, Y. Fast", "LaTeXML", "var x", "GitHub"):
        assert gone not in everything


def test_post_keeps_text_before_the_first_heading():
    html = (
        "<html><body><h1>Launch</h1><p>We released a model today.</p><h2>Details</h2>"
        "<p>It has 7B parameters.</p></body></html>"
    )
    sections = ft.parse_html(html, untitled="Post")
    assert [(s.heading, s.text) for s in sections] == [
        ("Post", "Launch\n\nWe released a model today."),
        ("Details", "It has 7B parameters."),
    ]


PDF_TEXT = """Medusa: multiple decoding heads
Abstract
Extra heads predict tokens.
1 Introduction
Decoding is slow. We fix it with ex-
tra heads.
2 Method
We add heads.
2 GPUs were used for training
2.1 Training the heads
Trained for one epoch.
7 Unexpected Jump
3 Results
A 2.3x speedup.
References
A. Smith and B. Jones. Fast decoding. 2023
B Not An Appendix Yet
A Hyperparameters
Learning rate 1e-3."""


def test_pdf_headings_are_guessed_with_numbering_rules():
    sections = ft.split_pdf_text(PDF_TEXT)
    assert [s.heading for s in sections] == [
        "1 Introduction",
        "2 Method",
        "2.1 Training the heads",
        "3 Results",
        "A Hyperparameters",
    ]
    by = {s.heading: s.text for s in sections}
    assert by["1 Introduction"] == "Decoding is slow. We fix it with extra heads."  # dehyphenated
    assert "2 GPUs were used" in by["2 Method"]  # out of sequence: body text, not a heading
    assert "7 Unexpected Jump" in by["2.1 Training the heads"]


def test_pdf_text_reads_a_real_pdf():
    text = ft.pdf_text(make_pdf(["1 Introduction", "Some words here."]))
    assert "1 Introduction" in text and "Some words here." in text


# ---- chunking ---------------------------------------------------------------------------------


def test_section_chunks_overlap_and_stay_inside_sections():
    sentences = [f"Sentence number {i} says something about decoding heads." for i in range(120)]
    sections = [Section("2 Method", " ".join(sentences)), Section("3 Results", "Short one.")]
    chunks = section_chunks("x1", "Medusa", sections, size=100, overlap=20)
    method = [c for c in chunks if c.section == "2 Method"]
    assert len(method) > 5
    assert all(c.text.startswith("Medusa\nSection: 2 Method\n\n") for c in method)
    for c in method:
        assert approx_tokens(c.text.split("\n\n", 1)[1]) <= 100 + 15
    for a, b in zip(method, method[1:], strict=False):  # b starts with a's last sentences
        a_sents = split_sentences(a.text.split("\n\n", 1)[1])
        b_sents = split_sentences(b.text.split("\n\n", 1)[1])
        carried = [s for s in b_sents if s in a_sents]
        assert carried and carried == a_sents[-len(carried) :] == b_sents[: len(carried)]
        assert sum(map(approx_tokens, carried)) <= 20
    assert [c.section for c in chunks][-1] == "3 Results"  # never merged across sections
    assert all(c.kind == "section" for c in chunks)


def test_tiny_tail_is_merged_and_long_sentences_are_cut():
    text = "word " * 400 + ". End."
    chunks = section_chunks("x1", "T", [Section("S", text)], size=100, overlap=10)
    assert len(chunks) >= 4
    assert chunks[-1].text.rstrip().endswith("End.")


# ---- fetching ---------------------------------------------------------------------------------

MEDUSA = make_item(
    "2401.10774",
    "Medusa: multiple decoding heads",
    "Extra decoding heads predict several future tokens to speed up LLM inference.",
)


def test_fetch_uses_arxiv_html_first():
    seen = []

    def handler(req):
        seen.append(str(req.url))
        return httpx.Response(200, text=HTML)

    got = ft.fetch(MEDUSA, mock_client(handler))
    assert got.source == "arxiv_html" and len(got.sections) == 5
    assert seen == ["https://arxiv.org/html/2401.10774"]


def test_fetch_falls_back_to_the_pdf_when_there_is_no_html():
    pdf = make_pdf(["1 Introduction", "Slow decoding.", "2 Method", "Extra heads."])

    def handler(req):
        if "/html/" in str(req.url):
            return httpx.Response(404)
        return httpx.Response(200, content=pdf)

    got = ft.fetch(MEDUSA, mock_client(handler))
    assert got.source == "arxiv_pdf"
    assert [s.heading for s in got.sections] == ["1 Introduction", "2 Method"]


def test_fetch_refuses_items_it_cannot_read():
    hn = Item(
        id="hn:1",
        source="hn",
        title="t",
        abstract="a",
        published=MEDUSA.published,
        url="https://ieeexplore.ieee.org/document/1",
        kind="post",
    )
    with pytest.raises(ft.FullTextUnavailable):
        ft.fetch(hn, mock_client(lambda r: httpx.Response(200, text=HTML)))


# ---- storing and searching --------------------------------------------------------------------


def html_fetch(monkeypatch, calls=None):
    def fake(item, client=None):
        if calls is not None:
            calls.append(item.id)
        return ft.FullText(item.id, "arxiv_html", "u", ft.parse_html(HTML))

    monkeypatch.setattr(ft, "fetch", fake)


def section_count(conn, item_id):
    ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM chunks WHERE item_id = ? AND kind = 'section'", (item_id,)
        )
    ]
    marks = ",".join("?" * len(ids)) or "NULL"
    fts = conn.execute(f"SELECT COUNT(*) FROM chunks_fts WHERE rowid IN ({marks})", ids)
    vec = sum(
        conn.execute("SELECT COUNT(*) FROM chunks_vec WHERE chunk_id = ?", (i,)).fetchone()[0]
        for i in ids
    )
    return len(ids), fts.fetchone()[0], vec


def test_full_text_is_indexed_cached_and_replaced(filled_ctx, monkeypatch):
    calls = []
    html_fetch(monkeypatch, calls)
    r = services.full_text(filled_ctx, "2401.10774")
    assert (r.status, r.source, r.sections) == ("ok", "arxiv_html", 5)
    n = r.chunks
    assert section_count(filled_ctx.conn, "2401.10774") == (n, n, n)

    again = services.full_text(filled_ctx, "2401.10774")
    assert again.status == "cached" and calls == ["2401.10774"]  # no second download

    services.full_text(filled_ctx, "2401.10774", force=True)
    assert section_count(filled_ctx.conn, "2401.10774") == (n, n, n)  # replaced, not doubled
    abstract = filled_ctx.conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE item_id = '2401.10774' AND kind = 'abstract'"
    ).fetchone()[0]
    assert abstract == 1


def test_failed_fetch_is_remembered(filled_ctx, monkeypatch):
    calls = []

    def broken(item, client=None):
        calls.append(item.id)
        raise ft.FullTextUnavailable("no HTML or PDF")

    monkeypatch.setattr(ft, "fetch", broken)
    assert services.full_text(filled_ctx, "2401.10774").status == "failed"
    assert services.full_text(filled_ctx, "2401.10774").error == "no HTML or PDF"
    assert calls == ["2401.10774"]  # not retried until FULL_TEXT_RETRY_DAYS pass


def test_search_inside_one_paper(filled_ctx, monkeypatch):
    html_fetch(monkeypatch)
    services.full_text(filled_ctx, "2401.10774")
    best = services.search_full_text(filled_ctx, "2401.10774", "learning rate for the heads")
    assert best[0].section == "Appendix A Hyperparameters"
    assert {p.item_id for p in best} == {"2401.10774"}
    # Store-wide search can now also land on a section, and says which.
    hits = hybrid_search(
        filled_ctx.conn, filled_ctx.embeddings, filled_ctx.vectors, "frozen backbone ShareGPT epoch"
    )
    assert hits[0].item_id == "2401.10774" and hits[0].section


def test_repeated_questions_trigger_a_fetch(filled_ctx, monkeypatch):
    calls = []
    html_fetch(monkeypatch, calls)
    with filled_ctx.conn:
        memory.log_question(filled_ctx.conn, "What is Medusa?", True, ["2401.10774"])
    assert services.fetch_repeated(filled_ctx, ["2401.10774"]) == []  # cited once: not yet
    with filled_ctx.conn:
        memory.log_question(filled_ctx.conn, "How are heads trained?", True, ["2401.10774"])
    out = services.fetch_repeated(filled_ctx, ["2401.10774"])
    assert [(o["item_id"], o["status"], o["times_cited"]) for o in out] == [("2401.10774", "ok", 2)]
    assert full_texts.get(filled_ctx.conn, "2401.10774")["reason"] == "cited by 2 questions"
    assert services.fetch_repeated(filled_ctx, ["2401.10774"]) == []  # only once
    assert calls == ["2401.10774"]


def test_agent_ask_reports_the_fetch(filled_ctx, monkeypatch):
    html_fetch(monkeypatch)
    for q in ("What is Medusa?", "Medusa again?"):
        agent = services.Agent(
            filled_ctx,
            llm=fake_chat(tool_call("store_search", query="Medusa"), "Heads [2401.10774]."),
            judge=fake_chat('{"sufficient": true, "missing": ""}'),
            checkpointer=InMemorySaver(),
        )
        result = agent.ask(q)
    assert [f["item_id"] for f in result.full_texts] == ["2401.10774"]


def test_qa_graph_reads_sections_with_the_tool(filled_ctx, monkeypatch):
    html_fetch(monkeypatch)
    llm = fake_chat(
        tool_call("store_search", query="Medusa"),
        tool_call("fetch_full_text", item_id="2401.10774", query="how are the heads trained"),
        "The heads are trained on a frozen backbone [2401.10774].",
    )
    judge = fake_chat(
        '{"sufficient": false, "missing": "how the heads are trained"}',
        '{"sufficient": true, "missing": ""}',
    )
    graph = build_qa_graph(llm, judge, make_tools(filled_ctx), checkpointer=InMemorySaver())
    state = graph.invoke(
        {"messages": [HumanMessage("How are Medusa heads trained?")]},
        {"configurable": {"thread_id": "t"}},
    )
    medusa = [p for p in state["passages"] if p["item_id"] == "2401.10774"]
    assert len(medusa) > 1  # the abstract plus sections: same paper, different evidence
    assert any(p.get("section") for p in medusa)
    assert state["cited"] == ["2401.10774"]
