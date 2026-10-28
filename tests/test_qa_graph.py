"""Graph routing with a scripted model: no Ollama, same result every run."""

from conftest import fake_chat, tool_call
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from radar.graphs.qa import build_qa_graph, keep_valid_citations, parse_verdict
from radar.tools import make_tools

SUFFICIENT = '{"sufficient": true, "missing": ""}'
INSUFFICIENT = '{"sufficient": false, "missing": "nothing on Medusa yet"}'


def run(ctx, llm, judge, question="What is Medusa?", max_steps=3, thread="t1", saver=None):
    graph = build_qa_graph(
        llm, judge, make_tools(ctx), max_steps=max_steps, checkpointer=saver or InMemorySaver()
    )
    return graph.invoke(
        {"messages": [HumanMessage(question)]}, {"configurable": {"thread_id": thread}}
    )


def test_search_then_answer_with_citation(filled_ctx):
    llm = fake_chat(
        tool_call("store_search", query="Medusa decoding heads"),
        "Medusa adds decoding heads that predict future tokens [2401.10774].",
    )
    state = run(filled_ctx, llm, fake_chat(SUFFICIENT))
    assert state["cited"] == ["2401.10774"]
    assert "Sources:" in state["answer"]
    assert [t["step"] for t in state["trace"]] == ["reason", "observe", "evidence", "answer"]


def test_trace_shows_thinking_text(filled_ctx):
    from langchain_core.messages import AIMessage

    call = tool_call("store_search", query="Medusa")
    thinking = AIMessage(
        "",
        tool_calls=call.tool_calls,
        additional_kwargs={"reasoning_content": "The store may already have Medusa."},
    )
    llm = fake_chat(thinking, "Medusa adds decoding heads [2401.10774].")
    state = run(filled_ctx, llm, fake_chat(SUFFICIENT))
    assert state["trace"][0]["thought"] == "The store may already have Medusa."


def test_weak_evidence_searches_again(filled_ctx, monkeypatch):
    from radar.sources import arxiv

    monkeypatch.setattr(arxiv, "search", lambda q, max_results=5: [])  # no network
    llm = fake_chat(
        tool_call("store_search", query="unrelated"),
        tool_call("search_arxiv", query="Medusa"),
        tool_call("store_search", query="Medusa decoding heads"),
        "Medusa uses extra heads [2401.10774].",
    )
    judge = fake_chat(INSUFFICIENT, INSUFFICIENT, SUFFICIENT)
    state = run(filled_ctx, llm, judge, max_steps=5)
    assert state["steps"] == 3
    assert state["cited"] == ["2401.10774"]


def test_stopping_rule_abstains(filled_ctx):
    llm = fake_chat(
        tool_call("store_search", query="a"),
        tool_call("store_search", query="b"),
    )
    judge = fake_chat(INSUFFICIENT, INSUFFICIENT)
    state = run(filled_ctx, llm, judge, max_steps=2)
    assert state["cited"] == []
    assert state["answer"].startswith("I couldn't find enough evidence")
    assert state["trace"][-1]["step"] == "abstain"


def test_fabricated_citations_are_removed(filled_ctx):
    llm = fake_chat(
        tool_call("store_search", query="Medusa"),
        "Medusa is fast [2401.10774]. Also see [9999.99999].",
    )
    state = run(filled_ctx, llm, fake_chat(SUFFICIENT))
    assert "9999.99999" not in state["answer"]
    assert state["cited"] == ["2401.10774"]


def test_answer_without_valid_citations_abstains(filled_ctx):
    llm = fake_chat(tool_call("store_search", query="Medusa"), "Medusa is great. [1234.5678]")
    state = run(filled_ctx, llm, fake_chat(SUFFICIENT))
    assert state["cited"] == []
    assert state["trace"][-1]["step"] == "abstain"


def test_model_that_skips_tools_still_searches(filled_ctx):
    llm = fake_chat("I think I know this already.", "Medusa uses heads [2401.10774].")
    state = run(filled_ctx, llm, fake_chat(SUFFICIENT))
    assert state["trace"][0]["tool"] == "store_search"  # fallback kicked in
    assert state["cited"] == ["2401.10774"]


def test_follow_up_is_rewritten_using_history(filled_ctx):
    saver = InMemorySaver()
    run(
        filled_ctx,
        fake_chat(tool_call("store_search", query="Medusa"), "Heads [2401.10774]."),
        fake_chat(SUFFICIENT),
        saver=saver,
    )
    llm = fake_chat(
        "How much faster is Medusa than normal decoding?",  # rewrite_query output
        tool_call("store_search", query="Medusa speedup"),
        "About 2x [2401.10774].",
    )
    state = run(
        filled_ctx, llm, fake_chat(SUFFICIENT), question="How much faster is it?", saver=saver
    )
    assert state["question"] == "How much faster is Medusa than normal decoding?"
    assert len(state["messages"]) == 4  # two questions, two answers


def test_parse_verdict_tolerates_noise():
    assert parse_verdict('Sure! {"sufficient": true, "missing": ""} done')["sufficient"] is True
    assert parse_verdict("no json here")["sufficient"] is False


def test_keep_valid_citations_handles_lists():
    passages = [{"item_id": "a"}, {"item_id": "b"}]
    text, cited = keep_valid_citations("x [a, z] y [b]", passages)
    assert text == "x [a] y [b]" and cited == ["a", "b"]
