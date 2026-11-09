"""M2.5 agents, the supervisor and the Q&A verifier, with scripted models (no Ollama)."""

import re
from datetime import date
from pathlib import Path

from conftest import FakeChat, fake_chat, tool_call
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from radar import services
from radar.graphs.agents import parse_json
from radar.graphs.agents.critic import build_critic
from radar.graphs.agents.team import build_team, new_stats
from radar.graphs.agents.verifier import build_verifier, remove_claims, split_claims
from radar.graphs.agents.writer import build_writer
from radar.graphs.qa import build_qa_graph
from radar.tools import make_tools

ITEM = {
    "item_id": "2401.10774",
    "title": "Medusa: multiple decoding heads",
    "abstract": "Extra decoding heads predict several future tokens to speed up LLM inference.",
    "tldr": None,
    "kind": "paper",
    "slot": "main",
    "reason": "Top score",
}


def item(n: int) -> dict:
    return {**ITEM, "item_id": f"2410.0000{n}", "title": f"Paper {n}", "reason": f"rank {n}"}


class RoleFake(BaseChatModel):
    """Answers by role, read from the prompt, and records every prompt it got.

    critic: drops items whose title is in `drop`; writer: one cited sentence;
    verifier: marks claims containing "UNSUPPORTED" as unsupported.
    """

    drop: set = set()
    writer_extra: str = ""
    prompts: list = []

    @property
    def _llm_type(self) -> str:
        return "role-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        prompt = str(messages[-1].content)
        self.prompts.append(prompt)
        if "You are the Critic" in prompt:
            title = re.search(r"Title: (.*)", prompt).group(1)
            keep = title not in self.drop
            reply = f'{{"keep": {str(keep).lower()}, "concerns": "{"" if keep else "weak"}"}}'
        elif "You are the Writer" in prompt:
            item_id = re.search(r"citation \[([^\]]+)\]", prompt).group(1)
            reply = (
                f"The paper proposes a method for faster decoding [{item_id}].{self.writer_extra}"
            )
        elif "You are the Verifier" in prompt:
            claims = re.findall(r"^(\d+)\. (.*)$", prompt, re.MULTILINE)
            verdicts = [
                f'{{"n": {n}, "supported": {str("UNSUPPORTED" not in c).lower()}, "why": "x"}}'
                for n, c in claims
            ]
            reply = "[" + ", ".join(verdicts) + "]"
        else:
            reply = "Summary."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(reply))])


def test_parse_json_finds_json_in_prose():
    assert parse_json('Sure:\n```json\n{"keep": false}\n```', {}) == {"keep": False}
    assert parse_json('[{"n": 1, "supported": true}] done', []) == [{"n": 1, "supported": True}]
    assert parse_json("no json", {"keep": True}) == {"keep": True}


def test_critic_keeps_unless_it_clearly_drops():
    drop = build_critic(fake_chat('{"keep": false, "concerns": "vague, no evaluation"}'))
    out = drop.invoke({"item": ITEM, "topics": "inference"})
    assert out["keep"] is False and "vague" in out["concerns"]
    unreadable = build_critic(fake_chat("I think it's fine?"))
    assert unreadable.invoke({"item": ITEM, "topics": "x"})["keep"] is True


def test_writer_adds_citation_and_uses_feedback():
    llm = RoleFake(prompts=[])
    writer = build_writer(llm)
    out = writer.invoke({"item": ITEM, "problems": [{"claim": "It is 3x faster.", "why": "no"}]})
    assert "[2401.10774]" in out["draft"]
    assert "It is 3x faster." in llm.prompts[-1]  # the Verifier's objection reached the Writer
    bare = build_writer(fake_chat("Medusa adds heads."))
    assert bare.invoke({"item": ITEM})["draft"].endswith("[2401.10774]")


def test_split_claims():
    text = "Medusa adds heads [a]. It is 2x faster [a].\n- Uses a tree of candidates [b]."
    claims = split_claims(text)
    assert [c["claim"] for c in claims] == [
        "Medusa adds heads [a].",
        "It is 2x faster [a].",
        "Uses a tree of candidates [b].",
    ]
    assert claims[2]["cites"] == ["b"]


def test_verifier_marks_unsupported_and_unknown():
    passages = [{"item_id": "a", "title": "t", "text": "Medusa adds heads."}]
    v = build_verifier(fake_chat('[{"n": 1, "supported": true}, {"n": 2, "supported": false}]'))
    out = v.invoke(
        {
            "text": "Medusa adds heads [a]. It is 10x faster [a]. Third claim here [a].",
            "passages": passages,
        }
    )
    assert [r["supported"] for r in out["results"]] == [True, False, None]
    bad = [r for r in out["results"] if r["supported"] is False]
    assert remove_claims("Medusa adds heads [a]. It is 10x faster [a].", bad) == (
        "Medusa adds heads [a]."
    )


def _run_team(llm, picks, reserve):
    team = build_team(llm)
    return team.invoke(
        {
            "topics": "t",
            "queue": picks,
            "reserve": reserve,
            "stats": new_stats(),
            "done": [],
            "notes": [],
            "log": [],
        },
        config={"recursion_limit": 200},
    )


def test_supervisor_replaces_dropped_pick():
    llm = RoleFake(drop={"Paper 1"}, prompts=[])
    out = _run_team(llm, [item(1), item(2)], [item(3)])
    ids = [d["item_id"] for d in out["done"]]
    assert ids == ["2410.00003", "2410.00002"]  # the replacement keeps the dropped one's place
    assert "replaced 2410.00001" in out["done"][0]["reason"]
    assert out["stats"]["critic_drops"] == 1
    assert any("Critic dropped 2410.00001" in n for n in out["notes"])


def test_supervisor_retries_then_cuts_unsupported_claims():
    llm = RoleFake(writer_extra=" It is UNSUPPORTED 10x faster [2410.00001].", prompts=[])
    out = _run_team(llm, [item(1)], [])
    done = out["done"][0]
    assert out["stats"]["rewrites"] == 2  # the retry limit
    assert sum("You are the Writer" in p for p in llm.prompts) == 3
    assert "UNSUPPORTED" not in done["summary"]
    assert "faster decoding [2410.00001]" in done["summary"]
    assert out["stats"]["claims_cut"] == 1


def test_critic_cannot_empty_the_digest():
    llm = RoleFake(drop={f"Paper {n}" for n in range(1, 9)}, prompts=[])
    out = _run_team(llm, [item(1), item(2)], [item(n) for n in range(3, 9)])
    assert len(out["done"]) == 2
    assert out["stats"]["critic_drops"] == 3  # MAX_DROPS, then items are kept with concerns
    assert out["done"][-1]["concerns"] == "weak"


def test_daily_multi_mode_end_to_end(ctx, tmp_path, monkeypatch):
    from conftest import make_item

    from radar.rag.index import index_items

    monkeypatch.setattr(ctx.settings, "digest_dir", tmp_path / "digests")
    papers = [make_item(f"2410.1000{i}", f"Agent paper {i}", "LLM agents.") for i in range(9)]
    index_items(ctx.conn, ctx.embeddings, ctx.vectors, papers)
    llm = RoleFake(drop={"Agent paper 0"}, prompts=[])
    result = services.daily(ctx, on=date(2026, 10, 22), collect=False, mode="multi", llm=llm)
    assert len(result.picks) == 8
    assert "2410.10000" not in [p["item_id"] for p in result.picks]
    assert result.stats["critic_drops"] == 1
    text = (tmp_path / "digests" / "2026-10-22.md").read_text()
    assert "Critic dropped 2410.10000" in text
    rows = ctx.conn.execute("SELECT COUNT(*) FROM summaries WHERE mode = 'multi'").fetchone()
    assert rows[0] == 8


def test_qa_verifier_removes_unsupported_sentence(filled_ctx):
    llm = fake_chat(
        tool_call("store_search", query="Medusa"),
        "Medusa adds decoding heads [2401.10774]. It beats every other method by 10x [2401.10774].",
        '[{"n": 1, "supported": true, "why": "stated"}, {"n": 2, "supported": false, '
        '"why": "no numbers in the abstract"}]',
    )
    graph = build_qa_graph(
        llm,
        fake_chat('{"sufficient": true, "missing": ""}'),
        make_tools(filled_ctx),
        checkpointer=InMemorySaver(),
        verifier=llm,
    )
    state = graph.invoke(
        {"messages": [HumanMessage("What is Medusa?")]}, {"configurable": {"thread_id": "v"}}
    )
    assert "10x" not in state["answer"]
    assert "Removed 1 claim" in state["answer"]
    assert state["trace"][-1]["step"] == "verify"
    assert state["messages"][-1].content == state["answer"]  # history holds the checked answer
    assert len(state["messages"]) == 2


def test_qa_verifier_abstains_when_nothing_is_supported(filled_ctx):
    llm = fake_chat(
        tool_call("store_search", query="Medusa"),
        "Medusa was invented in 1990 [2401.10774].",
        '[{"n": 1, "supported": false, "why": "date not in passage"}]',
    )
    graph = build_qa_graph(
        llm,
        FakeChat(messages=iter([AIMessage('{"sufficient": true, "missing": ""}')])),
        make_tools(filled_ctx),
        checkpointer=InMemorySaver(),
        verifier=llm,
    )
    state = graph.invoke(
        {"messages": [HumanMessage("When was Medusa made?")]},
        {"configurable": {"thread_id": "w"}},
    )
    assert state["cited"] == []
    assert state["answer"].startswith("I couldn't find enough evidence")


def test_compare_agents_report(ctx, tmp_path):
    from conftest import make_item

    from radar import evals_agents
    from radar.rag.index import index_items

    papers = [make_item(f"2410.1000{i}", f"Agent paper {i}", "LLM agents.") for i in range(9)]
    index_items(ctx.conn, ctx.embeddings, ctx.vectors, papers)
    llm = RoleFake(drop={"Agent paper 0"}, prompts=[])
    report = evals_agents.compare(ctx, date(2026, 10, 22), llm, out_dir=tmp_path)
    assert report["digest"]["single"]["llm_calls"] == 8
    assert report["digest"]["multi"]["llm_calls"] > 8
    assert report["digest"]["multi"]["support_rate"] == 1.0
    assert "| Claim support" in report["markdown"]
    assert Path(report["path"]).exists()
