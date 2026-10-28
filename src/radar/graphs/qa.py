"""The Q&A graph: a ReAct loop with an explicit evidence check and stopping rule.

    START -> prepare -> rewrite_query -> reason -> tools -> evidence_check
                                           ^                    |
                                           +-- insufficient ----+-- sufficient --> answer -> END
                                               (steps < max)    +-- out of steps -> abstain -> END

`messages` holds only the conversation (questions and final answers), so follow-ups can be
resolved. Everything about the current question (passages, steps, trace) is reset by `prepare`.
"""

import json
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from radar.graphs import prompts


class QAState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str  # standalone version of the latest question
    passages: list[dict]  # evidence gathered so far (Passage as dict)
    searches_done: list[str]
    pending_calls: list[dict]  # tool calls chosen by `reason`, run by `tools`
    steps: int  # tool calls made for this question
    verdict: Literal["", "sufficient", "insufficient"]
    missing: str
    trace: list[dict]  # human-readable log of each step, for the CLI and the writeup
    answer: str
    cited: list[str]


def build_qa_graph(
    llm: BaseChatModel,
    judge: BaseChatModel,
    tools: dict[str, BaseTool],
    max_steps: int = 5,
    checkpointer: Any = None,
    reasoner: BaseChatModel | None = None,
):
    """Compile the graph. `llm` rewrites and answers; `judge` checks evidence; `reasoner`
    (default `llm`) picks tools, so thinking mode can be turned on for that step alone."""
    llm_with_tools = (reasoner or llm).bind_tools(list(tools.values()))

    def prepare(state: QAState) -> dict:
        last = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        return {
            "question": str(last.content),
            "passages": [],
            "searches_done": [],
            "pending_calls": [],
            "steps": 0,
            "verdict": "",
            "missing": "",
            "trace": [],
            "answer": "",
            "cited": [],
        }

    def rewrite_query(state: QAState) -> dict:
        history = state["messages"][:-1]
        if not history:
            return {}  # first question in the conversation: nothing to resolve
        text = "\n".join(f"{m.type}: {m.content}" for m in history[-6:])
        out = llm.invoke(prompts.REWRITE.format(history=text, question=state["question"]))
        standalone = str(out.content).strip() or state["question"]
        trace = state["trace"] + [{"step": "rewrite", "question": standalone}]
        return {"question": standalone, "trace": trace}

    def reason(state: QAState) -> dict:
        missing = f"Still missing: {state['missing']}\n" if state.get("missing") else ""
        prompt = prompts.REASON.format(
            question=state["question"],
            evidence=prompts.format_passages(state["passages"], max_chars=300),
            searches=", ".join(state["searches_done"]) or "none",
            missing=missing,
        )
        out = llm_with_tools.invoke(prompt)
        calls = [c for c in getattr(out, "tool_calls", []) if c["name"] in tools]
        if not calls:
            # A small model sometimes answers instead of calling a tool. Fall back to the
            # obvious next search so the loop still makes progress.
            name = "store_search" if not state["searches_done"] else "search_arxiv"
            calls = [{"name": name, "args": {"query": state["question"]}, "id": "fallback"}]
        call = calls[0]  # one action per step
        trace = state["trace"] + [
            {
                "step": "reason",
                "thought": _thought(out),
                "tool": call["name"],
                "args": call["args"],
            }
        ]
        return {"pending_calls": [call], "trace": trace}

    def run_tools(state: QAState) -> dict:
        passages = list(state["passages"])
        seen = {p["item_id"] for p in passages}
        searches = list(state["searches_done"])
        trace = list(state["trace"])
        steps = state["steps"]
        for call in state["pending_calls"]:
            key = f"{call['name']}({json.dumps(call['args'], sort_keys=True)})"
            steps += 1
            if key in searches:
                trace.append({"step": "observe", "result": "skipped: already searched"})
                continue
            searches.append(key)
            try:
                found = tools[call["name"]].invoke(call["args"])
            except Exception as e:  # network errors etc. become an observation, not a crash
                trace.append({"step": "observe", "result": f"tool error: {e}"})
                continue
            new = [p for p in found if p["item_id"] not in seen]
            passages.extend(new)
            seen.update(p["item_id"] for p in new)
            trace.append(
                {
                    "step": "observe",
                    "result": f"{len(found)} results, {len(new)} new",
                    "ids": [p["item_id"] for p in found],
                }
            )
        return {
            "passages": passages,
            "searches_done": searches,
            "steps": steps,
            "pending_calls": [],
            "trace": trace,
        }

    def evidence_check(state: QAState) -> dict:
        if not state["passages"]:
            verdict = {"sufficient": False, "missing": "no relevant passages found yet"}
        else:
            out = judge.invoke(
                prompts.EVIDENCE_CHECK.format(
                    question=state["question"],
                    passages=prompts.format_passages(state["passages"]),
                )
            )
            verdict = parse_verdict(str(out.content))
        result = "sufficient" if verdict["sufficient"] else "insufficient"
        trace = state["trace"] + [
            {"step": "evidence", "verdict": result, "missing": verdict["missing"]}
        ]
        return {"verdict": result, "missing": verdict["missing"], "trace": trace}

    def route(state: QAState) -> str:
        if state["verdict"] == "sufficient":
            return "answer"
        if state["steps"] < max_steps:
            return "reason"
        return "abstain"

    def answer(state: QAState) -> dict:
        out = llm.invoke(
            prompts.ANSWER.format(
                question=state["question"], passages=prompts.format_passages(state["passages"])
            )
        )
        text, cited = keep_valid_citations(str(out.content), state["passages"])
        if not cited:
            # An answer with no citations isn't grounded; treat it as unsupported.
            return abstain({**state, "missing": "the answer could not be tied to any passage"})
        by_id = {p["item_id"]: p for p in state["passages"]}
        sources = "\n".join(
            f"[{i}] {by_id[i]['title']} ({by_id[i]['published']}) {by_id[i]['url']}" for i in cited
        )
        full = f"{text}\n\nSources:\n{sources}"
        trace = state["trace"] + [{"step": "answer", "cited": cited}]
        return {"messages": [AIMessage(full)], "answer": full, "cited": cited, "trace": trace}

    def abstain(state: QAState) -> dict:
        lines = [
            "I couldn't find enough evidence to answer that reliably, so I won't guess.",
            f"What's missing: {state.get('missing') or 'supporting passages'}.",
        ]
        if state["passages"]:
            lines.append("Closest items I found:")
            lines += [f"- [{p['item_id']}] {p['title']} {p['url']}" for p in state["passages"][:3]]
        text = "\n".join(lines)
        trace = state["trace"] + [{"step": "abstain"}]
        return {"messages": [AIMessage(text)], "answer": text, "cited": [], "trace": trace}

    g = StateGraph(QAState)
    g.add_node("prepare", prepare)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("reason", reason)
    g.add_node("tools", run_tools)
    g.add_node("evidence_check", evidence_check)
    g.add_node("answer", answer)
    g.add_node("abstain", abstain)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "rewrite_query")
    g.add_edge("rewrite_query", "reason")
    g.add_edge("reason", "tools")
    g.add_edge("tools", "evidence_check")
    g.add_conditional_edges("evidence_check", route, ["answer", "reason", "abstain"])
    g.add_edge("answer", END)
    g.add_edge("abstain", END)
    return g.compile(checkpointer=checkpointer)


def _thought(msg) -> str:
    """Qwen's thinking text (only present when thinking mode is on), else any plain content."""
    text = msg.additional_kwargs.get("reasoning_content") or str(msg.content)
    text = " ".join(text.split())
    return text if len(text) <= 300 else text[:297] + "..."


def parse_verdict(text: str) -> dict:
    """Read the judge's JSON, tolerating extra text around it. Unparseable means insufficient."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                "sufficient": bool(data.get("sufficient")),
                "missing": str(data.get("missing") or ""),
            }
        except json.JSONDecodeError:
            pass
    return {"sufficient": False, "missing": "could not read the evidence check's verdict"}


_CITATION = re.compile(r"\[([^\[\]]+)\]")


def keep_valid_citations(text: str, passages: list[dict]) -> tuple[str, list[str]]:
    """Remove citations to IDs that aren't in the passages; return the text and the cited IDs.

    This is the guard against fabricated citations: the model can only cite what was retrieved.
    """
    valid = {p["item_id"] for p in passages}
    cited: list[str] = []

    def check(m: re.Match) -> str:
        ids = [s.strip() for s in m.group(1).split(",")]
        good = [i for i in ids if i in valid]
        for i in good:
            if i not in cited:
                cited.append(i)
        return f"[{', '.join(good)}]" if good else ""

    return _CITATION.sub(check, text), cited
