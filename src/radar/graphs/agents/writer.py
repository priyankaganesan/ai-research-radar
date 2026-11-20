"""The Writer agent: a cited 2-3 sentence summary of one item.

This is M2's `summarize` step, now with a memory of what went wrong: when the Verifier rejects
claims, the supervisor sends them back here as `problems`, and the Writer rewrites.

    START -> write -> END
"""

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from radar.graphs import prompts
from radar.graphs.qa import keep_valid_citations
from radar.safety import fence


class WriterState(TypedDict, total=False):
    item: dict
    problems: list[dict]  # [{claim, why}] from the Verifier's last check
    draft: str


def build_writer(llm: BaseChatModel):
    def write(state: WriterState) -> dict:
        item = state["item"]
        problems = state.get("problems") or []
        feedback = ""
        if problems:
            lines = "\n".join(f"- {p['claim']} ({p['why']})" for p in problems)
            feedback = prompts.WRITER_FEEDBACK.format(problems=lines)
        out = llm.invoke(
            prompts.WRITER.format(
                item_id=item["item_id"],
                title=fence(item["title"]),
                abstract=fence((item.get("abstract") or "")[:2000]),
                tldr=fence(item.get("tldr")) or "(none)",
                feedback=feedback,
            )
        )
        text, cited = keep_valid_citations(str(out.content).strip(), [item])
        if not cited:
            text = f"{text} [{item['item_id']}]"
        return {"draft": text}

    g = StateGraph(WriterState)
    g.add_node("write", write)
    g.add_edge(START, "write")
    g.add_edge("write", END)
    return g.compile(name="writer")
