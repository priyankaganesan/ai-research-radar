"""The Critic agent: should this ranked item really be in the digest?

Answers Checkpoint 2 feedback "popularity must not stand in for quality". The scoring formula
can't read an abstract; the Critic can. It returns {keep, concerns}; the supervisor replaces a
dropped item with the next one by score.

    START -> judge -> END
"""

from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from radar.graphs import prompts
from radar.graphs.agents import parse_json


class CriticState(TypedDict, total=False):
    item: dict  # item_id, title, abstract, tldr, kind, slot, reason
    topics: str
    keep: bool
    concerns: str


def build_critic(llm: BaseChatModel):
    def judge(state: CriticState) -> dict:
        item = state["item"]
        out = llm.invoke(
            prompts.CRITIC.format(
                topics=state["topics"],
                item_id=item["item_id"],
                kind=item.get("kind", "paper"),
                slot=item.get("slot", "main"),
                reason=item.get("reason", ""),
                title=item["title"],
                abstract=(item.get("abstract") or "")[:2000],
                tldr=item.get("tldr") or "(none)",
            )
        )
        verdict = parse_json(str(out.content), {})
        # An unreadable verdict keeps the item: the Critic can veto, but only on purpose.
        return {
            "keep": bool(verdict.get("keep", True)),
            "concerns": str(verdict.get("concerns") or ""),
        }

    g = StateGraph(CriticState)
    g.add_node("judge", judge)
    g.add_edge(START, "judge")
    g.add_edge("judge", END)
    return g.compile(name="critic")
