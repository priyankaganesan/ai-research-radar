"""The Profile agent (M4): turns ratings and knowledge gaps into changes to the interest topics.

Ratings already move the profile *vector* and the scoring weights in code (`radar/feedback.py`).
What code can't do is read titles and notice a theme: "the last four useful papers were all
about KV-cache compression, and two unanswered questions were too". That needs language, so
it's an agent, built like the M2.5 agents (its own role, prompt and subgraph):

    START -> propose -> guard -> END

- propose: Qwen reads the topics, recently rated titles and open gaps, and suggests
           {add, remove, why}.
- guard:   code, not the model, decides what is applied: at most 2 additions and 1 removal per
           run, short phrases only, no duplicates, never fewer than MIN_TOPICS or more than
           MAX_TOPICS. The model proposes; the rules dispose. A garbled reply changes nothing.
"""

import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from radar.graphs import prompts
from radar.graphs.agents import parse_json

MAX_ADD = 2
MAX_REMOVE = 1
MIN_TOPICS = 3
MAX_TOPICS = 10
MAX_TOPIC_CHARS = 60


class ProfileState(TypedDict, total=False):
    topics: list[str]
    useful: list[str]  # titles rated useful since the last update
    not_relevant: list[str]  # titles rated not relevant
    gaps: list[str]  # open unanswered questions / gap topics
    proposal: dict  # what the model asked for
    added: list[str]  # what the guard let through
    removed: list[str]
    rejected: list[str]  # proposals the guard refused, with the reason
    why: str
    new_topics: list[str]


def _bullets(lines: list[str], limit: int = 12) -> str:
    return "\n".join(f"- {x}" for x in lines[:limit]) or "(none)"


def build_profile_agent(llm: BaseChatModel):
    def propose(state: ProfileState) -> dict:
        out = llm.invoke(
            prompts.PROFILE.format(
                topics=_bullets(state["topics"], MAX_TOPICS),
                useful=_bullets(state.get("useful") or []),
                not_relevant=_bullets(state.get("not_relevant") or []),
                gaps=_bullets(state.get("gaps") or []),
            )
        )
        return {"proposal": parse_json(str(out.content), {})}

    def guard(state: ProfileState) -> dict:
        return apply_proposal(state["topics"], state.get("proposal") or {})

    g = StateGraph(ProfileState)
    g.add_node("propose", propose)
    g.add_node("guard", guard)
    g.add_edge(START, "propose")
    g.add_edge("propose", "guard")
    g.add_edge("guard", END)
    return g.compile(name="profile")


def clean_topic(text) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip().strip(".,;:\"'")
    return text


def apply_proposal(topics: list[str], proposal: dict) -> dict:
    """The guard: apply the parts of a proposal that pass the rules."""
    current = list(topics)
    lower = {t.lower() for t in current}
    added, removed, rejected = [], [], []

    adds = proposal.get("add") if isinstance(proposal.get("add"), list) else []
    removes = proposal.get("remove") if isinstance(proposal.get("remove"), list) else []

    for raw in removes:
        t = clean_topic(raw)
        match = next((c for c in current if c.lower() == t.lower()), None)
        if match is None:
            rejected.append(f"remove {t!r}: not a current topic")
        elif len(removed) >= MAX_REMOVE:
            rejected.append(f"remove {t!r}: only {MAX_REMOVE} removal per run")
        elif len(current) <= MIN_TOPICS:
            rejected.append(f"remove {t!r}: would leave fewer than {MIN_TOPICS} topics")
        else:
            current.remove(match)
            lower.discard(match.lower())
            removed.append(match)

    for raw in adds:
        t = clean_topic(raw)
        if not t or len(t) > MAX_TOPIC_CHARS or len(t.split()) > 8:
            rejected.append(f"add {t[:40]!r}: not a short topic phrase")
        elif t.lower() in lower:
            rejected.append(f"add {t!r}: already a topic")
        elif len(added) >= MAX_ADD:
            rejected.append(f"add {t!r}: only {MAX_ADD} additions per run")
        elif len(current) >= MAX_TOPICS:
            rejected.append(f"add {t!r}: already {MAX_TOPICS} topics")
        else:
            current.append(t)
            lower.add(t.lower())
            added.append(t)

    return {
        "added": added,
        "removed": removed,
        "rejected": rejected,
        "why": str(proposal.get("why") or "").strip(),
        "new_topics": current,
    }
