"""The Verifier agent: is each claim supported by the passage it cites?

Answers Checkpoint 2 feedback "check each summary claim is supported by its cited source".
M1's citation guard only checks that a cited ID is real; this checks what the sentence says.
Shared by the digest team and the Q&A graph.

    START -> split (code) -> check (LLM) -> END

Splitting into claims is plain code (one sentence = one claim), so the model does only the
judging, in one call for all claims of a text.
"""

import re
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from radar.graphs import prompts
from radar.graphs.agents import parse_json

_CITE = re.compile(r"\[([^\[\]]+)\]")


class VerifierState(TypedDict, total=False):
    text: str
    passages: list[dict]  # {item_id, title, text}
    claims: list[dict]  # {n, claim, cites}
    results: list[dict]  # {n, claim, supported: bool | None, why}


def split_claims(text: str) -> list[dict]:
    """One claim per sentence or bullet. Citations stay attached to their sentence."""
    parts = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        # split after . ! ? (and any citation right after them) when a new sentence follows
        parts += re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=\])\s+(?=[A-Z])", line)
    claims = []
    for p in parts:
        p = p.strip()
        if len(re.sub(_CITE, "", p).strip(" .")) < 12:  # too short to be a claim
            continue
        cites = [i.strip() for m in _CITE.finditer(p) for i in m.group(1).split(",")]
        claims.append({"n": len(claims) + 1, "claim": p, "cites": cites})
    return claims


def build_verifier(llm: BaseChatModel):
    def split(state: VerifierState) -> dict:
        return {"claims": split_claims(state["text"])}

    def check(state: VerifierState) -> dict:
        claims = state["claims"]
        if not claims:
            return {"results": []}
        cited = {c for claim in claims for c in claim["cites"]}
        passages = [p for p in state["passages"] if p["item_id"] in cited] or state["passages"]
        listing = "\n".join(f"{c['n']}. {c['claim']}" for c in claims)
        sources = "\n\n".join(
            f"[{p['item_id']}] {p['title']}\n{p['text'][:2000]}" for p in passages
        )
        out = llm.invoke(prompts.VERIFIER.format(passages=sources, claims=listing))
        verdicts = parse_json(str(out.content), [])
        by_n = {}
        if isinstance(verdicts, list):
            by_n = {v.get("n"): v for v in verdicts if isinstance(v, dict)}
        results = []
        for c in claims:
            v = by_n.get(c["n"])
            # No verdict for a claim = unverified (None): kept, but counted separately.
            supported = None if v is None else bool(v.get("supported"))
            why = "" if v is None else str(v.get("why") or "")
            results.append({"n": c["n"], "claim": c["claim"], "supported": supported, "why": why})
        return {"results": results}

    g = StateGraph(VerifierState)
    g.add_node("split", split)
    g.add_node("check", check)
    g.add_edge(START, "split")
    g.add_edge("split", "check")
    g.add_edge("check", END)
    return g.compile(name="verifier")


def unsupported(results: list[dict]) -> list[dict]:
    return [r for r in results if r["supported"] is False]


def remove_claims(text: str, bad: list[dict]) -> str:
    """Cut the unsupported sentences out of the text."""
    for r in bad:
        text = text.replace(r["claim"], "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n\s*[-*•]\s*\n", "\n", text)
    return text.strip()


def support_rate(results: list[dict]) -> float | None:
    judged = [r for r in results if r["supported"] is not None]
    return sum(r["supported"] for r in judged) / len(judged) if judged else None
