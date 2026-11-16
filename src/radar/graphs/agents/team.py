"""The digest team: Critic -> Writer -> Verifier, coordinated by a supervisor written in code.

    START ──▶ supervisor ──▶ critic   ──┐
                 ▲  │  ──▶ writer   ──┤   every agent reports back to the supervisor
                 │  │  ──▶ verifier ──┤
                 └──┼─────────────────┘
                    └──▶ END (no picks left)

Supervisor rules, per pick:
  1. critic first. keep -> writer. drop -> the next item by score takes the slot, back to critic.
  2. writer -> verifier.
  3. verifier: unsupported claims and retries < 2 -> writer again, with the reasons.
     Otherwise cut what's still unsupported, file the pick, and start the next one.

Every arrow returns to the supervisor, and the supervisor alone decides what runs next. That's
the "supervisor" multi-agent pattern. Here the supervisor is plain code, not an LLM: the routing
rules are simple and fixed, so a model would only add time and a chance to get them wrong.
Each agent (critic.py, writer.py, verifier.py) is its own subgraph with its own prompt.
"""

from collections.abc import Callable
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from radar.graphs.agents.critic import build_critic
from radar.graphs.agents.verifier import (
    build_verifier,
    remove_claims,
    support_rate,
    unsupported,
)
from radar.graphs.agents.writer import build_writer

MAX_RETRIES = 2  # rewrites per item before unsupported claims are cut
MAX_DROPS = 3  # the Critic can't empty the digest: after this many drops, items are kept


class TeamState(TypedDict, total=False):
    topics: str
    queue: list[dict]  # picks still to process, in digest order
    reserve: list[dict]  # next-best items by score, to replace a dropped pick
    current: dict | None
    stage: Literal["", "critic", "writer", "verifier"]  # the agent that just ran
    keep: bool
    concerns: str
    draft: str
    results: list[dict]  # the Verifier's last verdicts
    problems: list[dict]
    retries: int
    next: str
    done: list[dict]
    notes: list[str]
    log: list[dict]  # every handoff, for the trace and the write-up
    stats: dict


def new_stats() -> dict:
    return {
        "llm_calls": 0,
        "critic_drops": 0,
        "rewrites": 0,
        "claims_checked": 0,
        "claims_supported_first_draft": 0,
        "claims_cut": 0,
    }


def build_team(llm: BaseChatModel, critic_llm=None, writer_llm=None, verifier_llm=None):
    """Each agent can get its own model; by default all three share `llm` (one Qwen)."""
    critic = build_critic(critic_llm or llm)
    writer = build_writer(writer_llm or llm)
    verifier = build_verifier(verifier_llm or llm)

    def bump(state: TeamState, **counts: int) -> dict:
        stats = dict(state.get("stats") or new_stats())
        for k, v in counts.items():
            stats[k] = stats.get(k, 0) + v
        return stats

    # ---- agents: each runs its subgraph and reports back ----------------------------------

    def critic_node(state: TeamState) -> dict:
        out = critic.invoke({"item": state["current"], "topics": state["topics"]})
        return {
            "keep": out["keep"],
            "concerns": out["concerns"],
            "stage": "critic",
            "stats": bump(state, llm_calls=1),
        }

    def writer_node(state: TeamState) -> dict:
        out = writer.invoke({"item": state["current"], "problems": state.get("problems") or []})
        return {"draft": out["draft"], "stage": "writer", "stats": bump(state, llm_calls=1)}

    def verifier_node(state: TeamState) -> dict:
        item = state["current"]
        passage = {
            "item_id": item["item_id"],
            "title": item["title"],
            "text": f"{item.get('abstract') or ''}\nTLDR: {item.get('tldr') or '(none)'}",
        }
        out = verifier.invoke({"text": state["draft"], "passages": [passage]})
        counts = {"llm_calls": 1 if out["claims"] else 0}
        if state.get("retries", 0) == 0:  # first-draft quality, for the eval
            counts["claims_checked"] = len(out["results"])
            counts["claims_supported_first_draft"] = sum(
                1 for r in out["results"] if r["supported"] is not False
            )
        return {"results": out["results"], "stage": "verifier", "stats": bump(state, **counts)}

    # ---- the supervisor: pure routing logic ------------------------------------------------

    def supervisor(state: TeamState) -> dict:
        cur, stage = state.get("current"), state.get("stage", "")
        log = list(state.get("log") or [])
        update: dict = {}

        if cur is not None and stage == "critic":
            drops = (state.get("stats") or {}).get("critic_drops", 0)
            reserve = list(state.get("reserve") or [])
            used = {d["item_id"] for d in state.get("done") or []} | {
                q["item_id"] for q in state.get("queue") or []
            }
            spare = [
                i
                for i, r in enumerate(reserve)
                if r.get("slot", "main") == cur.get("slot", "main") and r["item_id"] not in used
            ]
            if state["keep"] or drops >= MAX_DROPS or not spare:
                concerns = state["concerns"]  # kept items can still carry a concern
                log.append({"item": cur["item_id"], "critic": "keep", "concerns": concerns})
                cur = {**cur, "concerns": concerns}
                return {"current": cur, "next": "writer", "log": log}
            # Dropped: the next item that fits the same slot takes its place, and meets the
            # Critic too. No back-up for that slot: the pick stays, with its concerns shown.
            replacement = reserve.pop(spare[0])
            replacement["reason"] += f" (replaced {cur['item_id']}, dropped by the critic)"
            note = f"Critic dropped {cur['item_id']} ({cur['title'][:60]}): {state['concerns']}"
            log.append({"item": cur["item_id"], "critic": "drop", "concerns": state["concerns"]})
            return {
                "current": replacement,
                "reserve": reserve,
                "next": "critic",
                "notes": [*(state.get("notes") or []), note],
                "log": log,
                "stats": bump(state, critic_drops=1),
            }

        if cur is not None and stage == "writer":
            return {"next": "verifier"}

        if cur is not None and stage == "verifier":
            bad = unsupported(state["results"])
            retries = state.get("retries", 0)
            log.append(
                {"item": cur["item_id"], "verifier": f"{len(bad)} unsupported", "try": retries}
            )
            if bad and retries < MAX_RETRIES:
                return {
                    "problems": [{"claim": r["claim"], "why": r["why"]} for r in bad],
                    "retries": retries + 1,
                    "next": "writer",
                    "log": log,
                    "stats": bump(state, rewrites=1),
                }
            summary = state["draft"]
            notes = list(state.get("notes") or [])
            if bad:  # out of retries: cut what still isn't supported
                summary = remove_claims(summary, bad) or fallback_summary(cur)
                notes.append(f"Verifier cut {len(bad)} unsupported claim(s) from {cur['item_id']}")
            finished = {
                **cur,
                "summary": summary,
                "support_rate": support_rate(state["results"]),
                "rewrites": retries,
            }
            update = {
                "done": [*(state.get("done") or []), finished],
                "notes": notes,
                "log": log,
                "stats": bump(state, claims_cut=len(bad)),
            }
            cur = None

        # Nothing in progress: start the next pick, or finish.
        queue = list(state.get("queue") or [])
        if not queue:
            return {**update, "current": None, "next": "finish"}
        return {
            **update,
            "current": queue.pop(0),
            "queue": queue,
            "stage": "",
            "draft": "",
            "problems": [],
            "retries": 0,
            "next": "critic",
        }

    g = StateGraph(TeamState)
    g.add_node("supervisor", supervisor)
    g.add_node("critic", critic_node)
    g.add_node("writer", writer_node)
    g.add_node("verifier", verifier_node)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {"critic": "critic", "writer": "writer", "verifier": "verifier", "finish": END},
    )
    for agent in ("critic", "writer", "verifier"):
        g.add_edge(agent, "supervisor")  # every agent reports back to the supervisor
    return g.compile(name="digest_team")


def fallback_summary(item: dict) -> str:
    """When every claim was cut: quote the abstract's first sentence rather than say nothing."""
    first = (item.get("abstract") or item["title"]).split(". ")[0].strip().rstrip(".")
    return f'From the abstract: "{first}." [{item["item_id"]}]'


def make_team_summarizer(
    llm: BaseChatModel,
    topics: str | Callable[[], str],
    load_reserve: Callable[[dict], list[dict]],
) -> Callable[[dict], dict]:
    """The daily graph's `summarize` node, multi-agent version (DIGEST_MODE=multi).

    `topics` can be a function, so the Critic sees the profile as it is when the node runs
    (M4: the profile may have just changed earlier in the same daily run)."""
    team = build_team(llm)

    def summarize(state: dict) -> dict:
        out = team.invoke(
            {
                "topics": topics() if callable(topics) else topics,
                "queue": state["picks"],
                "reserve": load_reserve(state),
                "stats": new_stats(),
                "done": [],
                "notes": [],
                "log": [],
            },
            # each pick takes up to ~12 steps (critic, writer/verifier retries, supervisor hops)
            config={"recursion_limit": 40 * (len(state["picks"]) + MAX_DROPS) + 10},
        )
        return {
            "picks": out["done"],
            "notes": out.get("notes", []),
            "stats": out["stats"],
            "team_log": out.get("log", []),
        }

    return summarize
