"""The daily graph: a fixed pipeline that turns new items into a ranked, summarized digest.

    START -> collect -> candidates -> score -> pick -> summarize -> render -> END

Unlike the Q&A graph there's no LLM routing: every run takes the same path. Only `summarize`
calls Qwen. Everything else is plain code, so it's fast, testable and the same every day.

- collect:    run the sources (arXiv, HF, labs, HN, then Semantic Scholar enrich). Injected as a
              function, so tests and `radar daily --no-collect` can skip the network.
- candidates: items published in the last N days that weren't in an earlier digest ("dedupe":
              the store already merges duplicate IDs; this drops repeats across days).
- score:      build Features for each candidate from the store, then `scoring.score_all`.
- pick:       `scoring.pick`: 6 main + 2 reserved slots (early buzz, lab launch).
- summarize:  M2: one Qwen call per pick. M2.5 swaps this node for the multi-agent team.
- render:     write the Markdown file and the `digests` rows.
"""

import struct
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from radar import scoring
from radar.context import AppContext
from radar.digest import to_markdown
from radar.graphs import prompts
from radar.graphs.qa import keep_valid_citations
from radar.store import digests as digest_store
from radar.store import signals as signal_store


class DailyState(TypedDict, total=False):
    date: str
    collect_log: list[str]
    candidates: list[dict]  # Features as dicts
    scored: list[dict]  # {item_id, total, parts}
    picks: list[dict]  # item fields + slot, score, reason, summary
    notes: list[str]  # things worth telling the reader (e.g. claims cut by the verifier)
    path: str
    timings: dict[str, float]  # seconds per node, for the single vs. multi comparison
    stats: dict[str, Any]  # counters from the summarize step (M2.5 fills more)
    team_log: list[dict]  # M2.5: every handoff between the agents


def build_daily_graph(
    ctx: AppContext,
    llm: BaseChatModel,
    collect: Callable[[AppContext], list[str]] | None = None,
    summarize: Callable[[DailyState], dict] | None = None,
    mode: str = "single",
):
    """`summarize` replaces the default one-call-per-pick summarizer (M2.5 passes its team)."""
    s = ctx.settings

    def timed(name: str, fn: Callable[[DailyState], dict]):
        def run(state: DailyState) -> dict:
            start = time.perf_counter()
            out = fn(state)
            timings = {**state.get("timings", {}), name: round(time.perf_counter() - start, 2)}
            return {**out, "timings": timings}

        return run

    def collect_node(state: DailyState) -> dict:
        if collect is None:
            return {"collect_log": ["collection skipped (using the store as it is)"]}
        return {"collect_log": collect(ctx)}

    def candidates_node(state: DailyState) -> dict:
        today = date.fromisoformat(state["date"])
        feats = load_candidates(ctx, today, s.digest_window_days)
        return {"candidates": [asdict(f) for f in feats]}

    def score_node(state: DailyState) -> dict:
        weights = digest_store.get_profile(ctx.conn).get("weights")
        scored = scoring.score_all(
            [scoring.Features(**c) for c in state["candidates"]], weights=weights
        )
        return {"scored": [_scored_dict(x) for x in scored]}

    def pick_node(state: DailyState) -> dict:
        reserved = digest_store.get_profile(ctx.conn).get("reserved_slots", s.digest_reserved)
        scored = [_scored_from(d) for d in state["scored"]]
        picks = scoring.pick(scored, n_main=s.digest_main, reserved=reserved)
        rows = [
            item_row(ctx, p.scored.features.item_id, p.slot, p.scored.total, p.reason)
            for p in picks
        ]
        return {"picks": rows}

    def load_reserve(state: DailyState, n: int = 10) -> list[dict]:
        """The next items by score after the picks, for the Critic to fall back on."""
        chosen = {p["item_id"] for p in state["picks"]}
        out = []
        for d in state["scored"]:
            if len(out) >= n:
                break
            if d["features"]["item_id"] in chosen:
                continue
            sc = _scored_from(d)
            out.append(
                item_row(ctx, sc.features.item_id, "main", sc.total, scoring.explain(sc, "main"))
            )
        return out

    if summarize is None and mode == "multi":
        from radar.graphs.agents.team import make_team_summarizer

        summarize = make_team_summarizer(llm, s.interest_topics, load_reserve)

    def single_summarize(state: DailyState) -> dict:
        picks = []
        for p in state["picks"]:
            out = llm.invoke(prompts.SUMMARIZE.format(**_item_fields(p)))
            text = ensure_citation(str(out.content).strip(), p)
            picks.append({**p, "summary": text})
        return {"picks": picks, "stats": {"llm_calls": len(picks)}}

    def render_node(state: DailyState) -> dict:
        on = state["date"]
        md = to_markdown(on, state["picks"], mode=mode, notes=state.get("notes"))
        path = Path(s.digest_dir) / f"{on}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md)
        model = getattr(llm, "model", type(llm).__name__)
        with ctx.conn:
            digest_store.save_digest(
                ctx.conn,
                on,
                [
                    {k: p[k] for k in ("item_id", "slot", "score", "reason")} | {"rank": n}
                    for n, p in enumerate(state["picks"], 1)
                ],
            )
            for p in state["picks"]:
                if p.get("summary"):
                    digest_store.save_summary(ctx.conn, p["item_id"], p["summary"], model, mode)
        return {"path": str(path)}

    g = StateGraph(DailyState)
    g.add_node("collect", timed("collect", collect_node))
    g.add_node("candidates", timed("candidates", candidates_node))
    g.add_node("score", timed("score", score_node))
    g.add_node("pick", timed("pick", pick_node))
    g.add_node("summarize", timed("summarize", summarize or single_summarize))
    g.add_node("render", timed("render", render_node))
    g.add_edge(START, "collect")
    g.add_edge("collect", "candidates")
    g.add_edge("candidates", "score")
    g.add_edge("score", "pick")
    g.add_edge("pick", "summarize")
    g.add_edge("summarize", "render")
    g.add_edge("render", END)
    return g.compile()


# ---- reading features from the store -------------------------------------------------------


def load_candidates(ctx: AppContext, today: date, window_days: int) -> list[scoring.Features]:
    """Recent items not featured before, with everything the scoring function needs."""
    since = date.fromordinal(today.toordinal() - window_days).isoformat()
    shown = digest_store.featured_before(ctx.conn, today.isoformat())
    rows = ctx.conn.execute(
        "SELECT id, kind, source, published FROM items WHERE published >= ? AND published <= ?",
        (since, today.isoformat()),
    ).fetchall()
    rows = [r for r in rows if r["id"] not in shown]
    vectors = item_vectors(ctx, [r["id"] for r in rows])
    profile = profile_vector(ctx)
    out = []
    for r in rows:
        sig = signal_store.latest(ctx.conn, r["id"])
        age = max((today - date.fromisoformat(r["published"])).days, 0)
        vec = vectors.get(r["id"])
        out.append(
            scoring.Features(
                item_id=r["id"],
                kind=r["kind"],
                source=r["source"],
                age_days=age,
                relevance=scoring.cosine(vec, profile) if vec and profile else 0.0,
                hf_upvotes=sig["hf_upvotes"] or 0,
                hn_points=sig["hn_points"] or 0,
                citations=sig["citations"] or 0,
                citation_velocity=citation_velocity(ctx, r["id"], age),
            )
        )
    return out


def item_vectors(ctx: AppContext, ids: list[str]) -> dict[str, list[float]]:
    """Each item's abstract vector, read back from sqlite-vec (no re-embedding)."""
    out: dict[str, list[float]] = {}
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        marks = ",".join("?" * len(batch))
        rows = ctx.conn.execute(
            f"SELECT c.item_id, v.embedding FROM chunks c JOIN chunks_vec v ON v.chunk_id = c.id"
            f" WHERE c.kind = 'abstract' AND c.item_id IN ({marks})",
            batch,
        )
        for item_id, blob in rows:
            out[item_id] = list(struct.unpack(f"{len(blob) // 4}f", blob))
    return out


def profile_vector(ctx: AppContext) -> list[float]:
    """The interest profile as one vector: the mean of the seed topics' embeddings.

    M4 will add the vectors of items rated useful (and push away from not-relevant ones).
    """
    profile = digest_store.get_profile(ctx.conn)
    topics = profile.get("topics") or [
        t.strip() for t in ctx.settings.interest_topics.split(",") if t.strip()
    ]
    if not topics:
        return []
    return scoring.mean_vector(ctx.embeddings.embed_documents(topics))


def citation_velocity(ctx: AppContext, item_id: str, age_days: int) -> float:
    """Citations gained per day: the change between the first and last reading, or, with only
    one reading, citations spread over the paper's age."""
    rows = ctx.conn.execute(
        "SELECT date, citations FROM signals WHERE item_id = ? AND citations IS NOT NULL"
        " ORDER BY date",
        (item_id,),
    ).fetchall()
    if not rows:
        return 0.0
    if len(rows) >= 2:
        days = (date.fromisoformat(rows[-1][0]) - date.fromisoformat(rows[0][0])).days
        if days > 0:
            return max(rows[-1][1] - rows[0][1], 0) / days
    return rows[-1][1] / max(age_days, 1)


def item_row(ctx: AppContext, item_id: str, slot: str, score: float, reason: str) -> dict:
    """One digest entry: the item's fields plus why and where it was placed."""
    item = ctx.conn.execute(
        "SELECT id, title, abstract, tldr, url, published, kind FROM items WHERE id = ?",
        (item_id,),
    ).fetchone()
    return {
        "item_id": item["id"],
        "title": item["title"],
        "abstract": item["abstract"],
        "tldr": item["tldr"],
        "url": item["url"],
        "published": item["published"],
        "kind": item["kind"],
        "slot": slot,
        "score": score,
        "reason": reason,
    }


# ---- small helpers -------------------------------------------------------------------------


def _scored_dict(s: scoring.Scored) -> dict:
    return {"features": asdict(s.features), "total": s.total, "parts": s.parts}


def _scored_from(d: dict) -> scoring.Scored:
    return scoring.Scored(scoring.Features(**d["features"]), d["total"], d["parts"])


def _item_fields(p: dict) -> dict:
    return {
        "item_id": p["item_id"],
        "title": p["title"],
        "abstract": (p.get("abstract") or "")[:2000],
        "tldr": p.get("tldr") or "(none)",
    }


def ensure_citation(text: str, p: dict) -> str:
    """Keep only citations to this item; if the model left none, cite it at the end."""
    text, cited = keep_valid_citations(text, [p])
    return text if cited else f"{text} [{p['item_id']}]"
