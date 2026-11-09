"""Single- vs. multi-agent comparison (M2.5): is the agent team worth its extra time?

Digest: build the same day's digest twice from the same store (no collection, so both runs see
the same candidates): once with M2's one-call summarizer, once with the Critic/Writer/Verifier
team. Then a fresh Verifier pass judges every final summary from both runs the same way.

Q&A (optional, with a questions file): answer each question with and without the Verifier
agent, and judge both final answers the same way.

Caveat for the write-up: the judge is the same Qwen that the Verifier uses, so the multi-agent
side is graded by a model it has already satisfied once. A stronger, independent judge (a bigger
local model, or human labels on a sample) would make the comparison fairer.
"""

import time
from datetime import date, datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from radar.context import AppContext
from radar.graphs.agents.verifier import build_verifier, support_rate


def compare(
    ctx: AppContext,
    on: date,
    llm: BaseChatModel,
    questions: list[str] | None = None,
    out_dir: Path | None = None,
) -> dict:
    from radar import services

    judge = build_verifier(llm)
    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    out_dir = out_dir or Path("evals/results")
    report: dict = {"date": on.isoformat(), "digest": {}, "qa": []}

    for mode in ("single", "multi"):
        settings = ctx.settings.model_copy(
            update={"digest_dir": out_dir / f"agents-{stamp}" / mode}
        )
        run_ctx = AppContext(settings, ctx.conn, ctx.embeddings, ctx.vectors)
        start = time.perf_counter()
        result = services.daily(run_ctx, on=on, collect=False, mode=mode, llm=llm)
        total = time.perf_counter() - start
        verdicts = [judge_summary(judge, p) for p in result.picks]
        claims = [r for v in verdicts for r in v]
        report["digest"][mode] = {
            "seconds_total": round(total, 1),
            "seconds_summarize": result.timings.get("summarize"),
            "llm_calls": result.stats.get("llm_calls", len(result.picks)),
            "claims": len(claims),
            "support_rate": support_rate(claims),
            "picks": [p["item_id"] for p in result.picks],
            "stats": result.stats,
            "notes": result.notes,
            "path": result.path,
        }

    for q in questions or []:
        row = {"question": q}
        for verify in (False, True):
            agent = services.Agent(ctx, llm=llm, verify=verify)
            start = time.perf_counter()
            res = agent.ask(q)
            secs = time.perf_counter() - start
            body = res.answer.split("\n\nSources:")[0]
            claims = (
                judge.invoke({"text": body, "passages": res.passages})["results"]
                if res.cited
                else []
            )
            row["verified" if verify else "plain"] = {
                "seconds": round(secs, 1),
                "abstained": not res.cited,
                "claims": len(claims),
                "support_rate": support_rate(claims),
            }
        report["qa"].append(row)

    report["markdown"] = to_markdown(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"agents-{stamp}.md"
    path.write_text(report["markdown"])
    report["path"] = str(path)
    return report


def judge_summary(judge, pick: dict) -> list[dict]:
    passage = {
        "item_id": pick["item_id"],
        "title": pick["title"],
        "text": f"{pick.get('abstract') or ''}\nTLDR: {pick.get('tldr') or '(none)'}",
    }
    return judge.invoke({"text": pick.get("summary") or "", "passages": [passage]})["results"]


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def to_markdown(r: dict) -> str:
    d = r["digest"]
    lines = [
        f"# Single vs. multi-agent: digest for {r['date']}",
        "",
        "| | single (M2) | multi (M2.5) |",
        "|---|---|---|",
    ]
    rows = [
        ("Claim support (judged after the run)", "support_rate", _pct),
        ("Claims judged", "claims", str),
        ("Model calls in summarize", "llm_calls", str),
        ("Summarize time (s)", "seconds_summarize", str),
        ("Total time (s)", "seconds_total", str),
    ]
    for label, key, fmt in rows:
        lines.append(f"| {label} | {fmt(d['single'][key])} | {fmt(d['multi'][key])} |")
    s = d["multi"]["stats"]
    lines += [
        "",
        f"Multi-agent: critic dropped {s.get('critic_drops', 0)}, writer rewrites "
        f"{s.get('rewrites', 0)}, claims cut {s.get('claims_cut', 0)}, first-draft claims "
        f"supported {s.get('claims_supported_first_draft', 0)}/{s.get('claims_checked', 0)}.",
        "",
        f"Picks changed by the critic: "
        f"{sorted(set(d['single']['picks']) ^ set(d['multi']['picks'])) or 'none'}",
    ]
    if d["multi"]["notes"]:
        lines += ["", "Notes:"] + [f"- {n}" for n in d["multi"]["notes"]]
    if r["qa"]:
        lines += [
            "",
            "## Q&A: plain vs. with the Verifier",
            "",
            "| Question | plain support | verified support | plain s | verified s |",
            "|---|---|---|---|---|",
        ]
        for q in r["qa"]:
            p, v = q["plain"], q["verified"]
            lines.append(
                f"| {q['question'][:60]} | {_pct(p['support_rate'])} | "
                f"{_pct(v['support_rate'])} | {p['seconds']} | {v['seconds']} |"
            )
    lines += [
        "",
        "Judge: the same local Qwen as the Verifier (see caveat in evals_agents.py).",
        "Critic agreement with your ratings needs M4 ratings; not measured yet.",
    ]
    return "\n".join(lines) + "\n"
