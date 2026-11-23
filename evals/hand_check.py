"""Hand-check the LLM judge: does a human agree with Qwen's per-claim verdicts?

The eval numbers ("claims supported") come from the Verifier agent acting as a judge, and the
judge is the same local Qwen that wrote and checked the answers. So they are upper bounds. This
script measures how far to trust that judge by comparing it with your own labels on a few answers.

Two steps, run from the repo root:

    # 1. Answer a few questions, let Qwen judge every claim, write a blind labeling sheet.
    uv run python evals/hand_check.py export                 # the default 5 questions
    uv run python evals/hand_check.py export --pick 1,3,4    # or choose lines of questions.jsonl
    REASON_THINKING=false uv run python evals/hand_check.py export --fast   # much faster on the M3

    # 2. After you fill in the `human` column of the CSV: how often do you and Qwen agree?
    uv run python evals/hand_check.py agree evals/results/hand-check-<stamp>.csv
    uv run python evals/hand_check.py agree evals/results/hand-check-*[0-9].csv  # pool runs

Files written by `export` (all in evals/results/):
- hand-check-<stamp>.md         read this: each answer, its claims, the source text the judge saw
- hand-check-<stamp>.csv        label this: one row per claim, an empty `human` column
- hand-check-<stamp>.judge.json Qwen's verdicts, kept in a separate file so you label blind

Labels for the `human` column: S = supported by the source text, U = not supported (adds a fact,
number or comparison the source doesn't give, or contradicts it), M = meta (a statement about the
answer or the sources, not about the research; left out of the scores).
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

QUESTIONS = Path("evals/questions.jsonl")
OUT_DIR = Path("evals/results")
# Lines of questions.jsonl (1-based). Chosen because the earlier Q&A comparison produced claims
# for all five, and on two of them (7, 8) the judge rejected some claims, so the sheet should
# contain both verdicts.
DEFAULT_PICK = "3,4,6,7,8"


# ---- step 1: export ------------------------------------------------------------------------


def export(pick: str, fast: bool = False) -> None:
    from radar.cli import _ctx
    from radar.evals import load_questions, make_target
    from radar.graphs.agents.verifier import build_verifier
    from radar.models import get_chat_model

    rows = load_questions(QUESTIONS)
    numbers = [int(x) for x in pick.split(",") if x.strip()]
    chosen = [(n, rows[n - 1]) for n in numbers]

    ctx = _ctx()
    # The same answering path `radar eval -t agent` uses. Its evidence check normally runs Qwen
    # in thinking mode (several minutes per check on the M3); --fast turns thinking off there.
    # That changes how answers are produced a little, not how they are judged.
    answer = make_target(ctx, "agent", judge=get_chat_model(ctx.settings) if fast else None)
    judge = build_verifier(get_chat_model(ctx.settings))  # the same judge as `radar eval --judge`

    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / f"hand-check-{stamp}"

    md = [
        f"# Hand-check of the claim judge ({stamp})",
        "",
        "Label each claim in the CSV next to this file (`human` column: S, U or M).",
        "Judge a claim only against the source text shown under it: that is all the judge saw.",
        "Qwen's verdicts are in the .judge.json file; don't open it until you're done.",
    ]
    sheet, verdicts = [], {}

    for i, (n, row) in enumerate(chosen, 1):
        q = row["question"]
        print(f"[{i}/{len(chosen)}] answering: {q}", file=sys.stderr)
        out = answer({"question": q})
        body = out["answer"].split("\n\nSources:")[0]
        md += ["", f"## Q{n}. {q}", "", f"Answered in {out['seconds']}s.", "", "**Answer:**", ""]
        md += [f"> {line}" if line else ">" for line in body.splitlines()]
        if out["abstained"] or not out["passages"]:
            md += ["", "_The agent abstained (no citations), so there are no claims to check._"]
            continue

        print(f"[{i}/{len(chosen)}] judging claims", file=sys.stderr)
        results = judge.invoke({"text": body, "passages": out["passages"]})["results"]
        cited = {c for r in results for c in _cites(r["claim"])}
        shown = [p for p in out["passages"] if p["item_id"] in cited] or out["passages"]

        md += ["", "| Claim id | Claim |", "|---|---|"]
        for r in results:
            cid = f"Q{n}.{r['n']}"
            md.append(f"| {cid} | {r['claim'].replace('|', '/')} |")
            sheet.append({"id": cid, "question": n, "claim": r["claim"], "human": "", "notes": ""})
            verdicts[cid] = {"supported": r["supported"], "why": r["why"]}

        md += ["", "**Source text the judge saw** (the passages these claims cite):", ""]
        for p in shown:
            md += [f"<details><summary>[{p['item_id']}] {p['title']}</summary>", ""]
            md += [f"> {line}" if line else ">" for line in p["text"].splitlines()]
            md += ["", "</details>", ""]

    base.with_suffix(".md").write_text("\n".join(md) + "\n")
    with base.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "claim", "human", "notes"])
        w.writeheader()
        w.writerows(sheet)
    base.with_suffix(".judge.json").write_text(json.dumps(verdicts, indent=2) + "\n")

    print(f"\n{len(sheet)} claims from {len(chosen)} questions.")
    print(f"Read:  {base}.md")
    print(f"Label: {base}.csv  (fill the `human` column with S, U or M)")
    print(f"Then:  uv run python evals/hand_check.py agree {base}.csv")


def _cites(claim: str) -> list[str]:
    from radar.graphs.agents.verifier import _CITE

    return [i.strip() for m in _CITE.finditer(claim) for i in m.group(1).split(",")]


# ---- step 2: agreement ---------------------------------------------------------------------


def agree(csv_paths: list[Path]) -> str:
    """Pure Python (no radar imports), so it runs anywhere. Several CSVs (from separate export
    runs) are pooled; claim ids stay unique as long as the runs used different questions."""
    verdicts, sheet = {}, []
    for csv_path in csv_paths:
        verdicts.update(json.loads(_judge_file(csv_path).read_text()))
        # utf-8-sig drops the BOM Excel adds; skip any title lines Numbers puts above the header
        lines = csv_path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
        start = next((i for i, x in enumerate(lines) if x.startswith("id,")), 0)
        sheet += list(csv.DictReader(lines[start:]))
    csv_path = csv_paths[-1]

    pairs, meta, unlabeled, no_verdict = [], 0, 0, 0
    for row in sheet:
        h = (row.get("human") or "").strip().upper()[:1]
        j = verdicts.get(row["id"], {}).get("supported")
        if h == "M":
            meta += 1
        elif h not in ("S", "U"):
            unlabeled += 1
        elif j is None:
            no_verdict += 1
        else:
            pairs.append((row, h == "S", bool(j)))

    n = len(pairs)
    if not n:
        return "No claims labeled S or U yet: fill in the `human` column first."

    both_s = sum(h and j for _, h, j in pairs)
    both_u = sum(not h and not j for _, h, j in pairs)
    judge_only = sum(j and not h for _, h, j in pairs)  # judge said S, you said U: too lenient
    human_only = sum(h and not j for _, h, j in pairs)  # judge said U, you said S: too strict
    observed = (both_s + both_u) / n
    # Cohen's kappa: agreement beyond what two raters with these S/U rates would hit by chance.
    p_h, p_j = sum(h for _, h, _ in pairs) / n, sum(j for _, _, j in pairs) / n
    chance = p_h * p_j + (1 - p_h) * (1 - p_j)
    kappa = (observed - chance) / (1 - chance) if chance < 1 else None

    lines = [
        f"# Judge vs. human: {', '.join(p.name for p in csv_paths)}",
        "",
        f"Claims compared: {n} (left out: {meta} meta, {unlabeled} unlabeled,"
        f" {no_verdict} with no judge verdict)",
        "",
        "| | You: supported | You: not supported |",
        "|---|---|---|",
        f"| Judge: supported | {both_s} | {judge_only} |",
        f"| Judge: not supported | {human_only} | {both_u} |",
        "",
        f"- Agreement: {both_s + both_u}/{n} = {observed:.0%}",
        f"- Cohen's kappa: {'n/a (one label only)' if kappa is None else f'{kappa:.2f}'}",
        f"- Support rate by the judge: {p_j:.0%}; by you: {p_h:.0%} (same {n} claims)",
        f"- Judge too lenient (said supported, you didn't): {judge_only}",
        f"- Judge too strict (said not supported, you did): {human_only}",
    ]
    disagreements = [(r, h, j) for r, h, j in pairs if h != j]
    if disagreements:
        lines += ["", "## Disagreements", ""]
        for r, h, j in disagreements:
            why = verdicts[r["id"]].get("why") or ""
            lines += [
                f"- **{r['id']}** you {'S' if h else 'U'}, judge {'S' if j else 'U'}: {r['claim']}",
                f"  - judge's reason: {why or '(none)'}",
            ]
            if r.get("notes"):
                lines.append(f"  - your note: {r['notes']}")
    report = "\n".join(lines) + "\n"
    out = csv_path.with_name(csv_path.stem + "-agreement.md")
    out.write_text(report)
    return report + f"\nSaved: {out}"


def _judge_file(csv_path: Path) -> Path:
    """The .judge.json written next to the sheet. Matches on the hand-check-<stamp> part of the
    name, so a renamed copy of the sheet (e.g. "...-v2.csv") still finds its verdicts."""
    import re

    m = re.search(r"hand-check-\d{8}-\d{4}", csv_path.name)
    name = (m.group(0) if m else csv_path.stem) + ".judge.json"
    for folder in (csv_path.parent, OUT_DIR):
        if (folder / name).exists():
            return folder / name
    raise SystemExit(f"Can't find {name} next to {csv_path} or in {OUT_DIR}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="answer questions, judge claims, write the sheet")
    e.add_argument("--pick", default=DEFAULT_PICK, help="lines of questions.jsonl, e.g. 3,4,6")
    e.add_argument("--fast", action="store_true", help="no thinking mode in the evidence check")
    a = sub.add_parser("agree", help="compare your labels with the judge's")
    a.add_argument("csv", type=Path, nargs="+", help="one or more filled-in sheets")
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.pick, args.fast)
    else:
        print(agree(args.csv))


if __name__ == "__main__":
    main()
