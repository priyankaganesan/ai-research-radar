"""Render a digest to Markdown (data/digests/YYYY-MM-DD.md)."""

SLOT_LABEL = {"main": "Top pick", "early_buzz": "Early buzz", "lab_launch": "Lab launch"}


def to_markdown(
    on: str,
    picks: list[dict],
    mode: str = "single",
    notes: list[str] | None = None,
    review: list[str] | None = None,
) -> str:
    lines = [f"# AI Research Radar: {on}", ""]
    if review:  # safety: say up front when this digest needs a careful read
        lines.append("> **Needs review.** Read the sources before relying on this digest:")
        lines += [f"> - {r}" for r in review]
        lines.append("")
    lines.append(f"{len(picks)} items, ranked by `scoring.py`; summaries by local Qwen ({mode}).")
    lines.append("")
    for n, p in enumerate(picks, 1):
        lines.append(f"## {n}. {p['title']}")
        lines.append("")
        label = SLOT_LABEL[p["slot"]]
        lines.append(f"**{label}** · `{p['item_id']}` · {p['published']} · [link]({p['url']})")
        lines.append("")
        lines.append(p.get("summary") or "_No summary._")
        lines.append("")
        lines.append(f"_Why: {p['reason']}_")
        if p.get("concerns"):
            lines.append("")
            lines.append(f"_Critic: {p['concerns']}_")
        lines.append("")
    if notes:
        lines.append("---")
        lines.append("")
        lines += [f"- {n}" for n in notes]
        lines.append("")
    lines.append(
        "Rate an item: `radar rate ITEM_ID useful|not_relevant|skipped`."
        " Ratings tune tomorrow's digest (`radar profile` shows how)."
    )
    return "\n".join(lines) + "\n"
