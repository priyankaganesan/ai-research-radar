"""Safety checks that are plain code: injection scanning, the digest review flag, quiet sources.

Everything the agent reads (abstracts, blog posts, HN titles, full text) was written by someone
else and goes straight into a prompt. Text like "ignore the previous instructions and ..." is a
*prompt injection*: an attempt to make the model follow the source instead of us. Two layers:

1. `fence`: every prompt puts source text inside <source> tags and says that text in those tags
   is data, never instructions (`prompts.DATA_RULE`). `fence` removes any <source> tags inside
   the text itself, so a source can't close the fence early and "escape" it.
2. `scan`: a regex check for instruction-like text. A hit doesn't prove an attack, and papers
   *about* prompt injection quote such text on purpose, so a hit is treated as a signal for a
   human: the digest holds the item back (or, for a paper about attacks, keeps it and says so),
   and a Q&A answer that cites it carries a note.

Neither layer is complete. Regexes miss paraphrases, and a 9B model may still follow a fenced
instruction. What makes this acceptable here is the third layer, outside this file: every agent
tool is read-only, so a hijacked model can at worst write a bad summary, not act.

Also here, because they are thresholds rather than judgment:
- `review_reasons`: when a whole digest should be marked "needs review".
- `quiet_sources`: a source that fetched nothing on its last two runs.
"""

import re
from dataclasses import dataclass

# Phrases addressed to the model rather than to a human reader.
_INJECTION = [
    r"\b(ignore|disregard|forget|override)\b[^.\n]{0,30}\b(previous|prior|above|earlier|preceding|all|your)\b[^.\n]{0,20}\b(instructions?|prompts?|directions|rules|context)\b",
    r"\byou are now\b",
    r"\bnew (system )?instructions?\s*:",
    r"\b(reveal|print|show|repeat)\b[^.\n]{0,20}\bsystem prompt\b",
    r"\b(respond|reply|answer|output)\b[^.\n]{0,15}\bonly with\b",
    r"\bdo not (summari[sz]e|mention|cite) (this|the) (paper|text|document|article|post)\b",
    r"\b(do not|don't|never) reveal (this|the|your) (prompt|instructions|system)\b",
    r"\b(as|to) (an|the) (ai|assistant|language model|llm)\b[^.\n]{0,40}\b(must|should|will)\b",
    r"</?\s*(system|assistant|user|source|instructions?)\s*>",
    r"\[/?INST\]|<\|im_(start|end)\|>",
]
INJECTION = re.compile("|".join(f"(?:{p})" for p in _INJECTION), re.IGNORECASE)

# Research *about* attacks quotes attack text on purpose. Such items are flagged, not held.
ABOUT_ATTACKS = re.compile(
    r"prompt[- ]injection|jailbreak|adversarial prompt|red[- ]teaming|indirect injection",
    re.IGNORECASE,
)

_FENCE_TAG = re.compile(r"</?\s*source\b[^>]*>", re.IGNORECASE)


@dataclass
class Finding:
    snippet: str  # the matched text with a little context, for a human to look at
    about_attacks: bool  # the text is research about attacks: flag, don't hold back

    @property
    def hold(self) -> bool:
        return not self.about_attacks


def scan(*texts: str | None) -> Finding | None:
    """Instruction-like text in any of `texts`, or None."""
    joined = "\n".join(t for t in texts if t)
    m = INJECTION.search(joined)
    if not m:
        return None
    start, end = max(m.start() - 40, 0), min(m.end() + 40, len(joined))
    snippet = " ".join(joined[start:end].split())
    return Finding(snippet=snippet, about_attacks=bool(ABOUT_ATTACKS.search(joined)))


def fence(text: str | None) -> str:
    """Source text made safe to put inside <source> tags: no tags of its own."""
    return _FENCE_TAG.sub("", text or "")


# ---- the digest review flag ----------------------------------------------------------------


def review_reasons(
    picks: list[dict],
    stats: dict | None = None,
    notes: list[str] | None = None,
    min_support: float = 0.8,
    max_cut: int = 2,
    max_unverified: float = 0.25,
    held: int = 0,
    quiet: list[str] | None = None,
    critic_cap: int | None = None,
) -> list[str]:
    """Why a person should read this digest closely before trusting it. Empty = no flag.

    The per-pick fields come from the digest team (multi mode): `claims_judged`,
    `claims_supported`, `claims_cut`, `unverified`. In single mode nothing is verified, so only
    the input-side reasons (held items, quiet sources) can fire.
    """
    reasons = []
    judged = sum(p.get("claims_judged", 0) for p in picks)
    supported = sum(p.get("claims_supported", 0) for p in picks)
    if judged and supported / judged < min_support:
        reasons.append(
            f"the Verifier found only {supported / judged:.0%} of claims supported in the final"
            f" drafts (threshold {min_support:.0%})"
        )
    cut = [p["item_id"] for p in picks if p.get("claims_cut")]
    if len(cut) > max_cut:
        reasons.append(f"{len(cut)} summaries lost unsupported claims ({', '.join(cut)})")
    checked = judged + sum(p.get("unverified", 0) for p in picks)
    unverified = sum(p.get("unverified", 0) for p in picks)
    if checked and unverified / checked > max_unverified:
        reasons.append(f"the Verifier gave no readable verdict on {unverified} of {checked} claims")
    if critic_cap is not None and (stats or {}).get("critic_drops", 0) >= critic_cap:
        reasons.append("the Critic hit its drop limit, so some kept items carry its concerns")
    if held:
        reasons.append(f"{held} item(s) held back for instruction-like text (`radar flagged`)")
    if quiet:
        reasons.append(f"no items from {', '.join(quiet)} on the last two runs")
    return reasons


# ---- quiet sources -------------------------------------------------------------------------


def quiet_sources(runs: list[dict], streak: int = 2) -> list[str]:
    """Sources whose last `streak` runs (on different days) fetched nothing or failed.

    runs: {date, source, fetched, error} rows, any order. "Fetched" counts what the source
    returned, before de-duplication, so a normal day with nothing *new* doesn't count."""
    by_source: dict[str, list[dict]] = {}
    for r in sorted(runs, key=lambda r: r["date"], reverse=True):
        by_source.setdefault(r["source"], []).append(r)
    out = []
    for source, rows in sorted(by_source.items()):
        recent = rows[:streak]
        if len(recent) == streak and all(r["fetched"] == 0 or r.get("error") for r in recent):
            out.append(source)
    return out


# ---- the Critic vs. your ratings -----------------------------------------------------------


def agreement(rows: list[dict]) -> dict:
    """rows: {decision: keep|drop, rating: useful|not_relevant}. Agreement overall and per
    decision: how far the Critic's filtering matches the person it filters for."""
    agree = {("keep", "useful"), ("drop", "not_relevant")}

    def rate(sub: list[dict]) -> float | None:
        return (
            round(sum((r["decision"], r["rating"]) in agree for r in sub) / len(sub), 3)
            if sub
            else None
        )

    keeps = [r for r in rows if r["decision"] == "keep"]
    drops = [r for r in rows if r["decision"] == "drop"]
    return {
        "n": len(rows),
        "agreement": rate(rows),
        "keep_n": len(keeps),
        "keep_agreement": rate(keeps),  # kept items you found useful
        "drop_n": len(drops),
        "drop_agreement": rate(drops),  # dropped items you also found not relevant
    }
