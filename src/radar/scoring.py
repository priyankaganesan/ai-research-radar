"""Ranking for the daily digest: pure functions, no LLM, no database.

    score = w_rel * relevance          cosine(item vector, profile vector), 0..1
          + w_att * attention          HF upvotes + HN points, log-scaled, 0..1 within today's pool
          + w_cit * citation_velocity  new citations per day, log-scaled, 0..1 within the pool
          + w_rec * recency            exp(-age_days / 7)

Why code and not the LLM: a formula gives the same order every run, can be tested, and every
part of it can be shown in the digest's "why" line. Asking a 9B model to rank 300 abstracts
would be slow, and the order would change from run to run.

The two reserved slots answer Checkpoint 2 feedback: popularity alone shouldn't fill the digest,
and new work with few citations should still get a look.
"""

import math
import re
from dataclasses import dataclass, field

DEFAULT_WEIGHTS = {"relevance": 0.45, "attention": 0.25, "citations": 0.10, "recency": 0.20}
EARLY_BUZZ_MAX_CITATIONS = 5  # "few citations so far"
# Lab blogs also publish customer stories ("Merge builds ... with OpenAI"); those aren't launches.
NOT_A_LAUNCH = re.compile(
    r"\b(builds?|built|uses?|powers?|scales?|transforms?)\b.+\bwith\b|customer stor|case stud",
    re.IGNORECASE,
)


@dataclass
class Features:
    """The raw facts about one candidate, gathered from the store by the daily graph."""

    item_id: str
    kind: str
    source: str
    age_days: float
    relevance: float  # cosine similarity to the profile, -1..1
    hf_upvotes: int = 0
    hn_points: int = 0
    citations: int = 0
    citation_velocity: float = 0.0  # citations gained per day
    title: str = ""


@dataclass
class Scored:
    features: Features
    total: float
    parts: dict[str, float] = field(default_factory=dict)  # weighted contribution of each term


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """The average direction of several vectors (each normalized first, so none dominates)."""
    if not vectors:
        return []
    dim = len(vectors[0])
    total = [0.0] * dim
    for v in vectors:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        for i, x in enumerate(v):
            total[i] += x / n
    return [x / len(vectors) for x in total]


def recency(age_days: float, half_life_ish: float = 7.0) -> float:
    return math.exp(-max(age_days, 0.0) / half_life_ish)


def attention_raw(f: Features) -> float:
    """Upvotes and HN points on one log scale. HN points count half: they're easier to earn."""
    return math.log1p(f.hf_upvotes + 0.5 * f.hn_points)


def score_all(candidates: list[Features], weights: dict[str, float] | None = None) -> list[Scored]:
    """Score every candidate, highest first.

    Attention and citations are normalized against the best in today's pool, so a quiet day
    still has a top item and the weights mean the same thing every day.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    max_att = max((attention_raw(f) for f in candidates), default=0.0) or 1.0
    max_cit = max((math.log1p(f.citation_velocity) for f in candidates), default=0.0) or 1.0
    out = []
    for f in candidates:
        parts = {
            "relevance": w["relevance"] * max(f.relevance, 0.0),
            "attention": w["attention"] * attention_raw(f) / max_att,
            "citations": w["citations"] * math.log1p(f.citation_velocity) / max_cit,
            "recency": w["recency"] * recency(f.age_days),
        }
        out.append(Scored(f, round(sum(parts.values()), 4), parts))
    return sorted(out, key=lambda s: (-s.total, s.features.item_id))


@dataclass
class Pick:
    scored: Scored
    slot: str  # "main" | "early_buzz" | "lab_launch"
    reason: str


def pick(scored: list[Scored], n_main: int = 6, reserved: int = 2) -> list[Pick]:
    """Choose the digest: the reserved slots first, then the best of the rest.

    - early_buzz: the most-upvoted *paper* with few citations so far
    - lab_launch: the best-scoring post from an AI lab blog
    A reserved slot with no candidate goes to the main list, so the digest stays full.
    """
    chosen: list[Pick] = []
    taken: set[str] = set()

    def take(s: Scored, slot: str) -> None:
        chosen.append(Pick(s, slot, explain(s, slot)))
        taken.add(s.features.item_id)

    if reserved >= 1:
        buzz = slot_candidates(scored, "early_buzz")
        if buzz:
            take(buzz[0], "early_buzz")
    if reserved >= 2:
        labs = slot_candidates(scored, "lab_launch")
        if labs:
            take(labs[0], "lab_launch")

    total = n_main + reserved
    for s in scored:
        if len(chosen) >= total:
            break
        if s.features.item_id not in taken:
            take(s, "main")
    # Reserved picks first in the list would bury the best main item: show main first.
    order = {"main": 0, "early_buzz": 1, "lab_launch": 2}
    return sorted(chosen, key=lambda p: (order[p.slot], -p.scored.total))


def slot_candidates(scored: list[Scored], slot: str) -> list[Scored]:
    """Everything eligible for a slot, best first. Used to fill the slot and, in M2.5, to
    replace a pick the Critic drops with another item that fits the same slot."""
    if slot == "early_buzz":
        buzz = [
            s
            for s in scored
            if s.features.kind == "paper"
            and s.features.citations <= EARLY_BUZZ_MAX_CITATIONS
            and attention_raw(s.features) > 0
        ]
        return sorted(buzz, key=lambda s: (-attention_raw(s.features), -s.total))
    if slot == "lab_launch":
        return [
            s
            for s in scored
            if s.features.source.startswith("lab:") and not NOT_A_LAUNCH.search(s.features.title)
        ]
    return list(scored)  # main: anything, by score


def explain(s: Scored, slot: str) -> str:
    """The one-line "why" shown in the digest, built from the score's parts."""
    f = s.features
    bits = [f"relevance {max(f.relevance, 0):.2f}"]
    if f.hf_upvotes:
        bits.append(f"{f.hf_upvotes} HF upvotes")
    if f.hn_points:
        bits.append(f"{f.hn_points} HN points")
    if f.citations:
        bits.append(f"{f.citations} citations")
    bits.append("today" if f.age_days < 1 else f"{f.age_days:.0f} days old")
    lead = {
        "main": "Top score",
        "early_buzz": "Early buzz: lots of attention, few citations yet",
        "lab_launch": "Lab launch",
    }[slot]
    top = max(s.parts, key=s.parts.get)
    return f"{lead} ({s.total:.2f}, mostly {top}): " + ", ".join(bits)
