"""What a rating means for the profile (M4): pure functions, no LLM, no database.

Three things learn from ratings, each in code so it is predictable and testable:

1. The profile vector (what "relevant" means). Rocchio relevance feedback, a 1971 idea from
   information retrieval that still works:

       profile = a * mean(seed topics) + b * mean(items rated useful) - c * mean(not relevant)

   With a=1, b=0.75, c=0.25 (the textbook values) the seed topics stay the anchor, useful items
   pull the profile toward them, and not-relevant items push it away, more gently.

2. The scoring weights (how much relevance vs. attention vs. citations vs. recency counts).
   Each digest pick remembers its score parts. A useful rating moves the weights a small step
   *toward* the mix that scored that item; a not-relevant rating moves them a step *away*.
   Weights stay inside bounds and always add up to 1, so no single rating can swing the digest.

3. The reserved slots (early buzz, lab launch). Every few ratings on reserved-slot items is a
   "vote": mostly useful -> one more reserved slot (max 3); mostly not -> one fewer (min 1).
   The slot type that is rated useful more often is filled first.
"""

from radar.scoring import DEFAULT_WEIGHTS, mean_vector

# ---- 1. profile vector ---------------------------------------------------------------------

ROCCHIO = {"alpha": 1.0, "beta": 0.75, "gamma": 0.25}


def rocchio(
    topic_vecs: list[list[float]],
    useful_vecs: list[list[float]],
    not_relevant_vecs: list[list[float]],
    alpha: float = ROCCHIO["alpha"],
    beta: float = ROCCHIO["beta"],
    gamma: float = ROCCHIO["gamma"],
) -> list[float]:
    """The interest profile as one vector. Any of the three groups may be empty."""
    groups = [(alpha, topic_vecs), (beta, useful_vecs), (-gamma, not_relevant_vecs)]
    dim = next((len(v[0]) for _, v in groups if v), 0)
    if not dim:
        return []
    out = [0.0] * dim
    for coef, vecs in groups:
        if vecs:
            for i, x in enumerate(mean_vector(vecs)):
                out[i] += coef * x
    return out


# ---- 2. scoring weights --------------------------------------------------------------------

WEIGHT_STEP = 0.03  # how far one rating moves the weights (3% of the way toward the item's mix)
WEIGHT_BOUNDS = (0.05, 0.60)  # no term can vanish or take over


def nudge_weights(weights: dict[str, float], parts: dict[str, float], value: str) -> dict:
    """One small step toward (useful) or away from (not_relevant) the item's score mix.

    `parts` are the weighted contributions of each term to that item's score. Their shares
    say what got the item into the digest: e.g. 70% relevance, 20% attention, 10% recency.
    """
    sign = {"useful": 1.0, "not_relevant": -1.0}.get(value)
    total = sum(parts.values()) if parts else 0.0
    if sign is None or total <= 0:
        return dict(weights)  # "skipped" or no score on record: nothing to learn
    new = {k: w + sign * WEIGHT_STEP * parts.get(k, 0.0) / total for k, w in weights.items()}
    return bounded(new)


def bounded(weights: dict[str, float], bounds=WEIGHT_BOUNDS) -> dict[str, float]:
    """Clip each weight to the bounds and rescale so they add up to 1.

    Rescaling can push a weight back past a bound, so repeat a few times; it settles fast.
    """
    lo, hi = bounds
    w = dict(weights)
    for _ in range(10):
        w = {k: min(max(v, lo), hi) for k, v in w.items()}
        total = sum(w.values())
        w = {k: v / total for k, v in w.items()}
        if all(lo - 1e-9 <= v <= hi + 1e-9 for v in w.values()):
            break
    return {k: round(v, 4) for k, v in w.items()}


def default_weights() -> dict[str, float]:
    return dict(DEFAULT_WEIGHTS)


# ---- 3. reserved slots ---------------------------------------------------------------------

RESERVED_SLOTS = ("early_buzz", "lab_launch")
RESERVED_MIN, RESERVED_MAX = 1, 3
SLOT_VOTE_EVERY = 4  # ratings on reserved items per vote
SLOT_UP_RATE = 0.6  # useful share at or above this adds a slot
SLOT_DOWN_RATE = 0.25  # useful share at or below this removes one


def update_reserved(reserved: int, tally: dict, values: list[str]) -> tuple[int, dict, dict | None]:
    """Count reserved-slot ratings; every SLOT_VOTE_EVERY of them, maybe move the count by one.

    "skipped" counts as not useful here: a reserved slot the reader keeps skipping isn't
    earning its place. Returns (reserved, tally, change or None).
    """
    tally = {"useful": tally.get("useful", 0), "rated": tally.get("rated", 0)}
    change = None
    for v in values:
        tally["rated"] += 1
        tally["useful"] += v == "useful"
        if tally["rated"] >= SLOT_VOTE_EVERY:
            rate = tally["useful"] / tally["rated"]
            new = reserved
            if rate >= SLOT_UP_RATE:
                new = min(reserved + 1, RESERVED_MAX)
            elif rate <= SLOT_DOWN_RATE:
                new = max(reserved - 1, RESERVED_MIN)
            if new != reserved:
                change = {"from": reserved, "to": new, "useful_rate": round(rate, 2)}
            reserved = new
            tally = {"useful": 0, "rated": 0}
    return reserved, tally, change


def update_slot_stats(stats: dict, slot: str, value: str) -> dict:
    """Running useful/rated counts per reserved slot type."""
    s = {k: dict(v) for k, v in stats.items()}
    entry = s.setdefault(slot, {"useful": 0, "rated": 0})
    entry["rated"] += 1
    entry["useful"] += value == "useful"
    return s


def slot_order(stats: dict) -> list[str]:
    """Reserved slot types, most useful first. (useful + 1) / (rated + 2) is a smoothed rate:
    with no ratings every slot scores 0.5, and one lucky rating can't dominate."""

    def rate(slot: str) -> float:
        e = stats.get(slot, {})
        return (e.get("useful", 0) + 1) / (e.get("rated", 0) + 2)

    return sorted(RESERVED_SLOTS, key=lambda s: (-rate(s), RESERVED_SLOTS.index(s)))


def reserved_plan(reserved: int, order: list[str] | None = None) -> list[str]:
    """Which reserved slot each of the `reserved` places goes to, e.g. 3 ->
    [early_buzz, lab_launch, early_buzz]: the third goes to the more useful type again."""
    order = list(order or RESERVED_SLOTS)
    return [order[i % len(order)] for i in range(reserved)]
