"""Long-term memory at work (M4): ratings and gaps update the interest profile.

`update_profile` runs at the start of every `radar daily` (and on `radar profile --update`).
It reads the ratings that arrived since the last run and changes, in order:

1. scoring weights   small bounded steps toward/away from each rated pick's score mix (code)
2. reserved slots    every few reserved-slot ratings, maybe one more or one fewer (code)
3. interest topics   the Profile agent reads titles and gaps and proposes edits (LLM + guard)

and the profile vector follows from the ratings themselves each time it's built (Rocchio, see
`feedback.py`). Everything is stored in the `profile` table, and every change is written to
`profile_changes` with its reason, so `radar profile` can show what was learned and why.

Also here: the early-buzz watchlist re-check and collecting papers for open gaps.
"""

import struct
from dataclasses import dataclass, field
from datetime import date, timedelta

from langchain_core.language_models import BaseChatModel

from radar import feedback
from radar.context import AppContext
from radar.store import digests as digest_store
from radar.store import memory
from radar.store import safety as safety_store
from radar.store import signals as signal_store

PROFILE_AGENT_MIN_RATINGS = 3  # useful/not-relevant ratings needed before the agent runs

# Watchlist: re-check after 2 weeks; still unproven -> look again weekly, drop after 4 weeks.
WATCH_FIRST_CHECK_DAYS = 14
WATCH_GIVE_UP_DAYS = 28
ESTABLISHED_MIN_CITATIONS = 10  # new citations since it was added
GAP_MAX_ATTEMPTS = 3  # collection runs that search for one gap before it's left alone


# ---- reading the profile -------------------------------------------------------------------


def seed_topics(ctx: AppContext) -> list[str]:
    return [t.strip() for t in ctx.settings.interest_topics.split(",") if t.strip()]


@dataclass
class Profile:
    """Everything the digest reads from the `profile` table, with defaults filled in."""

    topics: list[str]
    weights: dict[str, float]
    reserved_slots: int
    slot_order: list[str]
    rated_seq: int = 0  # last rating already applied
    slot_tally: dict = field(default_factory=dict)  # reserved-slot ratings since the last vote
    slot_stats: dict = field(default_factory=dict)  # all-time useful/rated per reserved slot
    gap_seen: int = 0  # last gap id the Profile agent has seen


def load_profile(ctx: AppContext) -> Profile:
    p = digest_store.get_profile(ctx.conn)
    return Profile(
        topics=p.get("topics") or seed_topics(ctx),
        weights=p.get("weights") or feedback.default_weights(),
        reserved_slots=p.get("reserved_slots", ctx.settings.digest_reserved),
        slot_order=p.get("slot_order") or list(feedback.RESERVED_SLOTS),
        rated_seq=p.get("rated_seq", 0),
        slot_tally=p.get("slot_tally", {}),
        slot_stats=p.get("slot_stats", {}),
        gap_seen=p.get("gap_seen", 0),
    )


def save_profile(ctx: AppContext, prof: Profile) -> None:
    stored = digest_store.get_profile(ctx.conn)
    for key, value in vars(prof).items():
        if key == "topics" and "topics" not in stored and value == seed_topics(ctx):
            continue  # unchanged seeds stay unstored, so editing INTEREST_TOPICS still works
        digest_store.set_profile(ctx.conn, key, value)


def reset_profile(ctx: AppContext) -> None:
    """Back to the defaults. Ratings stay in the store, but are not re-applied to weights."""
    seq = ctx.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM ratings").fetchone()[0]
    with ctx.conn:
        ctx.conn.execute("DELETE FROM profile")
        digest_store.set_profile(ctx.conn, "rated_seq", seq)
        memory.log_change(ctx.conn, "reset", {"rated_seq": seq})


# ---- the profile vector --------------------------------------------------------------------


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
    """The interest profile as one vector (Rocchio): seed topics, plus items rated useful,
    minus items rated not relevant. Each item counts once, with its latest rating."""
    topics = load_profile(ctx).topics
    if not topics:
        return []
    rated = memory.latest_ratings(ctx.conn)
    vecs = item_vectors(ctx, list(rated))
    useful = [vecs[i] for i, v in rated.items() if v == "useful" and i in vecs]
    not_rel = [vecs[i] for i, v in rated.items() if v == "not_relevant" and i in vecs]
    return feedback.rocchio(ctx.embeddings.embed_documents(topics), useful, not_rel)


# ---- the update run ------------------------------------------------------------------------


@dataclass
class ProfileUpdate:
    """What one update did, for the CLI, the daily log and the walkthrough."""

    new_ratings: int = 0
    weights_before: dict = field(default_factory=dict)
    weights_after: dict = field(default_factory=dict)
    reserved_change: dict | None = None
    slot_order: list[str] = field(default_factory=list)
    agent_ran: bool = False
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)  # removals waiting for your approval
    why: str = ""

    def lines(self) -> list[str]:
        if not self.new_ratings and not self.agent_ran:
            return ["profile: no new ratings or gaps"]
        out = [f"profile: {self.new_ratings} new rating(s)"]
        if self.weights_after != self.weights_before:
            moved = ", ".join(
                f"{k} {self.weights_before[k]:.3f}->{v:.3f}" for k, v in self.weights_after.items()
            )
            out.append(f"  weights: {moved}")
        if self.reserved_change:
            c = self.reserved_change
            out.append(
                f"  reserved slots: {c['from']} -> {c['to']} (useful rate {c['useful_rate']})"
            )
        if self.agent_ran:
            what = []
            if self.added:
                what.append(f"added {self.added}")
            if self.removed:
                what.append(f"removed {self.removed}")
            out.append(f"  profile agent: {', '.join(what) or 'no change'}. {self.why}".rstrip())
            for t in self.pending:
                out.append(f"    waiting for you: remove {t!r} (`radar profile` to approve)")
            for r in self.rejected:
                out.append(f"    refused: {r}")
        return out


def update_profile(ctx: AppContext, llm: BaseChatModel | None = None) -> ProfileUpdate:
    """Apply new ratings (and, with `llm`, run the Profile agent). Safe to run any time:
    each rating is applied once, tracked by its `seq` number."""
    prof = load_profile(ctx)
    new = memory.ratings_since(ctx.conn, prof.rated_seq)
    report = ProfileUpdate(
        new_ratings=len(new), weights_before=dict(prof.weights), slot_order=prof.slot_order
    )

    # 1. weights
    weights = dict(prof.weights)
    for r in new:
        if r["parts"]:
            weights = feedback.nudge_weights(weights, r["parts"], r["value"])

    # 2. reserved slots
    reserved_values = [r["value"] for r in new if r["slot"] in feedback.RESERVED_SLOTS]
    stats = prof.slot_stats
    for r in new:
        if r["slot"] in feedback.RESERVED_SLOTS:
            stats = feedback.update_slot_stats(stats, r["slot"], r["value"])
    reserved, tally, change = feedback.update_reserved(
        prof.reserved_slots, prof.slot_tally, reserved_values
    )
    order = feedback.slot_order(stats)

    # 3. topics, by the Profile agent
    gaps = memory.open_gaps(ctx.conn)
    new_gaps = [g for g in gaps if g["id"] > prof.gap_seen]
    judged = [r for r in new if r["value"] in ("useful", "not_relevant")]
    topics = prof.topics
    if llm is not None and (len(judged) >= PROFILE_AGENT_MIN_RATINGS or new_gaps):
        from radar.graphs.agents.profile import build_profile_agent

        out = build_profile_agent(llm).invoke(
            {
                "topics": topics,
                "useful": [r["title"] for r in judged if r["value"] == "useful"],
                "not_relevant": [r["title"] for r in judged if r["value"] == "not_relevant"],
                "gaps": [g["question"] or g["topic"] for g in gaps],
            },
            config={"run_name": "radar-profile-agent"},
        )
        topics = out["new_topics"]
        report.agent_ran = True
        report.added, report.removed = out["added"], out["removed"]
        report.rejected, report.why = out["rejected"], out["why"]
        if ctx.settings.profile_confirm_removals and report.removed:
            # Safety: a removal hides papers, and you'd never see what you no longer see. So it
            # waits for a person (`radar profile --approve ID`) instead of applying now.
            topics = list(prof.topics) + report.added
            report.pending, report.removed = report.removed, []

    prof.weights, prof.reserved_slots, prof.slot_order = weights, reserved, order
    prof.slot_tally, prof.slot_stats, prof.topics = tally, stats, topics
    prof.rated_seq = max([r["seq"] for r in new], default=prof.rated_seq)
    prof.gap_seen = max([g["id"] for g in gaps], default=prof.gap_seen)
    report.weights_after, report.reserved_change, report.slot_order = weights, change, order

    with ctx.conn:
        save_profile(ctx, prof)
        if weights != report.weights_before:
            memory.log_change(
                ctx.conn,
                "weights",
                {"from": report.weights_before, "to": weights, "ratings": len(new)},
            )
        if change:
            memory.log_change(ctx.conn, "reserved", {**change, "order": order})
        for t in report.pending:
            safety_store.add_pending(ctx.conn, t, report.why)
        if report.added or report.removed:
            memory.log_change(
                ctx.conn,
                "topics",
                {"added": report.added, "removed": report.removed, "why": report.why},
            )
    return report


# ---- early-buzz watchlist ------------------------------------------------------------------


def watch_early_buzz(ctx: AppContext, picks: list[dict], on: date) -> int:
    """Put today's early-buzz picks on the watchlist. Returns how many are new."""
    before = len(memory.watching_ids(ctx.conn))
    for p in picks:
        if p["slot"] == "early_buzz":
            cites = signal_store.latest(ctx.conn, p["item_id"])["citations"] or 0
            memory.watch(ctx.conn, p["item_id"], on, cites, WATCH_FIRST_CHECK_DAYS)
    return len(memory.watching_ids(ctx.conn)) - before


def recheck_watchlist(ctx: AppContext, on: date) -> list[str]:
    """Papers past their re-check date: did the buzz turn into citations?

    established: at least ESTABLISHED_MIN_CITATIONS new citations since it was added.
    still unproven before WATCH_GIVE_UP_DAYS: look again in a week.
    unproven after that: dropped (it stays in the store and stays searchable).
    Citations come from the Semantic Scholar enrich step, which also covers watched papers.
    """
    log = []
    with ctx.conn:
        for w in memory.due_for_recheck(ctx.conn, on):
            cites = signal_store.latest(ctx.conn, w["item_id"])["citations"] or 0
            gained = cites - w["citations_at_add"]
            age = (on - date.fromisoformat(w["added_at"])).days
            note = f"{gained:+d} citations in {age} days ({cites} total)"
            if gained >= ESTABLISHED_MIN_CITATIONS:
                memory.set_watch_status(ctx.conn, w["item_id"], "established", on, note)
                log.append(f"established: {w['item_id']} {note}")
            elif age >= WATCH_GIVE_UP_DAYS:
                memory.set_watch_status(ctx.conn, w["item_id"], "dropped", on, note)
                log.append(f"dropped: {w['item_id']} {note}")
            else:
                again = on + timedelta(days=7)
                memory.set_watch_status(ctx.conn, w["item_id"], "watching", on, note, again)
                log.append(f"still watching: {w['item_id']} {note}, next check {again}")
    return log
