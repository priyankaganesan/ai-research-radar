"""M4's pure feedback rules: profile vector, weight steps, reserved slots. No store, no model."""

import pytest

from radar import feedback, scoring
from radar.graphs.agents.profile import MAX_TOPICS, apply_proposal

W = feedback.default_weights()
REL_HEAVY = {"relevance": 0.40, "attention": 0.05, "citations": 0.0, "recency": 0.05}


def test_rocchio_pulls_toward_useful_and_away_from_not_relevant():
    topic, liked, disliked = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    base = feedback.rocchio([topic], [], [])
    moved = feedback.rocchio([topic], [liked], [disliked])
    assert scoring.cosine(moved, liked) > scoring.cosine(base, liked)
    assert scoring.cosine(moved, disliked) < 0  # pushed away
    assert scoring.cosine(moved, topic) > 0.5  # the seed topics stay the anchor


def test_rocchio_with_nothing_is_empty():
    assert feedback.rocchio([], [], []) == []


def test_useful_moves_weights_toward_the_items_mix():
    new = feedback.nudge_weights(W, REL_HEAVY, "useful")
    assert new["relevance"] > W["relevance"]
    assert new["attention"] < W["attention"]
    assert sum(new.values()) == pytest.approx(1.0, abs=1e-3)


def test_not_relevant_moves_weights_away():
    new = feedback.nudge_weights(W, REL_HEAVY, "not_relevant")
    assert new["relevance"] < W["relevance"]
    assert sum(new.values()) == pytest.approx(1.0, abs=1e-3)


def test_one_step_is_small():
    new = feedback.nudge_weights(W, REL_HEAVY, "useful")
    assert max(abs(new[k] - W[k]) for k in W) < 0.03


def test_skipped_or_missing_parts_change_nothing():
    assert feedback.nudge_weights(W, REL_HEAVY, "skipped") == W
    assert feedback.nudge_weights(W, {}, "useful") == W


def test_weights_stay_in_bounds_after_many_ratings():
    w = dict(W)
    for _ in range(300):
        w = feedback.nudge_weights(w, REL_HEAVY, "useful")
    lo, hi = feedback.WEIGHT_BOUNDS
    assert all(lo - 1e-3 <= v <= hi + 1e-3 for v in w.values())
    assert w["relevance"] > hi - 0.02  # pressed against the upper bound, not past it
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-3)


def test_reserved_goes_up_after_a_useful_vote():
    reserved, tally, change = feedback.update_reserved(2, {}, ["useful"] * 3 + ["skipped"])
    assert reserved == 3 and change == {"from": 2, "to": 3, "useful_rate": 0.75}
    assert tally == {"useful": 0, "rated": 0}  # a vote resets the count


def test_reserved_goes_down_and_respects_bounds():
    r, _, _ = feedback.update_reserved(2, {}, ["not_relevant"] * 4)
    assert r == 1
    r, _, change = feedback.update_reserved(1, {}, ["not_relevant"] * 4)
    assert r == 1 and change is None  # already at the minimum
    r, _, _ = feedback.update_reserved(3, {}, ["useful"] * 8)
    assert r == 3


def test_reserved_waits_for_enough_ratings():
    r, tally, change = feedback.update_reserved(2, {}, ["useful", "useful"])
    assert r == 2 and change is None and tally == {"useful": 2, "rated": 2}
    r, _, change = feedback.update_reserved(r, tally, ["useful", "useful"])
    assert r == 3  # the tally carries over between runs


def test_mixed_ratings_keep_the_count():
    r, _, change = feedback.update_reserved(2, {}, ["useful", "not_relevant"] * 2)
    assert r == 2 and change is None


def test_slot_order_and_plan():
    stats = {}
    for v in ["useful", "useful", "useful"]:
        stats = feedback.update_slot_stats(stats, "lab_launch", v)
    stats = feedback.update_slot_stats(stats, "early_buzz", "not_relevant")
    order = feedback.slot_order(stats)
    assert order == ["lab_launch", "early_buzz"]
    assert feedback.reserved_plan(3, order) == ["lab_launch", "early_buzz", "lab_launch"]
    assert feedback.reserved_plan(1, order) == ["lab_launch"]
    assert feedback.slot_order({}) == ["early_buzz", "lab_launch"]  # default without ratings


def test_pick_follows_a_three_slot_plan():
    def f(i, **kw):
        base = dict(item_id=i, kind="paper", source="arxiv", age_days=1, relevance=0.5)
        return scoring.Features(**(base | kw))

    pool = [f(f"p{i}") for i in range(8)]
    pool += [f(f"b{i}", hf_upvotes=100 - i) for i in range(2)]
    pool += [f(f"l{i}", kind="post", source="lab:X", relevance=0.4 - i / 10) for i in range(2)]
    plan = ["lab_launch", "early_buzz", "lab_launch"]
    picks = scoring.pick(scoring.score_all(pool), n_main=5, plan=plan)
    assert len(picks) == 8
    slots = [p.slot for p in picks]
    assert slots.count("lab_launch") == 2 and slots.count("early_buzz") == 1


# ---- the Profile agent's guard ----


def test_guard_applies_small_changes():
    out = apply_proposal(["a b", "c d", "e f"], {"add": ["KV cache compression"], "why": "x"})
    assert out["added"] == ["KV cache compression"]
    assert out["new_topics"][-1] == "KV cache compression"


def test_guard_limits_additions_and_refuses_junk():
    out = apply_proposal(
        ["a b", "c d", "e f"],
        {"add": ["one topic", "two topic", "three topic", "x" * 80, "A B"]},
    )
    assert out["added"] == ["one topic", "two topic"]
    assert len(out["rejected"]) == 3
    assert any("already a topic" in r for r in out["rejected"])


def test_guard_keeps_a_minimum_and_only_removes_existing():
    out = apply_proposal(["a b", "c d", "e f"], {"remove": ["a b"]})
    assert out["removed"] == [] and out["new_topics"] == ["a b", "c d", "e f"]
    out = apply_proposal(["a b", "c d", "e f", "g h"], {"remove": ["G H", "nope", "a b"]})
    assert out["removed"] == ["g h"]  # case-insensitive match; one removal per run
    assert len(out["rejected"]) == 2


def test_guard_caps_the_topic_count():
    full = [f"topic {i}" for i in range(MAX_TOPICS)]
    out = apply_proposal(full, {"add": ["new one"]})
    assert out["added"] == [] and len(out["new_topics"]) == MAX_TOPICS


def test_guard_ignores_a_garbled_proposal():
    out = apply_proposal(["a b", "c d", "e f"], {"add": "not a list"})
    assert out["new_topics"] == ["a b", "c d", "e f"] and not out["added"]
