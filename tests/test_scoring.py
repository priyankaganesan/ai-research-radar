"""Scoring is pure arithmetic, so it gets plain unit tests."""

from radar import scoring
from radar.scoring import Features


def feat(id, **kw) -> Features:
    base = {"kind": "paper", "source": "arxiv", "age_days": 1, "relevance": 0.5}
    return Features(item_id=id, **{**base, **kw})


def total(f: Features, pool: list[Features] | None = None) -> float:
    scored = scoring.score_all(pool or [f])
    return next(s.total for s in scored if s.features.item_id == f.item_id)


def test_each_signal_raises_the_score():
    pool = [feat("ref", hf_upvotes=10, citation_velocity=1.0)]
    base = feat("x")
    assert total(feat("x", relevance=0.9), pool + [feat("x", relevance=0.9)]) > total(
        base, pool + [base]
    )
    up = feat("x", hf_upvotes=10)
    assert total(up, pool + [up]) > total(base, pool + [base])
    cited = feat("x", citation_velocity=1.0)
    assert total(cited, pool + [cited]) > total(base, pool + [base])


def test_recency_decays():
    assert scoring.recency(0) == 1.0
    assert scoring.recency(7) < scoring.recency(1) < scoring.recency(0)


def test_negative_relevance_counts_as_zero():
    assert total(feat("x", relevance=-0.4)) == total(feat("x", relevance=0.0))


def test_score_all_sorts_highest_first():
    out = scoring.score_all([feat("low", relevance=0.1), feat("high", relevance=0.9)])
    assert [s.features.item_id for s in out] == ["high", "low"]


def test_pick_fills_reserved_slots_and_main():
    pool = [feat(f"p{i}", relevance=0.9 - i * 0.05) for i in range(10)]
    pool.append(feat("buzz", relevance=0.1, hf_upvotes=200, citations=0))
    pool.append(feat("post:openai", kind="post", source="lab:OpenAI", relevance=0.2))
    picks = scoring.pick(scoring.score_all(pool), n_main=6, reserved=2)
    slots = {p.scored.features.item_id: p.slot for p in picks}
    assert len(picks) == 8
    assert slots["buzz"] == "early_buzz"
    assert slots["post:openai"] == "lab_launch"
    assert [p.slot for p in picks][:6] == ["main"] * 6
    assert "Early buzz" in next(p.reason for p in picks if p.slot == "early_buzz")


def test_well_cited_paper_is_not_early_buzz():
    pool = [feat("cited", hf_upvotes=300, citations=50), feat("other", relevance=0.9)]
    picks = scoring.pick(scoring.score_all(pool), n_main=6, reserved=2)
    assert all(p.slot == "main" for p in picks)


def test_missing_reserved_candidates_go_to_main():
    pool = [feat(f"p{i}") for i in range(10)]
    picks = scoring.pick(scoring.score_all(pool), n_main=6, reserved=2)
    assert len(picks) == 8 and all(p.slot == "main" for p in picks)


def test_mean_vector_and_cosine():
    v = scoring.mean_vector([[1.0, 0.0], [0.0, 2.0]])
    assert abs(scoring.cosine(v, [1.0, 1.0]) - 1.0) < 1e-9


def test_customer_stories_are_not_lab_launches():
    pool = [
        feat(
            "post:story",
            kind="post",
            source="lab:OpenAI",
            relevance=0.9,
            title="Merge builds product tools with OpenAI technology",
        ),
        feat(
            "post:launch",
            kind="post",
            source="lab:OpenAI",
            relevance=0.3,
            title="Introducing a new reasoning model",
        ),
    ]
    picks = scoring.pick(scoring.score_all(pool), n_main=0, reserved=2)
    assert [p.scored.features.item_id for p in picks if p.slot == "lab_launch"] == ["post:launch"]


def test_hn_self_posts_never_take_main_slots():
    pool = scoring.score_all(
        [
            feat("hn:1", source="hn", kind="post", title="Show HN: my agent app", hn_points=500),
            feat("hn:2", source="hn", kind="post", title="Launch HN: Foo (YC F26)", hn_points=400),
            feat("hn:3", source="hn", kind="post", title="A real story about agents", hn_points=50),
            feat("p1"),
        ]
    )
    ids = [p.scored.features.item_id for p in scoring.pick(pool, n_main=3, plan=[])]
    assert "hn:1" not in ids and "hn:2" not in ids
    assert set(ids) == {"hn:3", "p1"}  # the digest is shorter rather than filled with launches


def test_relevance_floor_puts_on_topic_items_first_and_fills_thin_days():
    pool = scoring.score_all(
        [
            feat("popular-off-topic", relevance=0.45, hf_upvotes=300),
            feat("on-topic-a", relevance=0.6),
            feat("on-topic-b", relevance=0.55),
        ]
    )
    picks = scoring.pick(pool, n_main=2, plan=[], min_relevance=0.5)
    assert [p.scored.features.item_id for p in picks] == ["on-topic-a", "on-topic-b"]
    # a thin day: not enough items clear the floor, so the best one below it fills the slot
    picks = scoring.pick(pool, n_main=3, plan=[], min_relevance=0.5)
    below = [p for p in picks if p.scored.features.item_id == "popular-off-topic"]
    assert below and "below the relevance floor" in below[0].reason
