-- M4: long-term memory and feedback. Ratings, questions, knowledge gaps, the early-buzz
-- watchlist, and a log of every change made to the interest profile.

-- One rating per item per day (rating again the same day replaces it).
CREATE TABLE ratings (
    item_id   TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    date      TEXT NOT NULL,
    value     TEXT NOT NULL CHECK (value IN ('useful', 'not_relevant', 'skipped')),
    slot      TEXT,              -- the digest slot it was shown in, if any (for reserved slots)
    seq       INTEGER NOT NULL,  -- increasing number, so the profile update knows what's new
    PRIMARY KEY (item_id, date)
);

-- Every question asked through `radar ask` (or MCP), answered or not.
CREATE TABLE questions (
    id        INTEGER PRIMARY KEY,
    asked_at  TEXT NOT NULL DEFAULT (datetime('now')),
    text      TEXT NOT NULL,
    answered  INTEGER NOT NULL,  -- 0 = the agent abstained
    cited_ids TEXT NOT NULL DEFAULT '[]'  -- JSON list
);

-- An unanswered question becomes a gap: a topic to collect papers on in the next runs.
CREATE TABLE gaps (
    id           INTEGER PRIMARY KEY,
    question_id  INTEGER REFERENCES questions(id) ON DELETE SET NULL,
    topic        TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    attempts     INTEGER NOT NULL DEFAULT 0,  -- collection runs that searched for it
    resolved_at  TEXT
);

-- Early-buzz picks: lots of attention, few citations. Re-checked after 2 to 4 weeks.
CREATE TABLE watchlist (
    item_id        TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    added_at       TEXT NOT NULL,
    recheck_after  TEXT NOT NULL,
    citations_at_add INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'watching'
                   CHECK (status IN ('watching', 'established', 'dropped')),
    checked_at     TEXT,
    note           TEXT
);

-- What changed in the profile and why: weights, reserved slots, topics.
CREATE TABLE profile_changes (
    id      INTEGER PRIMARY KEY,
    at      TEXT NOT NULL DEFAULT (datetime('now')),
    kind    TEXT NOT NULL,   -- 'weights' | 'reserved' | 'topics' | 'reset'
    detail  TEXT NOT NULL    -- JSON
);

-- The score's parts per digest pick, so a rating can tell which part of the score was right.
ALTER TABLE digests ADD COLUMN parts TEXT;  -- JSON {relevance, attention, citations, recency}
