-- M2: the daily digest, its summaries, and the interest profile the scoring reads.

CREATE TABLE digests (
    date     TEXT NOT NULL,                     -- ISO date of the digest
    item_id  TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    rank     INTEGER NOT NULL,
    slot     TEXT NOT NULL CHECK (slot IN ('main', 'early_buzz', 'lab_launch')),
    score    REAL NOT NULL,
    reason   TEXT NOT NULL,
    PRIMARY KEY (date, item_id)
);

-- One summary per item per writer ("single" = M2's one-step summarize, "multi" = M2.5 agents),
-- so the two can be compared side by side.
CREATE TABLE summaries (
    item_id     TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    mode        TEXT NOT NULL DEFAULT 'single',
    text        TEXT NOT NULL,
    model       TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (item_id, mode)
);

-- Scoring weights and seed topics. Empty means "use the defaults"; M4 ratings will write here.
CREATE TABLE profile (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL  -- JSON
);
