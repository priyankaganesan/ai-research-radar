-- Safety: the records behind the guardrails in radar/safety.py and the review flags.

-- One row per source per collection day: did the source return anything? Two empty or failed
-- days in a row means it's probably broken, not quiet (`safety.quiet_sources`).
CREATE TABLE source_runs (
    date     TEXT NOT NULL,
    source   TEXT NOT NULL,           -- arxiv | hf | labs | hn | s2
    fetched  INTEGER NOT NULL DEFAULT 0,
    added    INTEGER NOT NULL DEFAULT 0,
    error    TEXT,                    -- set when the source was unavailable
    PRIMARY KEY (date, source)
);

-- Items whose text reads like instructions to the model (possible prompt injection).
-- Held items stay out of the digest until a person releases them.
CREATE TABLE flagged_items (
    item_id       TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    flagged_at    TEXT NOT NULL DEFAULT (datetime('now')),
    snippet       TEXT NOT NULL,
    about_attacks INTEGER NOT NULL DEFAULT 0,  -- 1 = research about attacks: flagged, not held
    released_at   TEXT                         -- a person looked and let it through
);

-- Why a digest was marked "needs review" (no row = not flagged).
CREATE TABLE digest_reviews (
    date     TEXT PRIMARY KEY,
    reasons  TEXT NOT NULL  -- JSON list of sentences
);

-- What the Critic decided on each item, so its decisions can be compared with your ratings.
CREATE TABLE critic_decisions (
    date      TEXT NOT NULL,
    item_id   TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    decision  TEXT NOT NULL CHECK (decision IN ('keep', 'drop')),
    concerns  TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (date, item_id)
);

-- Topic removals the Profile agent proposed. They wait here until you approve or reject them:
-- a removed topic hides papers, and you wouldn't notice what you no longer see.
CREATE TABLE profile_pending (
    id          INTEGER PRIMARY KEY,
    proposed_at TEXT NOT NULL DEFAULT (datetime('now')),
    action      TEXT NOT NULL DEFAULT 'remove',
    topic       TEXT NOT NULL,
    why         TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    decided_at  TEXT
);
