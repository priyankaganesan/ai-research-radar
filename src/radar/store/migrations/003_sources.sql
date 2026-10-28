-- M3: signals from other sources, and Semantic Scholar's one-sentence summaries.

ALTER TABLE items ADD COLUMN tldr TEXT;  -- Semantic Scholar TLDR, when it has one

-- One row per item per day, so attention over time stays visible (citation velocity in M2
-- is the change in `citations` between rows). A source fills only its own columns.
CREATE TABLE signals (
    item_id                TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    date                   TEXT NOT NULL,  -- ISO date the numbers were read
    hf_upvotes             INTEGER,
    citations              INTEGER,
    influential_citations  INTEGER,
    hn_points              INTEGER,
    hn_comments            INTEGER,
    PRIMARY KEY (item_id, date)
);
