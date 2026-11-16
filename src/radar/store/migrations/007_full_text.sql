-- M5: full texts. One row per item whose full text was fetched (or tried). The text itself is
-- kept too, so it can be re-chunked with other sizes without fetching again; its chunks live in
-- `chunks` with kind = 'section', next to the item's abstract chunk.

CREATE TABLE full_texts (
    item_id     TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    status      TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
    source      TEXT,              -- 'arxiv_html' | 'arxiv_pdf' | 'html'
    url         TEXT,
    sections    TEXT,              -- JSON [{"heading": ..., "text": ...}]
    n_sections  INTEGER NOT NULL DEFAULT 0,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    chars       INTEGER NOT NULL DEFAULT 0,
    reason      TEXT,              -- why it was fetched: 'asked 2 times', 'agent tool', 'manual'
    error       TEXT,              -- for status 'failed'
    fetched_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
