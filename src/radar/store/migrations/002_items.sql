-- M1: generalize papers into items (papers, posts, videos, podcasts), and let chunks carry
-- transcript timestamps. SQLite can't change a CHECK constraint in place, so chunks is rebuilt.

ALTER TABLE papers RENAME TO items;
ALTER TABLE items ADD COLUMN kind TEXT NOT NULL DEFAULT 'paper'
    CHECK (kind IN ('paper', 'post', 'video', 'podcast'));

CREATE TABLE chunks_new (
    id           INTEGER PRIMARY KEY,
    item_id      TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL CHECK (kind IN ('abstract', 'section', 'transcript')),
    section      TEXT,
    text         TEXT NOT NULL,
    token_count  INTEGER NOT NULL,
    start_s      REAL,  -- transcript chunks only
    end_s        REAL
);

INSERT INTO chunks_new (id, item_id, kind, section, text, token_count)
    SELECT id, paper_id, kind, section, text, token_count FROM chunks;

DROP TABLE chunks;
ALTER TABLE chunks_new RENAME TO chunks;
CREATE INDEX chunks_item_id ON chunks(item_id);
CREATE INDEX items_published ON items(published);
