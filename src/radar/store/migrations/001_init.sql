-- Knowledge store: papers, their chunks, and the two search indexes over chunks.

CREATE TABLE papers (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    title       TEXT NOT NULL,
    abstract    TEXT NOT NULL,
    authors     TEXT NOT NULL DEFAULT '[]',  -- JSON list
    published   TEXT NOT NULL,               -- ISO date
    url         TEXT NOT NULL,
    pdf_url     TEXT,
    categories  TEXT NOT NULL DEFAULT '[]',  -- JSON list
    added_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE chunks (
    id           INTEGER PRIMARY KEY,
    paper_id     TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL CHECK (kind IN ('abstract', 'section')),
    section      TEXT,
    text         TEXT NOT NULL,
    token_count  INTEGER NOT NULL
);

CREATE INDEX chunks_paper_id ON chunks(paper_id);

-- Vector index, keyed by chunks.id
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[{EMBED_DIM}]
);

-- Keyword index, keyed by chunks.id (rowid)
CREATE VIRTUAL TABLE chunks_fts USING fts5(title, text, tokenize = 'porter unicode61');
