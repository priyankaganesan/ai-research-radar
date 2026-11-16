-- Tracing: a local log of every traced call (services, MCP tools, source fetchers, agent tools).
-- Written by radar/tracing.py, read by `radar calls`. Works with LangSmith off; when LangSmith is
-- on, the two langsmith_* columns link a row to its run in the LangSmith UI.

CREATE TABLE tool_calls (
    id                  TEXT PRIMARY KEY,           -- short random ID
    parent_id           TEXT,                       -- the traced call this one ran inside
    root_id             TEXT NOT NULL,              -- the outermost call (= id for a top-level call)
    name                TEXT NOT NULL,              -- e.g. services.ingest_hf, mcp.get_item
    kind                TEXT NOT NULL,              -- service | source | mcp | agent_tool
    args                TEXT NOT NULL,              -- JSON, secrets and plumbing removed
    summary             TEXT,                       -- short description of the result
    error               TEXT,                       -- "ExceptionType: message", NULL if it worked
    started_at          TEXT NOT NULL,              -- ISO timestamp, UTC
    duration_ms         REAL NOT NULL,
    langsmith_trace_id  TEXT,
    langsmith_run_id    TEXT
);

CREATE INDEX tool_calls_started ON tool_calls (started_at);
CREATE INDEX tool_calls_root ON tool_calls (root_id);
