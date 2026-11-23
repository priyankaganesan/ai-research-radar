-- Web UI: remember which conversation each question belongs to, and the answer it got, so the
-- browser can list past conversations and reopen one. (The LangGraph checkpointer also has the
-- messages, but keyed by thread only, with no list of threads; this keeps it in our own store.)

ALTER TABLE questions ADD COLUMN thread_id TEXT;
ALTER TABLE questions ADD COLUMN answer TEXT;

CREATE INDEX questions_thread ON questions (thread_id);
