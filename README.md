# AI Research Radar

A personal agent that keeps me current on AI research. It collects new papers and lab announcements each day, picks the few worth reading, summarizes them, and answers follow-up questions with citations from a local knowledge store.

The agent runs entirely on a local model (Qwen 3.5 9B through Ollama). LangSmith is used for tracing and evals: **local inference, cloud tracing**.

See [PLAN.md](PLAN.md) for the build plan, [CLAUDE.md](CLAUDE.md) for the design rules.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b

uv python install 3.12
uv sync
cp .env.example .env      # then fill in LANGSMITH_API_KEY
uv run radar init-db
```

## Commands

```bash
uv run radar --help
uv run radar config       # show active settings
uv run radar init-db      # create the local store
uv run pytest             # tests (never call Ollama or the network)
uv run ruff check .       # lint
```

`ingest`, `ask`, `daily`, `rate`, `watchlist` and `eval` are stubs until their milestones land.
