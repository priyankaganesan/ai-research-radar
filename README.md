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
uv run radar ingest --days 3          # fetch recent arXiv papers and embed them
uv run radar ingest hf                # Hugging Face Daily Papers + upvotes
uv run radar ingest labs              # AI lab blog posts
uv run radar ingest hn                # Hacker News AI stories + points
uv run radar ingest all               # every source, then Semantic Scholar enrichment
uv run radar enrich                   # citation counts + TLDRs for recent papers
uv run radar show 2401.10774          # one item with its TLDR and signals
uv run radar search "speculative decoding"   # retrieval only, no LLM
uv run radar ask "How does Medusa speed up decoding?" --steps
uv run radar ask "..." --no-retrieval  # same model, no tools: the comparison baseline
uv run radar eval --file evals/questions.jsonl   # retrieval recall@5
uv run radar reindex --backend lance [--ann]  # copy vectors into LanceDB (optional IVF-PQ index)
uv run radar compare-backends                  # sqlite-vec vs. LanceDB: recall, latency, disk
uv run pytest             # tests (never call Ollama or the network)
uv run ruff check .       # lint
```

`daily`, `rate` and `watchlist` are stubs until their milestones land.
