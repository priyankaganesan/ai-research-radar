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
uv run radar daily                    # collect, rank, summarize: data/digests/YYYY-MM-DD.md
uv run radar daily --no-collect       # build the digest from the store as it is (fast)
uv run radar daily --mode multi       # Critic, Writer and Verifier agents write the digest
uv run radar ask "..." --verify       # the Verifier agent checks each claim of the answer
uv run radar compare-agents           # single vs. multi-agent: claim support and time
uv run radar mcp                      # MCP server over stdio (see below)
uv run pytest             # tests (never call Ollama or the network)
uv run ruff check .       # lint
```

`rate` and `watchlist` are stubs until their milestones land.

## Use Radar from Claude Code or Claude Desktop (MCP)

`radar mcp` is an MCP server. It offers `store_search`, `search_arxiv`,
`semantic_scholar_lookup`, `get_item` and `ask`, the resources `radar://digest/latest` and
`radar://stats`, and the prompt `explain_paper`.

Claude Code (VS Code extension or CLI) picks up the project's `.mcp.json` automatically: reload
the window, approve the `radar` server, and check it with `/mcp`. If `uv` isn't at
`~/.local/bin/uv` on your machine, put the output of `which uv` in its `command`.

```bash
# Claude Code CLI, as an alternative to .mcp.json
claude mcp add radar -- uv run --directory /Users/Sindhu/cmu/ai_research_radar radar mcp

# Inspect it by hand in a browser
npx @modelcontextprotocol/inspector uv run --directory /Users/Sindhu/cmu/ai_research_radar radar mcp
```

Claude Desktop: add this to `~/Library/Application Support/Claude/claude_desktop_config.json`
(use the full path to `uv`, e.g. from `which uv`):

```json
{
  "mcpServers": {
    "radar": {
      "command": "/Users/Sindhu/.local/bin/uv",
      "args": ["run", "--directory", "/Users/Sindhu/cmu/ai_research_radar", "radar", "mcp"]
    }
  }
}
```

Set `TOOLS_VIA_MCP=true` in `.env` and `radar ask` loads its own tools through this server
instead of importing them. Answers are the same; it shows the protocol end to end.
