# AI Research Radar

A personal agent that keeps one practitioner current on AI research. Each day it collects new papers and lab announcements, picks the few worth reading, summarizes them, and answers follow-up questions from a local knowledge store. This is a capstone project; the design comes from Checkpoint 2 ("Reasoning, Memory and Tools", 19 Oct 2026). Keep the code faithful to that design, and flag any change to it rather than drifting silently.

`PLAN.md` has the milestones, repo layout, data model and graph details. Work one milestone at a time, and update PLAN.md when a decision changes.

## Hard constraints

- **The agent's LLM is local.** It runs Qwen 3.5 9B (4-bit) through Ollama. Never call a cloud LLM API from agent code. Claude is only used to write the code.
- **Local inference, cloud tracing.** LangSmith (cloud) is the one allowed outbound service besides the paper sources: it receives traces, never serves the model. Tracing must be switchable off with `LANGSMITH_TRACING=false`, and the agent must work identically without it. Describe the project as "local inference, cloud tracing", not "fully local".
- **Target machine:** M3 MacBook Air, 16 GB RAM, about 29 GB of free disk, shared with another macOS account. Keep model plus context under about 11 GB. Don't add models or dependencies that assume more.
- **No Homebrew, no sudo.** Tools are installed per-user (`~/.local/bin`, `~/Applications`). Don't suggest `chown` of `/opt/homebrew` or system-wide installs.
- **Respect source terms.** IEEE Xplore: store titles and links only, never feed its content to the model. LinkedIn: not used at all (no API, scraping prohibited).

## Stack

- Python 3.12, managed with `uv`
- Agent framework: **LangGraph** for control flow, **LangChain** for components
  - Chat model via `langchain-ollama` (`ChatOllama(...).bind_tools(...)`, `base_url` from `OLLAMA_HOST`)
  - Embeddings via `OllamaEmbeddings`
  - Chat model: `qwen3.5:9b` (fallback `qwen3.5:4b`)
  - Embeddings: `qwen3-embedding:0.6b`. Queries get an instruction prefix (`Instruct: ...\nQuery: ...`); documents are embedded as is (title + abstract).
- Store: one SQLite file, using `sqlite-vec` for vectors and FTS5 for keyword search (hybrid retrieval). SQLite is the **source of truth** for all data and memory.
- Second vector backend (M1.5, for learning and comparison): **LanceDB**, holding only chunk vectors that can be rebuilt from SQLite. Both sit behind the `VectorIndex` interface in `rag/vector_index.py`; `VECTOR_BACKEND` picks one. Never store anything only in LanceDB.
- HTTP and feeds: `httpx`, `feedparser`
- CLI and config: `typer`, `pydantic-settings`
- Tests and lint: `pytest`, `ruff`
- Web UI (M7): FastAPI + Jinja2 + htmx, served by uvicorn on localhost with a single worker. CLI, MCP and web all call `services.py`; none of them touch the store or graphs directly.
- MCP: the official `mcp` Python SDK (`FastMCP`) for `radar mcp` (M1.6, required), and `langchain-mcp-adapters` so the agent can load its own tools over MCP. Tools are defined once in `tools/` and reused by both. On stdio, never print to stdout.
- Observability and evals: **LangSmith** (tracing on every graph run; datasets for retrieval and groundedness evals)
- Build the Q&A loop as a custom `StateGraph`, not the prebuilt `create_agent`: the evidence check and stopping rule must be explicit nodes and edges.

## Configuration

All settings go in `.env` (never committed). `.env.example` lists them:

```
OLLAMA_HOST=http://127.0.0.1:11434   # may be :11435 if the other account's Ollama holds the default port
CHAT_MODEL=qwen3.5:9b
EMBED_MODEL=qwen3-embedding:0.6b
DB_PATH=data/radar.db
VECTOR_BACKEND=sqlite                # sqlite | lance
LANCE_PATH=data/lance
SEMANTIC_SCHOLAR_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=ai-research-radar
```

## Architecture

### Scheduled vs. reasoned

- **Fixed, scheduled (no LLM decisions):** the daily trigger, collection from sources, deduplication against the store. It runs on the Mac via `launchd`/cron, not GitHub Actions (Actions can't reach local Ollama).
- **ReAct loop (LLM decides):** choosing the digest, answering questions, deciding when to dig deeper.

### Graphs

Two LangGraph graphs:

- **Daily graph (fixed order, no LLM routing):** `collect → learn → candidates (dedupe) → score → pick → summarize → render → recheck_watchlist`. Only `summarize` and the Profile agent inside `learn` call Qwen.
- **Q&A graph (ReAct):** `reason → tool → evidence_check → (reason again | answer | abstain)`. Conditional edges decide the route; a step counter in state enforces the stopping rule.

Export both graphs as Mermaid diagrams (`graph.get_graph().draw_mermaid()`) for the writeup.

### ReAct loop

Thought → one tool call → observation → next step.
- Weak evidence triggers an outside search; strong evidence goes straight to an answer.
- **Stopping rule:** a hard cap of a few tool calls per question (make it configurable, default 5). If the evidence still falls short, say so and log the gap to the watchlist. Never guess.
- Repeated questions on one paper trigger a full-text fetch.
- Every claim in an answer cites a paper in the store (ID, link, publication date).

### Tools (each covers a model limitation)

| Tool | Provides | Limitation |
|---|---|---|
| `search_arxiv`, `search_hf_papers` | Papers after the model's cutoff | Retrieval |
| `semantic_scholar_lookup` | Citations, references, related papers, TLDRs | Retrieval |
| `fetch_full_text` | Whole paper when the abstract lacks method detail | Retrieval |
| `store_search`, `store_write` | Real passages with sources | Grounding |
| `score_items` | Consistent ranking score | Computation |

Ranking is **always** `score_items` in code (relevance to the interest profile, HF upvotes, citations, recency). Never ask the LLM to rank a list. Rank cheaply on abstracts plus Semantic Scholar TLDRs first; write full LLM summaries only for the picked items.

### Digest

- 8 items a day. **2 slots are reserved** for:
  - Highly upvoted on Hugging Face Daily Papers but few citations so far
  - Lab launches and announcements from AI lab blogs/RSS
- Each pick is labeled with why it was included.
- Upvoted papers go on the **early-buzz watchlist**; after 2 to 4 weeks the agent re-checks citations and marks each paper established or drops it.
- User ratings (useful / not relevant / skipped) update the interest profile weights and the number of reserved slots.

### Memory

- **Short-term (one question or run):** question plus follow-ups, loop steps so far (so searches aren't repeated), retrieved passages. Stored in the graph's typed state and persisted per conversation with LangGraph's SQLite checkpointer (`thread_id` per conversation), so follow-ups resolve.
- **Long-term (our own SQLite store, not LangGraph's `BaseStore`):**
  - Knowledge: papers, summaries, chosen full texts, notes
  - History: read items, ratings, questions asked, unanswered questions
  - Interest profile: weights built from ratings and questions
  - Watchlists: topics to collect next run, early-buzz items to re-check
- Old items fade from digest ranking but stay searchable.

### Sources

arXiv API (AI categories), Semantic Scholar, Hugging Face Daily Papers, AI lab RSS feeds and pages, Hacker News.

## Conventions

- Keep it small and readable. This code is graded and explained in writing.
- Put each tool in its own module as a LangChain `@tool` with type hints and a clear docstring (that becomes the schema Qwen sees).
- Build the chat model in one factory function so tests can swap in LangChain's fake chat model (`GenericFakeChatModel`). **Tests never call Ollama, LangSmith, or the network**; use recorded fixtures and set `LANGSMITH_TRACING=false` in tests and CI.
- Every loop step is traced in LangSmith; also keep a short local log of each step so traces exist when tracing is off.
- Evals live in `evals/`: a LangSmith dataset of about 20 questions with known answer papers, scored on top-5 retrieval recall and on whether answers cite real store papers. Use it to compare models.
- Never commit `.env`, `data/`, `walkthroughs/`, or model files.
- After each milestone, write a plain-language walkthrough to `walkthroughs/mN-explained.md` covering what each new file does, the concepts behind it, and how to run and check it. The owner is using this project to learn.

## Commands

```
uv sync                 # install deps
uv run pytest           # tests
uv run ruff check .     # lint
uv run radar daily      # run the daily pipeline
uv run radar ask "..."  # ask a question
uv run radar eval       # run LangSmith evals
```
