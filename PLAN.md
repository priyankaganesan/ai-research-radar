# AI Research Radar: build plan

The plan for the whole codebase. `CLAUDE.md` holds the rules and design; this file holds the order of work and the shape of each module. Build one milestone at a time, and keep every milestone runnable end to end.

## Milestones

| # | Milestone | Outcome you can demo | Depends on |
|---|---|---|---|
| M0 | Skeleton | `uv run pytest` and `radar --help` work; CI green | none |
| M1 | RAG Q&A (this week) | `radar ingest` + `radar ask`: answers from the store with citations, falls back to live arXiv, abstains when unsupported. Also lands: `items` table with `kind`, WAL mode, `VectorIndex` interface, `services.py`, first eval set, `--no-retrieval` demo | M0 |
| M1.5 | Second vector backend | `VECTOR_BACKEND=lance` runs the same Q&A on LanceDB; eval compares exact (sqlite-vec) vs. approximate (LanceDB) search on recall@5, latency and disk | M1 |
| M1.6 | MCP server (required) | `radar mcp` exposes the store and search tools over MCP; Claude Desktop / Claude Code can search the research store; the Q&A graph can load its tools through MCP | M1 |
| M2 | Daily pipeline | `radar daily` writes a ranked 8-item digest from arXiv | M1 |
| M2.5 | Multi-agent digest and verification | The digest's `summarize` step becomes Critic → Writer → Verifier agents under a supervisor; Q&A gets the same Verifier for claim-level checks; eval compares single- vs. multi-agent on quality and time | M2 |
| M3 | More sources | Hugging Face upvotes, Semantic Scholar citations and TLDRs, lab RSS, Hacker News feed into scoring. Collection built first (before M2) so the data is queryable; scoring uses it in M2 | M2 |
| M4 | Long-term memory and feedback | Ratings update the profile, reserved slots, early-buzz watchlist re-checks, gap logging; a Profile agent turns ratings and gaps into interest updates (uses M2.5's agent pattern) | M2, M3, M2.5 |
| M5 | Going deeper | Full-text fetch and section chunking when a paper gets repeated questions | M1 |
| M7 | FastAPI web UI | `radar serve` opens a local web app: ask with streamed answers and citations, browse and rate the digest, search the store, watchlist | M2, M4 (build before recording the presentation) |
| M6 | Evals and polish | LangSmith eval suite, model comparisons, scheduled runs, graph diagrams, demo script. Stretch 1: a pgvector backend. Stretch 2: full cloud pipeline (see Stretch goals) | all |

M3 and M5 are independent of each other; do whichever the next assignment needs first. M1.5 is for learning: it exists to compare two vector stores, not because the agent needs two. M7 comes after M4 because the UI is most useful once there's a digest to rate, and before M6 so the presentation can demo it.

## Repository layout

```
ai-research-radar/
├── CLAUDE.md
├── PLAN.md
├── README.md
├── pyproject.toml            # uv project; console script `radar = radar.cli:app`
├── .env.example
├── .gitignore                # .env, data/, .venv/
├── .github/workflows/ci.yml  # ruff + pytest, LANGSMITH_TRACING=false
├── langgraph.json            # for `langgraph dev` (Studio)
├── src/radar/
│   ├── config.py             # Settings loaded from .env (pydantic-settings)
│   ├── models.py             # get_chat_model(), get_embeddings(); the only place models are built
│   ├── schema.py             # Paper, Passage, DigestItem, Rating dataclasses
│   ├── store/
│   │   ├── db.py             # connect(), migrations, sqlite-vec loading
│   │   ├── items.py          # upsert_item, get_item, exists
│   │   ├── memory.py         # ratings, questions, gaps, profile, watchlists
│   │   └── migrations/       # 001_init.sql, 002_memory.sql, ...
│   ├── sources/
│   │   ├── arxiv.py          # fetch_recent(categories, since), search(query)
│   │   ├── http.py           # shared client + retry/backoff, SourceUnavailable (M3)
│   │   ├── hf_papers.py      # daily papers + upvotes
│   │   ├── semantic_scholar.py  # batch lookup (citations, TLDR) + search
│   │   ├── lab_blogs.py      # lab blog RSS/Atom feeds -> kind "post"
│   │   └── hackernews.py     # AI stories via Algolia HN search; arXiv links -> hn_points
│   ├── rag/
│   │   ├── chunking.py       # abstract = 1 chunk; full text = sections, ~500 tokens, overlap
│   │   ├── index.py          # embed + write to the vector index and FTS table
│   │   ├── vector_index.py   # VectorIndex protocol + SqliteVecIndex (M1)
│   │   ├── lance_index.py    # LanceIndex: flat or IVF-PQ search (M1.5)
│   │   ├── compare.py        # exact vs. approximate: recall, latency, disk (M1.5)
│   │   └── retrieve.py       # hybrid search + RRF -> list[Passage]
│   ├── tools/                # LangChain @tool wrappers, one per file
│   │   ├── store_search.py
│   │   ├── search_arxiv.py
│   │   ├── semantic_scholar_lookup.py
│   │   ├── fetch_full_text.py
│   │   └── score_items.py
│   ├── scoring.py            # pure functions; no LLM
│   ├── graphs/
│   │   ├── qa.py             # Q&A ReAct graph
│   │   ├── daily.py          # fixed-order daily graph
│   │   └── prompts.py        # all prompt templates in one place
│   ├── context.py            # AppContext: settings, DB connection, embeddings, vector index
│   ├── services.py           # ask(), search(), get_item(), digest(), rate(): one layer used by CLI, MCP and web
│   ├── mcp_server.py         # MCP server (M1.6): wraps services
│   ├── web/                  # FastAPI app (M7)
│   │   ├── app.py            # create_app(): routers, lifespan (open DB, build graph once)
│   │   ├── api.py            # JSON endpoints under /api
│   │   ├── pages.py          # HTML pages (Jinja2 templates)
│   │   ├── templates/        # base, ask, digest, search, item, watchlist
│   │   └── static/           # small CSS; htmx loaded from a pinned CDN version
│   ├── digest.py             # render digest to Markdown
│   └── cli.py                # typer app: ingest, ask, daily, rate, watchlist, eval
├── tests/
│   ├── conftest.py           # temp DB, fake chat model, fake embeddings, recorded HTTP fixtures
│   ├── fixtures/             # saved arXiv/HF/S2 responses
│   └── test_*.py
├── evals/
│   ├── questions.jsonl       # ~20 questions with expected paper IDs
│   └── run.py                # LangSmith evaluate(): recall@5, citation validity
├── walkthroughs/             # gitignored: plain-language notes per milestone
└── data/                     # gitignored: radar.db, lance/, digests/YYYY-MM-DD.md
```

## Data model (SQLite)

M1:
- `items(id TEXT PK, kind TEXT 'paper'|'post'|'video'|'podcast', source, title, abstract, authors JSON, published DATE, url, pdf_url, categories JSON, added_at)`
  - `id` is the canonical arXiv ID when there is one, otherwise `source:hash`.
  - `kind` keeps the store open to lab posts (M3) and future video/podcast transcripts. Later tables still say `paper_id`; it refers to `items.id`.
- `chunks(id INTEGER PK, item_id FK, kind TEXT 'abstract'|'section'|'transcript', section TEXT, text, token_count, start_s REAL NULL, end_s REAL NULL)`
  - `start_s`/`end_s` are timestamps for transcript chunks, so answers can cite "@ 14:05".
- `chunks_vec` virtual table (`vec0`, 1024-dim float) keyed by `chunks.id`
- `chunks_fts` virtual table (FTS5 over `text`, plus the paper title)

M3 (built, `003_sources.sql`):
- `items.tldr`: Semantic Scholar's one-sentence summary, when it has one
- `signals(item_id, date, hf_upvotes, citations, influential_citations, hn_points, hn_comments)`: one row per item per day, so attention over time is visible; each source fills only its own columns (upsert with COALESCE). Citation velocity is computed in M2 from the change between rows.
- Item IDs: arXiv ID for papers (HF, S2 and HN-linked papers merge with arXiv ones), `post:<sha1 of link>` for lab posts, `hn:<story id>` for other HN stories, `s2:<paperId>` for S2 papers with no arXiv ID.

M2 (built, `004_digest.sql`):
- `digests(date, item_id, rank, slot TEXT 'main'|'early_buzz'|'lab_launch', score, reason)`
- `summaries(item_id, mode 'single'|'multi', text, model, created_at)`: one per writer, so M2 and M2.5 summaries can be compared
- `profile(key, value JSON)`: created now (empty = defaults: `INTEREST_TOPICS`, default weights, 2 reserved slots); M4 ratings write to it

M4:
- `ratings(paper_id, date, value TEXT 'useful'|'not_relevant'|'skipped')`
- `questions(id, asked_at, text, answered BOOL, cited_ids JSON)`
- `gaps(id, question_id, topic, created_at, resolved_at)`: unanswered questions become collection topics
- `profile(key, value)`: scoring weights, number of reserved slots, topic terms
- `watchlist(paper_id, added_at, recheck_after, status 'watching'|'established'|'dropped')`

`connect()` turns on WAL mode (`PRAGMA journal_mode=WAL`) so `radar daily` can write while `radar ask` reads.

The LangGraph checkpointer gets its own file (`data/checkpoints.db`) so short-term memory stays separate from long-term memory.

## Retrieval (M1)

- **Chunking:** each paper becomes one chunk (title + abstract, about 250 tokens). Full text (M5) splits by section heading, then into ~500-token chunks with ~50 tokens of overlap.
- **Embeddings:** `qwen3-embedding:0.6b`. The query gets an instruction prefix; documents don't.
- **Vector index interface:** `VectorIndex` has `add(chunk_ids, vectors)`, `search(vector, k, filters) -> [(chunk_id, distance)]`, `delete(chunk_ids)`, `count()`. `SqliteVecIndex` (M1) and `LanceIndex` (M1.5) implement it; `VECTOR_BACKEND` picks one. Keyword search stays in FTS5 for both.
- **Hybrid search:** take the top 20 by vector distance and the top 20 by FTS5 BM25, merge them with reciprocal rank fusion (`1 / (60 + rank)`), and collapse chunks to one per paper.
- **Output:** the top 5 `Passage(paper_id, title, url, published, text, score)`. `k` is configurable.
- **Filters:** optional date range and category, so "this month's papers" works.

## Second vector backend (M1.5)

- **SQLite stays the source of truth.** LanceDB stores only `(chunk_id, vector)`. Filters (`kind`, `since`) are applied in SQLite after the vector search, as with sqlite-vec, so LanceDB needs no filter columns. Anything in LanceDB can be rebuilt from SQLite.
- Writes go through `index.py`, which always writes vectors to SQLite's `chunks_vec` inside the batch transaction, then copies them to LanceDB after the commit when `VECTOR_BACKEND=lance`. If that copy fails, `reindex` repairs it.
- `radar reindex --backend lance [--ann]` copies vectors from SQLite into `data/lance/` (no re-embedding). It builds an IVF-PQ index with `--ann`, or automatically at 5,000+ vectors; below that LanceDB does a flat (exact) scan. `radar reindex --backend sqlite` re-embeds every chunk with the current `EMBED_MODEL`.
- IVF-PQ settings: partitions = min(sqrt(n), n / 256), because LanceDB trains each partition on at least 256 vectors (a few hundred papers get one partition); 64 sub-vectors (16 dims each), so a 4 KB vector compresses to 64 bytes. Search-time knobs: `LANCE_NPROBES`, `LANCE_REFINE_FACTOR`.
- **Comparison:** `radar compare-backends [--file evals/questions.jsonl]` runs the same queries on both: recall@k of LanceDB against exact search (no labels needed; default queries are 50 item titles), p50/p95 vector-search latency, disk size, a sweep over `nprobes` x `refine_factor` once an ANN index exists, and recall against expected papers when the file has `expected_ids`. Results are saved to `evals/results/backends-YYYYMMDD-HHMM.md`.
- LanceDB, like embedded Chroma, assumes one writing process. Run `daily` and `reindex` one at a time.

## MCP server (M1.6, required)

MCP (Model Context Protocol) is the standard way for AI apps to discover and call tools. Radar's tools are defined once in `tools/` and exposed two ways: as LangChain tools inside the agent, and as an MCP server any MCP client can use.

- Built with the official `mcp` Python SDK (`FastMCP`), **stdio** transport by default (the client launches `radar mcp` as a subprocess); optional `--http` for streamable HTTP on localhost.
- **Tools** (built 2026-11-07):
  - `store_search`, `search_arxiv`, `semantic_scholar_lookup`: the agent's own LangChain tools from `tools/`, converted with `langchain_mcp_adapters.tools.to_fastmcp`, so they're defined once. (Names follow the agent tools rather than the earlier `search_store` draft.)
  - `get_item(item_id)`: full record plus TLDR and latest signals
  - `ask(question, thread_id=None)`: runs the full Q&A graph on local Qwen and returns the cited answer
- **Resources:** `radar://digest/latest` (newest file in `data/digests/`), `radar://stats`. `radar://watchlist` comes with M4.
- **Prompts:** `explain_paper(item_id)`: a ready-made prompt template
- **Writes are labelled.** The live-search tools save what they find to the store (the Q&A loop relies on that), so they carry the MCP annotation `readOnlyHint=false` and say so in their title; clients can ask the user first. The other tools are marked read-only.
- **Dogfooding:** a setting `TOOLS_VIA_MCP=true` makes the Q&A graph load its tools from the MCP server with `langchain-mcp-adapters`, instead of importing them directly. Same behavior, but it shows the protocol end to end. `mcp_client.py` keeps the async MCP session on a background event loop and gives the sync graph thin sync wrappers.
- **Client setup** goes in the README: `claude mcp add radar -- uv run --directory /Users/Sindhu/cmu/ai_research_radar radar mcp`, plus the matching `claude_desktop_config.json` entry.
- **Testing:** an in-memory MCP client session lists the tools and calls `search_store` against a temp store with fake embeddings. Inspect manually with the MCP Inspector (`npx @modelcontextprotocol/inspector`).
- Logs go to stderr only. On stdio, stdout is the protocol channel, so a stray `print` breaks it.

## Q&A graph (M1)

State:
```python
class QAState(TypedDict):
    messages: Annotated[list, add_messages]  # question, follow-ups, tool calls
    passages: list[Passage]  # evidence gathered so far
    searches_done: list[str]  # avoid repeating a query
    steps: int  # stopping rule
    verdict: Literal["", "sufficient", "insufficient"]
```

Nodes and edges:
```
START → rewrite_query → reason ──tool call──▶ tools ──▶ evidence_check
                          ▲                               │
                          └──── insufficient, steps<MAX ──┤
                                                          ├── sufficient ──▶ answer ──▶ END
                                                          └── insufficient, steps≥MAX ──▶ abstain ──▶ END
```
- `rewrite_query`: Qwen turns a follow-up into a self-contained search query, using message history.
- `reason`: Qwen with tools bound (`store_search`, `search_arxiv`, `semantic_scholar_lookup` (M3); later `fetch_full_text`).
- `evidence_check`: Qwen with structured output `{sufficient: bool, missing: str}`, given only the question and the passages. It runs with thinking mode on.
- `answer`: it answers only from passages, and every claim has a `[paper_id]` citation. A post-check drops any citation not in `passages`.
- `abstain`: it says what is missing, writes a row to `gaps` (M4), and lists the closest papers it found.
- `--no-retrieval` mode: Qwen answers the same question with no tools, for demos and evals.

## Daily graph (M2)

Built 2026-11-07 as:
```
START → collect → candidates → score → pick → summarize → render → END      (recheck_watchlist: M4)
```
- `collect` runs every source and then Semantic Scholar enrich (`services.collect_all`), so `enrich` is folded in. `radar daily --no-collect` skips it.
- `candidates` is the dedupe step: the store already merges duplicate IDs across sources, so this drops items shown in an earlier digest and keeps items published in the last `DIGEST_WINDOW_DAYS` (7). HN stories without a paper are candidates too.
- Each node records its wall-clock time in `timings`, for the single vs. multi-agent comparison.

Original sketch:
```
START → collect → dedupe → enrich → score → pick → summarize → render → recheck_watchlist → END
```
- `collect`: arXiv recent (M2), then HF, RSS and HN (M3), plus topics from `gaps` and the watchlist.
- `enrich`: Semantic Scholar citations and TLDRs, and HF upvotes (M3).
- `score`: `scoring.score(paper, signals, profile)`. It's a pure function; see below.
- `pick`: 6 main slots, plus 2 reserved slots (early buzz, lab launch). The reserved count comes from `profile`.
- `summarize`: the only LLM step. Qwen summarizes the 8 picks, with thinking off.
- `render`: writes `data/digests/YYYY-MM-DD.md`, with each item's label and one-line reason.
- `recheck_watchlist` (M4): papers past `recheck_after` are re-scored on citations and marked established or dropped.

## Multi-agent digest and verification (M2.5)

Rule: an agent is a step that needs its own role, prompt and tools. Anything deterministic (collect, dedupe, score, pick, render) stays a plain code node. Every agent is the same local Qwen with a different prompt; Ollama runs one call at a time, so each agent adds 10-30 s and must earn its place.

```
daily graph:  ... → score → pick → supervisor ──▶ critic ──▶ writer ──▶ verifier ──▶ render
                                       ▲                                   │
                                       └──── unsupported claims, retries<2 ┘
```
- `critic` (agent): reads each candidate's abstract and TLDR and judges methods and relevance to the profile: `{keep: bool, concerns: str}`. A dropped pick is replaced by the next by score. Answers Checkpoint 2 feedback "popularity is not quality".
- `writer` (agent): the old `summarize` step; 2-3 sentence summary per pick with `[paper_id]` citations.
- `verifier` (agent, shared with Q&A): splits each summary into claims and checks each against the cited passage: `{claim, supported: bool, why}`. Unsupported claims go back to the writer with the reason; after 2 retries the claim is cut. Answers feedback "check claims are supported by their source".
- `supervisor`: a LangGraph router (code, not an LLM) that sends work to the next agent and enforces the retry limit. Each agent is a subgraph in `graphs/agents/`, so it can be tested alone with a fake LLM.
- Q&A: after `answer`, the same `verifier` runs; unsupported sentences are removed or marked, and the trace shows why.
- Eval: run the same questions and a week of digests single-agent (M2) and multi-agent; compare claim support rate, critic agreement with your ratings, and wall-clock time. Report the trade-off either way.
- Exploration hook: LangGraph's prebuilt supervisor / handoff tools vs. the hand-written router.

Built 2026-11-07:
- `graphs/agents/critic.py`, `writer.py`, `verifier.py` (one subgraph each) and `team.py` (the supervisor graph). `DIGEST_MODE=multi` or `radar daily --mode multi` swaps the team in for `summarize`.
- Limits: 2 rewrites per item (`MAX_RETRIES`), at most 3 critic drops per digest (`MAX_DROPS`, so the critic can't empty it); a dropped pick is replaced by the next item by score, in the same position, and also meets the critic. An unreadable critic verdict keeps the item.
- The verifier splits claims in code (one sentence = one claim) and judges all claims of a text in one call; a claim with no verdict counts as unverified, not unsupported.
- Q&A: `VERIFY_ANSWERS=true` or `radar ask --verify` adds a `verify` node after `answer`; unsupported sentences are removed (with a note), and if none survive the graph abstains.
- `radar compare-agents [--date D] [--file questions.jsonl]` builds the same day's digest in both modes and judges every final claim with a fresh verifier pass; writes `evals/results/agents-*.md`. Critic agreement with ratings waits for M4.

## Scoring (M2, tuned in M4)

```
score = w_rel * relevance        # cosine(paper embedding, profile vector)
      + w_att * attention        # normalized HF upvotes (+ HN points), relative to age
      + w_cit * log1p(citation_velocity)
      + w_rec * exp(-age_days / 7)
```
- The profile vector is the mean embedding of papers rated useful plus the seed topic terms. Papers rated not relevant push it away.
- **Early-buzz slot:** high `attention`, low citations.
- **Lab-launch slot:** the best-scoring RSS item from a lab feed.
- Ratings adjust the weights in small steps, within bounds. Usefulness rates on the reserved slots move the reserved count between 1 and 3.
- Keep every weight in `profile`, so the digest's "why" line can show the components.

## CLI

```
radar ingest [arxiv|hf|labs|hn|all] [--days N]            # M1 (arxiv), M3 (others; all = every source + enrich)
radar enrich [--days 30] [--limit 500]                     # M3: S2 citations + TLDRs
radar show ITEM_ID                                         # M3: one item, its TLDR and signals
radar search "query" [--k 5] [--since YYYY-MM-DD]          # M1 (retrieval only, no LLM)
radar ask "question" [--no-retrieval] [--thread ID] [--steps]  # M1
radar rate PAPER_ID useful|not_relevant|skipped            # M4
radar watchlist                                            # M4
radar reindex --backend sqlite|lance [--ann]               # M1.5
radar compare-backends [--file F] [--k 5] [--sample 50]    # M1.5
radar mcp [--http]                                         # M1.6
radar daily [--mode single|multi] [--no-collect] [--date D]  # M2, M2.5
radar compare-agents [--date D] [--file F]                 # M2.5
radar serve [--port 8000]                                  # M7
radar eval [--model ...] [--embed-model ...] [--backend ...]  # M1 basic, M6 full
```

## Testing

- Use a fake chat model (`GenericFakeChatModel`) with scripted responses to test graph routing: sufficient goes to answer, insufficient goes to search again, and hitting the cap goes to abstain.
- Use fake embeddings (deterministic hash vectors) to test indexing and RRF ordering.
- Use recorded HTTP fixtures for every source; tests never hit the network.
- `scoring.py` gets plain unit tests: monotonic in each signal, recency decay, slot selection.
- CI runs `ruff` and `pytest` only.

## Evals (start in M1, grow in M6)

- `evals/questions.jsonl`: `{question, expected_paper_ids, should_abstain}`
- Metrics: recall@5 for retrieval, the share of citations that point to real store papers, and whether the agent abstains correctly on questions the store can't answer.
- Run through LangSmith `evaluate()` to compare embedding models, 9B vs. 4B, and retrieval on vs. off.

## Scheduling (M6)

A `launchd` plist in `~/Library/LaunchAgents/` runs `radar daily` each morning while the Mac is awake. It's per-user and needs no admin rights.

## Service layer (from M1)

`services.py` holds the operations the agent offers: `ask`, `search`, `get_item`, `digest`, `rate`, `watchlist`. The CLI (M1), the MCP server (M1.6) and the web app (M7) are thin wrappers that call these functions. They never touch the store or the graphs directly. That keeps the three interfaces consistent, so each new interface is only a little code.

## FastAPI web UI (M7)

FastAPI serves both a JSON API and simple server-rendered pages. The pages use Jinja2 templates plus **htmx** for interactivity (rating buttons, streaming answers) without a JavaScript framework.

- **API (`/api`)**, with Pydantic request and response models (and auto-generated docs at `/docs`):
  - `POST /api/ask` with `{question, thread_id?, no_retrieval?}`: streams the graph's steps and final answer as Server-Sent Events, using LangGraph's `astream`
  - `GET /api/search?q=&k=&kind=&since=`
  - `GET /api/items/{id}`
  - `GET /api/digest/{date}` (default today), `POST /api/ratings` with `{item_id, value}`
  - `GET /api/watchlist`
  - `GET /api/health`: checks that the database, the Ollama connection and the models are reachable
- **Pages:**
  - **Ask:** a chat view with clickable citations and an expandable "how I got this" panel showing each thought, tool call and observation. This is the ReAct trace, made visible for the demo.
  - **Digest:** today's 8 items with slot labels, reasons and useful / not relevant / skip buttons.
  - **Search** and **Item** pages, plus **Watchlist**.
- **Runs locally:** `radar serve` starts uvicorn on `127.0.0.1:8000`, with **one worker** (the Qwen graph and the embedded stores assume one process). No login, because it's bound to localhost only.
- **Lifespan:** open the database and build the graphs once at startup; inject them into routes with `Depends`.
- **Async:** route handlers are `async`. Blocking work (SQLite, Ollama calls through sync clients) runs in a threadpool.
- **Testing:** `TestClient` with the fake chat model and a temp store. Covers each endpoint, the SSE stream shape and the rating flow.

## Explorations (optional, one at a time)

Each one plugs into an existing seam and has something measurable to compare. Qwen stays the LLM throughout.

| Exploration | Plugs into | Compare / measure | When |
|---|---|---|---|
| Reranker (`Qwen3-Reranker-0.6B` or `bge-reranker`) | `rag/retrieve.py`, after hybrid search: re-score top 20, keep 5 | recall@5 and answer quality with vs. without | after M1 |
| Docling or Marker for PDFs | `rag/chunking.py` full-text path | section quality vs. plain PyMuPDF | M5 |
| RAGAS or DeepEval | `evals/` | faithfulness and context precision vs. our recall@5 | M6 |
| Serving runtimes: MLX-LM, llama.cpp | `models.py` `base_url` | tokens/sec, memory, tool-call reliability vs. Ollama | any time |
| Constrained decoding (Ollama JSON schema, Outlines) | `evidence_check` structured output | parse-failure rate | after M1 |
| DSPy prompt optimization | `graphs/prompts.py` | eval score before/after | M6 |
| Arize Phoenix (local OpenTelemetry tracing) | alongside LangSmith | local vs. cloud tracing | after M1 |
| Prefect or Dagster | daily graph scheduling | retries, logs, UI vs. launchd | M6 |
| Streamlit or Gradio | `services.py` peer | quick-prototype UI vs. the FastAPI app (M7) | after M7 |
| pgvector | `VectorIndex` third backend | vs. sqlite-vec and LanceDB | M6 stretch 1 |

## Stretch goals (M6)

1. **pgvector backend.** A third `VectorIndex` backend on Postgres + pgvector, compared with sqlite-vec and LanceDB through `radar compare-backends`.
2. **Full cloud pipeline** (builds on 1). The daily ingest runs while the laptop is off:
   - A GitHub Actions workflow with a daily cron runs `radar ingest all` on GitHub's machines.
   - The same job installs Ollama and runs `qwen3-embedding:0.6b` on the runner's CPU, so vectors match the ones made on the Mac. It's an open model running inside the job, not a cloud LLM API.
   - Items, signals and vectors go to hosted Postgres + pgvector (for example the Neon or Supabase free tier). Keyword search moves from SQLite FTS5 to Postgres full-text search (`tsvector`), so hybrid RRF retrieval keeps working.
   - The Mac's agent queries the hosted store over the network; Qwen still answers locally.
   - Needs one GitHub secret (the database URL). The schedule only reads public sources.
   - Considered and set aside for now (2026-10-28): cloud collects into Postgres while the Mac embeds via `radar sync`, and a local launchd schedule.

## Open decisions

- **How the digest is delivered.** It's a Markdown file for now. Email or a notification could come later.
- **Interface beyond the CLI.** A small web UI (Streamlit) for the presentation is optional and could come in M6.
