"""All prompt templates in one place, so they're easy to read, compare and tune."""

REWRITE = """Rewrite the user's latest message as one standalone search question, using the \
conversation for context (resolve words like "it", "that paper", "the second approach").
Reply with the question only.

Conversation:
{history}

Latest message: {question}"""

REASON = """You are AI Research Radar, a research assistant that answers questions about AI \
papers using tools. You must gather evidence with a tool before answering.

Tools:
- store_search: search the local store of papers. Try this first.
- search_arxiv: search arXiv live. Use it when the store lacks what the question needs.
- semantic_scholar_lookup: search Semantic Scholar live; results include citation counts.
  Use it when the store lacks the answer, when citations matter, or when search_arxiv fails.

Question: {question}

Evidence gathered so far:
{evidence}

Searches already done (don't repeat them): {searches}
{missing}
Call exactly one tool with a focused query."""

EVIDENCE_CHECK = """Decide whether the passages below contain enough evidence to answer the \
question with citations. Judge only by the passages, not by what you know.

Question: {question}

Passages:
{passages}

Reply with JSON only, in this exact form:
{{"sufficient": true or false, "missing": "what is still missing, or empty if sufficient"}}"""

ANSWER = """Answer the question using ONLY the passages below.
- After every claim, cite the passage it comes from by its ID in square brackets, like [2410.01234].
- Mention publication dates when recency matters.
- If the passages only partly answer the question, say which part is not covered.
- Be concise: a short paragraph or a few bullet points.

Question: {question}

Passages:
{passages}"""

NO_RETRIEVAL = """Answer this question about AI research from your own knowledge. \
Name specific papers if you can.

Question: {question}"""


def format_passages(passages: list[dict], max_chars: int = 1200) -> str:
    if not passages:
        return "(none)"
    return "\n\n".join(
        f"[{p['item_id']}] {p['title']} ({p['published']})\n{p['text'][:max_chars]}"
        for p in passages
    )
