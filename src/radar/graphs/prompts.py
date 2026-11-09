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


# ---- Daily digest (M2) ----

SUMMARIZE = """Summarize this item for a busy ML practitioner in 2-3 sentences: what problem it \
tackles, the key idea, and the main result or claim. Use only the text below. End each sentence \
with the citation [{item_id}].

Title: {title}
Abstract: {abstract}
TLDR (Semantic Scholar): {tldr}"""


# ---- Multi-agent digest and verification (M2.5) ----

CRITIC = """You are the Critic on a daily AI research digest team. A scoring formula already \
ranked items by relevance, popularity and recency. Your job is to catch items that ranked high \
but don't deserve a slot: off-topic for this reader, or popular but weak.

Reader's interests: {topics}

Item: {item_id} ({kind}, slot: {slot})
Why it ranked: {reason}
Title: {title}
Abstract: {abstract}
TLDR (Semantic Scholar): {tldr}

Judge two things:
1. Relevance: is it about one of the reader's interests? Judge by topic, not by format: a lab \
blog post or research note on an interest counts as much as a paper. Read the interests \
broadly: faster decoding, drafting, quantization, smaller or cheaper models, serving and \
running models on modest hardware all count as efficient inference; memory, planning and tool \
use count as agents.
2. Substance (papers only): does the abstract describe a concrete method and evaluation, or is \
it vague, hype or only a position? For blog posts, judge relevance only.
Business, funding, pricing and customer stories are not research: drop them.
Popularity alone is not a reason to keep an item. Drop only when you are fairly sure.

Reply with JSON only:
{{"keep": true or false, "concerns": "one short sentence, or empty if none"}}"""

WRITER = """You are the Writer on a daily AI research digest team. Summarize this item for a \
busy ML practitioner in 2-3 sentences: the problem, the key idea, and the main result. Use only \
the text below; don't add numbers or claims it doesn't contain. End each sentence with the \
citation [{item_id}].

Title: {title}
Abstract: {abstract}
TLDR (Semantic Scholar): {tldr}
{feedback}"""

WRITER_FEEDBACK = """
Your previous draft had claims the source does not support. Rewrite without them, or state \
them only as far as the text supports:
{problems}"""

VERIFIER = """You are the Verifier. Check each numbered claim against the source passages. A \
claim is supported only if the passages state it or directly imply it. Background knowledge \
doesn't count; a claim that adds a number, a comparison or a result the passages don't give is \
unsupported.

Passages:
{passages}

Claims:
{claims}

Reply with JSON only, one entry per claim:
[{{"n": 1, "supported": true or false, "why": "short reason"}}]"""
