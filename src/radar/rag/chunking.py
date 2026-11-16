"""Turning items into searchable chunks.

M1 rule: one chunk per item (title + abstract, about 250 tokens). Abstracts are short and
self-contained, so splitting them would cut ideas in half. Full texts (M5) are split by section,
then into ~500-token windows that overlap by ~50 tokens (`section_chunks`).
"""

import re
from dataclasses import dataclass

from radar.schema import Item


@dataclass
class Chunk:
    item_id: str
    kind: str
    text: str
    token_count: int
    section: str | None = None


def approx_tokens(text: str) -> int:
    """Rough token count (about 1.3 tokens per English word). Good enough for sizing chunks."""
    return round(len(text.split()) * 1.3)


def abstract_chunk(item: Item) -> Chunk:
    text = f"{item.title}\n\n{item.abstract}".strip()
    return Chunk(item_id=item.id, kind="abstract", text=text, token_count=approx_tokens(text))


# ---- Full text (M5): split by section, then into overlapping windows -------------------------

SECTION_CHUNK_TOKENS = 500
SECTION_OVERLAP_TOKENS = 50
MIN_CHUNK_TOKENS = 40  # a shorter tail is merged into the chunk before it
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[$])|\n\n+")


def split_sentences(text: str) -> list[str]:
    """Split at sentence ends and paragraph breaks. Rough (it splits after "et al." before a
    capital), but chunks only need *sensible* cut points, not perfect sentences."""
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _pieces(sentences: list[str], size: int) -> list[str]:
    """Sentences, with any single one longer than a chunk (a table or formula dump) cut by
    words."""
    out = []
    step = max(1, int(size / 1.3))  # words per chunk
    for sent in sentences:
        if approx_tokens(sent) <= size:
            out.append(sent)
        else:
            words = sent.split()
            out += [" ".join(words[i : i + step]) for i in range(0, len(words), step)]
    return out


def _windows(sentences: list[str], size: int, overlap: int) -> list[list[str]]:
    """Pack sentences into windows of about `size` tokens. Each window starts with the last
    ~`overlap` tokens of the one before, so an idea cut at a boundary appears whole in one."""
    out: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    fresh = 0  # sentences in `cur` that aren't overlap carried from the previous window
    for piece in _pieces(sentences, size):
        pt = approx_tokens(piece)
        if fresh and cur_tokens + pt > size:
            out.append(cur)
            tail: list[str] = []  # carry the last sentences over, up to `overlap` tokens
            for s in reversed(cur):
                if sum(map(approx_tokens, tail)) + approx_tokens(s) > overlap:
                    break
                tail.insert(0, s)
            cur, cur_tokens, fresh = tail, sum(map(approx_tokens, tail)), 0
        cur.append(piece)
        cur_tokens += pt
        fresh += 1
    if fresh:
        new = cur[len(cur) - fresh :]
        if out and sum(map(approx_tokens, new)) < MIN_CHUNK_TOKENS:
            out[-1].extend(new)  # a tiny tail isn't worth its own chunk
        else:
            out.append(cur)
    return out


def section_chunks(
    item_id: str,
    title: str,
    sections: list,  # list[radar.sources.full_text.Section]
    size: int = SECTION_CHUNK_TOKENS,
    overlap: int = SECTION_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Full text -> chunks: never across a section boundary, ~`size` tokens each, overlapping.

    Each chunk's text starts with the paper title and section name ("contextual header"), so a
    chunk like "We set k=4 ..." still says which paper and which part it came from, both to the
    embedding model and to Qwen reading the passage.
    """
    chunks = []
    for sec in sections:
        for window in _windows(split_sentences(sec.text), size, overlap):
            body = " ".join(window)
            text = f"{title}\nSection: {sec.heading}\n\n{body}"
            chunks.append(Chunk(item_id, "section", text, approx_tokens(text), section=sec.heading))
    return chunks
