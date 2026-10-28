"""Turning items into searchable chunks.

M1 rule: one chunk per item (title + abstract, about 250 tokens). Abstracts are short and
self-contained, so splitting them would cut ideas in half. Full texts (M5) split by section.
"""

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
