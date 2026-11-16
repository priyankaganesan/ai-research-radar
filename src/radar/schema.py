"""Plain records passed between sources, the store, retrieval and the graphs."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

ItemKind = Literal["paper", "post", "video", "podcast"]


@dataclass
class Item:
    """Anything the agent can read and cite: a paper, a lab post, later a video or podcast."""

    id: str  # canonical arXiv ID when there is one, otherwise "source:hash"
    source: str
    title: str
    abstract: str
    published: date
    url: str
    kind: ItemKind = "paper"
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    pdf_url: str | None = None
    tldr: str | None = None  # one-sentence summary from Semantic Scholar (M3)


@dataclass
class Passage:
    """One retrieved piece of evidence. This is what answers cite."""

    item_id: str
    title: str
    url: str
    published: str  # ISO date; a string so graph state stays JSON-friendly
    text: str
    score: float
    section: str | None = None  # M5: set when the passage comes from a full-text section


Slot = Literal["main", "early_buzz", "lab_launch"]


@dataclass
class DigestItem:
    item: Item
    rank: int
    slot: Slot
    score: float
    reason: str
    summary: str = ""


RatingValue = Literal["useful", "not_relevant", "skipped"]


@dataclass
class Rating:
    item_id: str
    value: RatingValue
    on: date
