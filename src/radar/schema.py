"""Plain records passed between sources, the store, retrieval and the graphs."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal


@dataclass
class Paper:
    id: str  # canonical arXiv ID when there is one, otherwise "source:hash"
    source: str
    title: str
    abstract: str
    published: date
    url: str
    authors: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    pdf_url: str | None = None


@dataclass
class Passage:
    paper_id: str
    title: str
    url: str
    published: date
    text: str
    score: float


Slot = Literal["main", "early_buzz", "lab_launch"]


@dataclass
class DigestItem:
    paper: Paper
    rank: int
    slot: Slot
    score: float
    reason: str
    summary: str = ""


RatingValue = Literal["useful", "not_relevant", "skipped"]


@dataclass
class Rating:
    paper_id: str
    value: RatingValue
    on: date
