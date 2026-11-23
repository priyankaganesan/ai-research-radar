"""Turning data into safe HTML: Markdown, citation links, times and durations.

Everything the agent shows comes from outside (paper titles, abstracts, lab posts) or from the
model, and the safety guardrails already assume that text can be hostile. So:
- Templates autoescape: `{{ title }}` can never inject HTML.
- Markdown (answers, summaries) is rendered with raw HTML switched off, and links are allowed
  only to http(s) addresses or to pages of this app, opening with rel="noopener noreferrer".
- Nothing from the store or the model is ever marked `|safe` except the output of `md()` below.
"""

import re
from datetime import UTC, datetime
from urllib.parse import quote

from markdown_it import MarkdownIt
from markupsafe import Markup, escape

_md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})


def _allowed_link(url: str) -> bool:
    u = url.strip().lower()
    return u.startswith(("http://", "https://", "/")) and not u.startswith("//")


_md.validateLink = _allowed_link


def _link_open(self, tokens, idx, options, env):
    tokens[idx].attrSet("rel", "noopener noreferrer")
    if (tokens[idx].attrGet("href") or "").startswith("http"):
        tokens[idx].attrSet("target", "_blank")
    return self.renderToken(tokens, idx, options, env)


_md.add_render_rule("link_open", _link_open)

# A citation as the agent writes it: [2401.10774], [post:3f2a...], [hn:4211], [s2:ab12]
CITE = re.compile(r"\[((?:\d{4}\.\d{4,5}(?:v\d+)?)|(?:(?:post|hn|s2):[A-Za-z0-9]+))\]")


def md(text: str | None, cited: list[str] | None = None) -> Markup:
    """Markdown to safe HTML. Citations become links to the item page; with `cited`, only
    those IDs are linked (an ID the answer step didn't keep stays plain text)."""
    html = _md.render(text or "")
    allowed = set(cited) if cited is not None else None

    def link(m: re.Match) -> str:
        item_id = m.group(1)
        if allowed is not None and item_id not in allowed:
            return m.group(0)
        return f'<a class="cite" href="/items/{quote(item_id, safe=":")}">[{item_id}]</a>'

    return Markup(CITE.sub(link, html))


def item_url(item_id: str) -> str:
    return f"/items/{quote(str(item_id), safe=':')}"


def _parse(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts.replace(" ", "T"))
    except (AttributeError, ValueError):
        return None
    if dt.tzinfo is None:  # SQLite's datetime('now') is UTC without a zone
        dt = dt.replace(tzinfo=UTC)
    return dt


def localtime(ts: str | None, with_date: bool = True) -> str:
    """A stored UTC timestamp in the Mac's local time, e.g. "Nov 21, 9:41 AM"."""
    dt = _parse(ts or "")
    if dt is None:
        return ts or ""
    dt = dt.astimezone()
    fmt = "%b %-d, %-I:%M %p" if with_date else "%-I:%M %p"
    return dt.strftime(fmt)


def duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(int(seconds), 60)
    return f"{m} min {s:02d} s"


def ms(value: float | None) -> str:
    return duration((value or 0) / 1000)


def pct(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.0f}%"


SLOTS = {"main": "Main picks", "early_buzz": "Early buzz", "lab_launch": "Lab launches"}
RATINGS = {"useful": "Useful", "not_relevant": "Not relevant", "skipped": "Skip"}


def install(env) -> None:
    env.filters.update(
        md=md, localtime=localtime, duration=duration, ms=ms, pct=pct, item_url=item_url
    )
    env.globals.update(SLOTS=SLOTS, RATINGS=RATINGS, escape=escape)


# ---- describing agent steps in plain words (the Ask page's progress list) --------------------


def describe_step(node: str, entry: dict | None) -> tuple[str, str]:
    """(headline, detail) for one entry of the Q&A graph's trace, or for a node with none."""
    if entry is None:
        return {
            "prepare": ("Reading the question", ""),
            "rewrite_query": ("Question is self-contained", ""),
            "tools": ("Ran the search", ""),
            "verify": ("Verified the answer", ""),
        }.get(node, (f"Finished {node}", ""))
    step = entry.get("step")
    if step == "rewrite":
        return "Rewrote the follow-up as a standalone question", entry.get("question", "")
    if step == "reason":
        args = entry.get("args") or {}
        what = args.get("query") or args.get("item_id") or ", ".join(map(str, args.values()))
        tool = {
            "store_search": "Searching the local store",
            "search_arxiv": "Searching arXiv",
            "semantic_scholar_lookup": "Looking it up on Semantic Scholar",
            "fetch_full_text": "Fetching a paper's full text",
        }.get(entry.get("tool", ""), f"Calling {entry.get('tool')}")
        return f"{tool}: “{what}”", entry.get("thought", "")
    if step == "observe":
        return f"Got {entry.get('result', '')}", ", ".join(entry.get("ids") or [])
    if step == "evidence":
        verdict = entry.get("verdict")
        if verdict == "sufficient":
            return "Evidence check: enough to answer", ""
        return "Evidence check: not enough yet", entry.get("missing", "")
    if step == "answer":
        return f"Wrote the answer citing {len(entry.get('cited') or [])} source(s)", ""
    if step == "verify":
        bad = entry.get("unsupported") or []
        head = f"Verifier checked {entry.get('claims', 0)} claim(s)"
        return (head + (f", {len(bad)} unsupported" if bad else ", all supported")), "; ".join(
            b["claim"] for b in bad
        )
    if step == "safety":
        return "Safety: a cited source contains instruction-like text", ", ".join(
            entry.get("instruction_like") or []
        )
    if step == "abstain":
        return "Not enough evidence: the agent won't guess", ""
    return step or node, ""
