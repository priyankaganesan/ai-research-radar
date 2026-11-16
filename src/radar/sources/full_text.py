"""Full text of a paper or lab post, split into sections (M5).

An abstract is ~250 tokens: enough to know *what* a paper does, rarely enough to explain *how*.
When questions keep coming back to one paper, the agent fetches the whole thing.

Where the text comes from, best first:
1. **arXiv HTML** (`arxiv.org/html/<id>`). arXiv renders most papers since late 2023 from their
   LaTeX source with a tool called LaTeXML. The HTML keeps the real structure: every section is a
   `<section>` with an `<h2>`/`<h3>` heading, and every formula carries its LaTeX in `alttext`.
2. **arXiv PDF**, read with PyMuPDF. A PDF is a page of positioned glyphs, not a document tree,
   so headings have to be guessed from lines like "3 Method" or "4.2 Results". This is the
   "plain PyMuPDF" baseline that tools like Docling or Marker improve on.
3. **Lab posts**: the post's own web page, split at its `<h2>`/`<h3>` headings.

Everything here is parsing; `services.full_text` decides when to fetch and stores the result.
"""

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

import httpx

from radar.schema import Item
from radar.sources import http
from radar.tracing import traced

ARXIV_HTML = "https://arxiv.org/html/{id}"
ARXIV_PDF = "https://arxiv.org/pdf/{id}"
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")

# Sections that add tokens but no answers. Matched on the heading without its number.
SKIP_SECTIONS = re.compile(
    r"^(references|bibliography|acknowledge?ments?|abstract|author contributions?|"
    r"impact statement|broader impacts?|ethics statement|reproducibility statement|"
    r"(data|code|data and code) availability|"
    r"limitations and broader impact|checklist|neurips paper checklist)\b",
    re.IGNORECASE,
)
# Sites whose terms forbid feeding their content to a model (see CLAUDE.md).
BLOCKED_HOSTS = ("ieeexplore.ieee.org", "linkedin.com")


class FullTextUnavailable(http.SourceUnavailable):
    """No usable full text for this item (no HTML or PDF, or nothing readable in it)."""


@dataclass
class Section:
    heading: str  # e.g. "3.2 Draft heads"; "" for text before the first heading
    text: str


@dataclass
class FullText:
    item_id: str
    source: str  # "arxiv_html" | "arxiv_pdf" | "html"
    url: str
    sections: list[Section] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(s.text) for s in self.sections)


# ---- HTML -------------------------------------------------------------------------------------


class _SectionParser(HTMLParser):
    """Walks an HTML page and cuts its text into sections at h2/h3 headings.

    html.parser (standard library) calls handle_starttag / handle_data / handle_endtag as it
    reads, like a SAX parser: we keep a little state instead of building a whole DOM tree.
    """

    SKIP = {
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "noscript",
        "svg",
        "button",
        "form",
        "table",
        "figure",
        "aside",
    }
    BLOCK = {
        "p",
        "div",
        "li",
        "section",
        "article",
        "br",
        "h1",
        "h4",
        "h5",
        "h6",
        "dd",
        "dt",
        "blockquote",
        "pre",
        "figcaption",
    }
    HEADINGS = {"h2", "h3"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self.top = ""  # current h2 heading, so h3s can be named "Method > 3.1 Heads"
        self.heading: str | None = None  # text of the heading being read, if inside one
        self.heading_tag = ""
        self.current = Section("", "")
        self.buf: list[str] = []
        self.skip_tag = ""  # the tag whose contents we're skipping (nav, table, ...)
        self.skip_depth = 0  # how deep inside nested copies of that tag we are
        self.in_math = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        cls = a.get("class") or ""
        if self.skip_depth:
            if tag == self.skip_tag:
                self.skip_depth += 1
            return
        if "ltx_equation" in cls or "ltx_eqn" in cls:
            # LaTeXML lays out display equations as a <table>; keep those (their <math> has
            # the LaTeX), while real tables are still skipped.
            self._data("\n")
            return
        if tag in self.SKIP or "ltx_page_header" in cls or "ltx_page_footer" in cls:
            self.skip_tag, self.skip_depth = tag, 1
            return
        if tag == "math":
            # MathML spells formulas out symbol by symbol; the LaTeX in alttext reads better.
            self.in_math += 1
            if self.in_math == 1 and a.get("alttext"):
                self._data(f" ${a['alttext']}$ ")
            return
        if tag in self.HEADINGS:
            self.heading, self.heading_tag = "", tag
            return
        if tag in self.BLOCK:
            self._data("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            if tag == self.skip_tag:
                self.skip_depth -= 1
            return
        if tag == "math":
            self.in_math = max(0, self.in_math - 1)
            return
        if tag == self.heading_tag and self.heading is not None:
            title = " ".join(self.heading.split())
            self.heading, self.heading_tag = None, ""
            self._flush()
            if tag == "h2":
                self.top = title
                self.current = Section(title, "")
            else:
                self.current = Section(f"{self.top} > {title}" if self.top else title, "")
            return
        if tag in self.BLOCK:
            self._data("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth or self.in_math:
            return
        self._data(data)

    def _data(self, text: str) -> None:
        if self.heading is not None:
            self.heading += text
        else:
            self.buf.append(text)

    def _flush(self) -> None:
        self.current.text = clean_text("".join(self.buf))
        if self.current.text:
            self.sections.append(self.current)
        self.buf = []

    def result(self) -> list[Section]:
        self._flush()
        return self.sections


def clean_text(text: str) -> str:
    """Collapse runs of spaces inside lines and blank lines between paragraphs."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    text = re.sub(r"\n{2,}", "\n\n", "\n".join(lines)).strip()
    # A formula ends in "$ "; drop the space before punctuation that follows it ("$f$ .").
    return re.sub(r"\$ ([.,;:)])", r"$\1", text)


def strip_number(heading: str) -> str:
    """'3.2 Draft heads' -> 'Draft heads'; 'Appendix A Proofs' -> 'Proofs'."""
    last = heading.split(" > ")[-1]
    return re.sub(r"^(appendix\s+)?([A-Z]|\d+)(\.\d+)*\.?\s+", "", last, flags=re.I).strip()


def keep_sections(sections: list[Section], untitled: str | None = None) -> list[Section]:
    """Drop references, acknowledgements and similar. Text before the first heading is dropped
    too (on arXiv that's the title, authors and abstract, which the store already has), unless
    `untitled` names it: a blog post often says the most before its first heading."""
    out = []
    for s in sections:
        if not s.heading:
            if untitled and s.text:
                out.append(Section(untitled, s.text))
            continue
        top = s.heading.split(" > ")[0]
        if SKIP_SECTIONS.match(strip_number(top)) or SKIP_SECTIONS.match(strip_number(s.heading)):
            continue
        out.append(s)
    return out


def parse_html(html: str, untitled: str | None = None) -> list[Section]:
    """HTML -> sections. Pure function, so it's easy to test with a saved page."""
    p = _SectionParser()
    p.feed(html)
    p.close()
    return keep_sections(p.result(), untitled)


# ---- PDF --------------------------------------------------------------------------------------

# "3 Method", "3. Method", "4.2 Draft heads", "A Proofs". Headings are short, start with a
# capital and have no sentence punctuation; the length cap keeps most body lines out.
_NUMBERED = re.compile(r"^(\d{1,2})((?:\.\d{1,2}){0,2})\.?\s+([A-Z][^.!?]{1,70})$")
_LETTERED = re.compile(r"^(?:Appendix\s+)?([A-H])((?:\.\d{1,2}){0,2})\.?\s+([A-Z][^.!?]{1,70})$")
_NAMED = re.compile(
    r"^(abstract|introduction|related work|background|conclusions?|references|"
    r"acknowledge?ments?|appendix)$",
    re.IGNORECASE,
)


def pdf_text(data: bytes) -> str:
    """Plain text of every page of a PDF, in reading order as PyMuPDF sees it."""
    import pymupdf  # imported here so the rest of the app doesn't pay for it

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text("text", sort=True) for page in doc)


class _HeadingGuesser:
    """Decides whether a PDF line is a section heading.

    A line like "2 GPUs were used for training" looks like a heading too, so we also require
    the numbers to make sense: section N can only be followed by N or N+1 ("3.2" after "3.1"
    is fine, "7" after "3" is not). After References, only appendix headings count, lettered
    in order (A, B, C...), so bibliography lines don't start new sections.
    """

    def __init__(self) -> None:
        self.top = 0  # last top-level section number
        self.letter = "@"  # last appendix letter ("@" is the character before "A")
        self.after_refs = False

    def __call__(self, line: str) -> bool:
        if _NAMED.match(line):
            if line.lower() == "references":
                self.after_refs = True
            return True
        if not self.after_refs:
            m = _NUMBERED.match(line)
            if m and not line[-1].isdigit():
                n = int(m.group(1))
                if n in (self.top, self.top + 1) and (n == self.top + 1 or m.group(2)):
                    self.top = n
                    return True
            return False
        m = _LETTERED.match(line)
        if m and m.group(1) in (self.letter, chr(ord(self.letter) + 1)):
            if m.group(1) != self.letter or m.group(2):
                self.letter = m.group(1)
                return True
        return False


def split_pdf_text(text: str) -> list[Section]:
    """Guess section boundaries in PDF text from heading-shaped lines."""
    is_heading = _HeadingGuesser()
    sections: list[Section] = []
    heading, buf = "", []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if line and is_heading(line):
            sections.append(Section(heading, clean_text("\n".join(buf))))
            heading, buf = line, []
        else:
            buf.append(line)
    sections.append(Section(heading, clean_text("\n".join(buf))))
    return _join_pdf_paragraphs(keep_sections([s for s in sections if s.text]))


def _join_pdf_paragraphs(sections: list[Section]) -> list[Section]:
    """PDF text breaks every line; join lines into paragraphs and undo end-of-line hyphens."""
    for s in sections:
        s.text = re.sub(r"(\w)-\n(\w)", r"\1\2", s.text)
        s.text = re.sub(r"(?<!\n)\n(?!\n)", " ", s.text)
    return sections


# ---- fetching ---------------------------------------------------------------------------------


def is_arxiv_id(item_id: str) -> bool:
    return bool(ARXIV_ID.match(item_id))


def allowed(url: str) -> bool:
    return not any(h in url for h in BLOCKED_HOSTS)


MIN_SECTIONS = 2  # fewer than this from arXiv HTML means a stub page; try the PDF instead


@traced("full_text.fetch", "tool", kind="source")
def fetch(item: Item, client: httpx.Client | None = None) -> FullText:
    """Fetch and split the full text of a paper (arXiv HTML, else PDF) or a lab post."""
    client = client or http.make_client(timeout=60)
    if is_arxiv_id(item.id):
        url = ARXIV_HTML.format(id=item.id)
        try:
            resp = http.request_with_retry(client, url, name="arXiv HTML")
            sections = parse_html(resp.text)
            if len(sections) >= MIN_SECTIONS:
                return FullText(item.id, "arxiv_html", str(resp.url), sections)
        except httpx.HTTPStatusError:
            pass  # 404: arXiv has no HTML for this paper (older, or the LaTeX didn't convert)
        url = item.pdf_url or ARXIV_PDF.format(id=item.id)
        resp = _get(client, url, "arXiv PDF")
        try:
            sections = split_pdf_text(pdf_text(resp.content))
        except Exception as e:  # a corrupt or encrypted PDF: PyMuPDF raises its own errors
            raise FullTextUnavailable(f"Couldn't read the PDF of {item.id}: {e}") from e
        if not sections:
            raise FullTextUnavailable(f"The PDF of {item.id} had no readable text.")
        return FullText(item.id, "arxiv_pdf", url, sections)

    if item.kind == "post" and item.url and allowed(item.url):
        resp = _get(client, item.url, "the post's page")
        sections = parse_html(resp.text, untitled="Post")
        if not sections:
            raise FullTextUnavailable(f"No readable text found on {item.url}.")
        return FullText(item.id, "html", str(resp.url), sections)
    raise FullTextUnavailable(
        f"No full-text source for {item.id}: only arXiv papers and lab posts are supported."
    )


def _get(client: httpx.Client, url: str, name: str) -> httpx.Response:
    try:
        return http.request_with_retry(client, url, name=name)
    except httpx.HTTPStatusError as e:
        raise FullTextUnavailable(f"{name} answered {e.response.status_code} for {url}.") from e
