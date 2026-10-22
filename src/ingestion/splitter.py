"""Section-aware, criterion-level chunking for clinical trial protocols.

Pipeline:
1. Detect section headers (Markdown, ALL-CAPS, dotted-numbered, "Inclusion/Exclusion Criteria").
2. Label each section inclusion / exclusion / other.
3. Inside eligibility sections, emit one chunk per top-level criterion (numbered or bulleted item).
4. Split all other sections with RecursiveCharacterTextSplitter.
5. Prefix every chunk with "[<protocol_id> | <section>]" and attach metadata used for filtering and citations.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.schemas import CriterionType

PROTOCOL_SEPARATORS = ["\n## ", "\n### ", "\nInclusion Criteria:", "\nExclusion Criteria:", "\n\n", "\n", " "]
DEFAULT_SECTION = "General"

_MARKDOWN_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
_ELIGIBILITY_HEADER = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:#{1,6}\s+)?(?P<kind>inclusion|exclusion)\s+criteria\s*(?::\s*(?P<rest>.*)|$)",
    re.IGNORECASE,
)
_NUMBERED_HEADER = re.compile(r"^\s*\d+(?:\.\d+)+\.?\s+(?P<title>[A-Z][^.:;!?]{2,80})$")
_ALL_CAPS_HEADER = re.compile(r"^\s*(?P<title>[A-Z][A-Z0-9 /&()\-]{2,79})\s*:?\s*$")
# Top-level list item: "1." "1)" "-" "*" "•" with at most 2 spaces of indentation.
_TOP_LEVEL_ITEM = re.compile(r"^ {0,2}(?:\d{1,3}[.)]|[-*•●▪])\s+(?P<text>\S.*)$")
_CONTEXT_PREFIX = re.compile(r"^\[NCT\d{8} \| [^\]]*\]\n")


@dataclass
class _Line:
    text: str
    page: int | None


@dataclass
class _Section:
    title: str
    criterion_type: CriterionType
    lines: List[_Line] = field(default_factory=list)
    starts_with_header: bool = False

    @property
    def has_body(self) -> bool:
        body = self.lines[1:] if self.starts_with_header else self.lines
        return any(line.text.strip() for line in body)

    @property
    def first_page(self) -> int | None:
        return next((line.page for line in self.lines if line.page is not None), None)


def build_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=PROTOCOL_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        keep_separator="start",
        strip_whitespace=True,
    )


def strip_context_prefix(page_content: str) -> str:
    """Return chunk text without the "[NCT... | Section]" prefix line."""
    return _CONTEXT_PREFIX.sub("", page_content, count=1)


def make_chunk_id(protocol_id: str, chunk_index: int, text: str) -> str:
    return hashlib.sha256(f"{protocol_id}|{chunk_index}|{text}".encode("utf-8")).hexdigest()[:32]


def _criterion_type_for(title: str) -> CriterionType:
    lowered = title.lower()
    if "exclusion" in lowered:
        return CriterionType.EXCLUSION
    if "inclusion" in lowered:
        return CriterionType.INCLUSION
    return CriterionType.OTHER


def _match_header(line: str) -> tuple[str, str] | None:
    """Return (section_title, trailing_text) if the line is a section header."""
    stripped = line.strip()
    if not stripped:
        return None
    match = _ELIGIBILITY_HEADER.match(stripped)
    if match:
        return f"{match.group('kind').title()} Criteria", (match.group("rest") or "").strip()
    match = _MARKDOWN_HEADER.match(line)
    if match:
        return match.group("title").strip(), ""
    match = _NUMBERED_HEADER.match(stripped)
    if match:
        return match.group("title").strip(), ""
    match = _ALL_CAPS_HEADER.match(stripped)
    if match and any(c.isalpha() for c in match.group("title")) and len(match.group("title").split()) <= 8:
        return match.group("title").strip().title(), ""
    return None


def _to_lines(documents: Sequence[Document]) -> List[_Line]:
    lines: List[_Line] = []
    for doc in documents:
        page = doc.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else None
        text = doc.page_content.replace("\r\n", "\n").replace("\r", "\n")
        lines.extend(_Line(text=raw.rstrip(), page=page_number) for raw in text.split("\n"))
    return lines


def _to_sections(lines: Sequence[_Line]) -> List[_Section]:
    sections: List[_Section] = [_Section(title=DEFAULT_SECTION, criterion_type=CriterionType.OTHER)]
    for line in lines:
        header = _match_header(line.text)
        if header is None:
            sections[-1].lines.append(line)
            continue
        title, rest = header
        section = _Section(
            title=title,
            criterion_type=_criterion_type_for(title),
            lines=[_Line(line.text.strip(), line.page)],
            starts_with_header=True,
        )
        if section.criterion_type is not CriterionType.OTHER:
            # The header line itself is replaced by the context prefix; keep only trailing text.
            section.lines = [_Line(rest, line.page)] if rest else []
            section.starts_with_header = False
        sections.append(section)
    return [s for s in sections if s.has_body]


def _split_criteria(section: _Section) -> tuple[List[tuple[str, int | None]], List[tuple[str, int | None]]]:
    """Split an eligibility section into (preamble_parts, criteria). Each part is (text, page)."""
    preamble: List[_Line] = []
    items: List[List[_Line]] = []
    for line in section.lines:
        if not line.text.strip():
            continue
        match = _TOP_LEVEL_ITEM.match(line.text)
        if match:
            items.append([_Line(match.group("text").strip(), line.page)])
        elif items:
            items[-1].append(_Line(line.text.strip(), line.page))  # wrapped line or sub-item
        else:
            preamble.append(line)

    def join(parts: List[_Line]) -> tuple[str, int | None]:
        return "\n".join(p.text.strip() for p in parts).strip(), parts[0].page

    preamble_parts = [join(preamble)] if preamble else []
    return preamble_parts, [join(item) for item in items]


def split_protocol(
    documents: Sequence[Document],
    protocol_id: str,
    source_filename: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[Document]:
    """Turn loaded protocol pages into enriched, criterion-level chunks."""
    splitter = build_text_splitter(chunk_size, chunk_overlap)
    pieces: List[tuple[str, _Section, int | None, bool]] = []  # (text, section, page, is_criterion)

    for section in _to_sections(_to_lines(documents)):
        if section.criterion_type is CriterionType.OTHER:
            body = "\n".join(line.text for line in section.lines).strip()
            pieces.extend((text, section, section.first_page, False) for text in splitter.split_text(body))
            continue

        preamble, criteria = _split_criteria(section)
        if criteria:
            pieces.extend((text, section, page, False) for text, page in preamble if text)
            pieces.extend((text, section, page, True) for text, page in criteria if text)
        else:
            # Eligibility written as prose: fall back to size-based splitting, still labelled.
            body = "\n".join(line.text for line in section.lines).strip()
            pieces.extend((text, section, section.first_page, True) for text in splitter.split_text(body))

    chunks: List[Document] = []
    for index, (text, section, page, is_criterion) in enumerate(pieces):
        metadata: dict = {
            "protocol_id": protocol_id,
            "source_document": source_filename,
            "section": section.title,
            "criterion_type": section.criterion_type.value,
            "is_criterion": is_criterion,
            "chunk_index": index,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chunk_id": make_chunk_id(protocol_id, index, text),
        }
        if page is not None:
            metadata["page"] = page
        chunks.append(Document(page_content=f"[{protocol_id} | {section.title}]\n{text}", metadata=metadata))
    return chunks
