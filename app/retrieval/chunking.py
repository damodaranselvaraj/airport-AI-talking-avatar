"""Chunking: deterministic chunk IDs and the two-stage (header, then size) chunker."""
from __future__ import annotations

from hashlib import sha256

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.retrieval.parsing import ParsedDocument

_HEADER_LEVELS = (("#", "h1"), ("##", "h2"), ("###", "h3"))
# Paragraph -> newline -> sentence -> word -> character fallback.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def normalize_whitespace(text: str) -> str:
    """Collapse every run of whitespace to a single space and trim the ends."""
    return " ".join(text.split())


def make_chunk_id(source: str, heading_path: str, ordinal: int, text: str) -> str:
    """Stable chunk ID: same source/heading path/ordinal/normalized text -> same ID."""
    normalized = normalize_whitespace(text)
    digest = sha256(f"{source}|{heading_path}|{ordinal}|{normalized}".encode()).hexdigest()
    return f"{source}:{ordinal}:{digest[:16]}"


class Chunk(BaseModel):
    chunk_id: str = Field(..., min_length=1)
    text: str  # headings + parent context + body
    source: str  # citation filename
    metadata: dict[str, str]  # source, h1, h2, h3, order, plus frontmatter


class Chunker:
    """Header-aware split (H1/H2/H3, H2 primary) then size-aware split of oversized sections."""

    def __init__(
        self, chunk_size: int = 300, chunk_overlap: int = 0, primary_header_level: int = 2
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if not 1 <= primary_header_level <= len(_HEADER_LEVELS):
            raise ValueError("primary_header_level must be 1, 2 or 3")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.primary_header_level = primary_header_level
        self._header_splitter = MarkdownHeaderTextSplitter(
            list(_HEADER_LEVELS), strip_headers=False
        )

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        if not doc.body.strip():
            return []

        chunks: list[Chunk] = []
        for section in self._header_splitter.split_text(doc.body):
            headings = [
                (level, section.metadata[key])
                for level, (_, key) in enumerate(_HEADER_LEVELS, start=1)
                if key in section.metadata
            ]
            all_lines = [f"{'#' * level} {name}" for level, name in headings]
            body = self._body_without_headings(section.page_content, set(all_lines))
            own_level = headings[-1][0] if headings else 0
            # Own heading plus enclosing headings up to the primary level.
            context = [
                f"{'#' * level} {name}"
                for level, name in headings
                if level == own_level or level <= self.primary_header_level
            ]
            heading_path = " > ".join(name for _, name in headings)
            for text in self._section_texts(context, body):
                order = len(chunks)
                metadata = dict(doc.frontmatter)
                metadata.update(
                    {
                        "source": doc.source,
                        "h1": section.metadata.get("h1", ""),
                        "h2": section.metadata.get("h2", ""),
                        "h3": section.metadata.get("h3", ""),
                        "order": str(order),
                    }
                )
                chunks.append(
                    Chunk(
                        chunk_id=make_chunk_id(doc.source, heading_path, order, text),
                        text=text,
                        source=doc.source,
                        metadata=metadata,
                    )
                )
        return chunks

    @staticmethod
    def _body_without_headings(content: str, heading_lines: set[str]) -> str:
        """Drop the heading lines the header splitter left at the top of a section."""
        lines = content.splitlines()
        while lines and (not lines[0].strip() or lines[0].strip() in heading_lines):
            lines.pop(0)
        return "\n".join(lines).strip()

    def _section_texts(self, context: list[str], body: str) -> list[str]:
        own = context[-1:]
        if len(self._join(own, body)) <= self.chunk_size:
            # Small section: one chunk, with as much parent context as still fits.
            for start in range(len(context) + 1):
                text = self._join(context[start:], body)
                if len(text) <= self.chunk_size:
                    return [text]
        prefix_lines = self._prefix_lines(context)
        prefix_len = len("\n".join(prefix_lines)) + 1 if prefix_lines else 0
        budget = self.chunk_size - prefix_len
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget,
            chunk_overlap=min(self.chunk_overlap, budget - 1),
            length_function=len,
            separators=_SEPARATORS,
            keep_separator="end",
        )
        return [self._join(prefix_lines, piece) for piece in splitter.split_text(body)]

    def _prefix_lines(self, context: list[str]) -> list[str]:
        """Longest heading context that still leaves at least half the chunk for body text."""
        min_body = max(1, self.chunk_size // 2)
        for start in range(len(context)):
            lines = context[start:]
            if self.chunk_size - (len("\n".join(lines)) + 1) >= min_body:
                return lines
        if not context:
            return []
        # Even the deepest heading is too long: truncate it so the body keeps room.
        return [context[-1][: max(1, self.chunk_size - min_body - 1)]]

    @staticmethod
    def _join(lines: list[str], body: str) -> str:
        return "\n".join([*lines, body] if body else lines)
