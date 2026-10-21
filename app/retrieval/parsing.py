"""Markdown parsing: split a document into frontmatter metadata and body."""
from __future__ import annotations

from pathlib import Path

import frontmatter
from pydantic import BaseModel, Field


class MarkdownParseError(ValueError):
    """A document could not be read or its frontmatter is invalid."""


class ParsedDocument(BaseModel):
    source: str = Field(..., min_length=1, description="Filename used for citations")
    frontmatter: dict[str, str]
    body: str


class MarkdownParser:
    def parse(self, path: Path) -> ParsedDocument:
        path = Path(path)
        try:
            post = frontmatter.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # unreadable file or invalid YAML frontmatter
            raise MarkdownParseError(f"Cannot parse {path}: {exc}") from exc
        if not isinstance(post.metadata, dict):
            raise MarkdownParseError(f"Cannot parse {path}: frontmatter is not a mapping")
        return ParsedDocument(
            source=path.name,
            frontmatter={str(k): str(v) for k, v in post.metadata.items()},
            body=post.content,
        )
