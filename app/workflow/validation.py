"""Answer validation: is every claim supported by the cited passages?

``Validator`` is the seam. ``CitationOverlapValidator`` is the deterministic
implementation used now. The README names Jev (TypeSafe System One) as the
primary validator, with this overlap check as the fallback when Jev is
unavailable; Jev is DEFERRED until its package name/version is verified.

To add Jev later: implement ``Validator`` in a ``JevValidator`` class (call Jev
once per turn, and fall back to ``CitationOverlapValidator`` if Jev raises or
times out), then pass it to the workflow constructor. No other change
is required.

Known limit of a lexical check: it cannot detect a claim that reuses words from
the passages but states them wrongly (e.g. a negation flip). It reliably
catches invented facts (gate codes, times, phone numbers, prices) and answers
that drift away from the cited text, which are the failures a kiosk must avoid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.workflow.contracts import RetrievedPassage

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[:.][0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = frozenset(
    """a an the and or but if so of to in on at by for from with about as into than then
    is are was were be been being am do does did done have has had having it its this that
    these those there here you your yours we our i me my he she they them his her their
    can could will would shall should may might must not no yes also just only very more
    most some any all each every other such own same too up down out off over under again
    please thank thanks sorry which what who whom whose when where why how""".split()
)


@dataclass(frozen=True)
class ValidationResult:
    supported: bool
    reason: str = ""


class Validator(Protocol):
    name: str

    def validate(
        self, answer: str, cited: Sequence[RetrievedPassage]
    ) -> ValidationResult: ...


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("\u2013", " ").replace("\u2014", " "))


def _stem(token: str) -> str:
    """Very light plural/verb folding so 'gates' matches 'gate'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content_tokens(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text) if t not in _STOPWORDS]


def _has_digit(token: str) -> bool:
    return any(c.isdigit() for c in token)


class CitationOverlapValidator:
    """Deterministic validator: lexical overlap + exact match of every number/code."""

    name = "citation_overlap"

    def __init__(self, min_overlap: float = 0.6) -> None:
        if not 0.0 < min_overlap <= 1.0:
            raise ValueError("min_overlap must be in (0, 1]")
        self._min_overlap = min_overlap

    def validate(
        self, answer: str, cited: Sequence[RetrievedPassage]
    ) -> ValidationResult:
        if not answer or not answer.strip():
            return ValidationResult(False, "empty answer")
        if not cited:
            return ValidationResult(False, "no cited passages")

        passage_raw = {t for p in cited for t in tokenize(p.text)}
        passage_stems = {_stem(t) for t in passage_raw}

        # 1. Every digit-bearing token (gate codes, times, phone numbers, counts) must
        #    appear verbatim in the cited passages. Invented specifics are the worst failure.
        for token in tokenize(answer):
            if _has_digit(token) and token not in passage_raw:
                return ValidationResult(False, "unsupported specific detail")

        # 2. Each substantive sentence must be mostly made of words from the passages.
        for sentence in filter(None, (s.strip() for s in _SENTENCE_RE.split(answer))):
            words = [w for w in _content_tokens(sentence) if not _has_digit(w)]
            if len(words) < 2:
                continue
            hits = sum(1 for w in words if w in passage_stems)
            if hits / len(words) < self._min_overlap:
                return ValidationResult(False, "sentence not supported by cited passages")
        return ValidationResult(True)
