"""Deterministic intent routing (rules + policy) and approved response wording.

No LLM and no retrieval is involved here. Greetings and small talk get
predefined answers, sensitive / unsupported categories are denied by explicit
policy, and everything else is treated as an airport question. A classifier
never authorizes a sensitive action on its own.

Policy and wording live in JSON (``config/policy.json``,
``config/responses.json``) keyed by language, so adding a language is a config
change, not a code change.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.workflow.contracts import Route

_PUNCT_RE = re.compile(r"[^\w\s'\-]")
_SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lower-case, drop punctuation, collapse whitespace (apostrophes removed)."""
    lowered = text.lower().replace("\u2019", "'").replace("'", "")
    lowered = _PUNCT_RE.sub(" ", lowered).replace("-", " ")
    return _SPACE_RE.sub(" ", lowered).strip()


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    category: Optional[str] = None  # greeting id or denial category id


@dataclass(frozen=True)
class _Rule:
    id: str
    patterns: tuple["re.Pattern[str]", ...]


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class Router:
    """Rule-based router. Build with ``Router.from_file`` or ``Router(policy_dict)``."""

    def __init__(self, policy: dict, default_language: str = "en") -> None:
        self._default = default_language
        self._greetings: dict[str, list[_Rule]] = {}
        self._denied: dict[str, list[_Rule]] = {}
        for lang, block in policy.get("languages", {}).items():
            self._greetings[lang] = [self._compile(r, full=True) for r in block.get("greeting", [])]
            self._denied[lang] = [self._compile(r, full=False) for r in block.get("denied", [])]
        if default_language not in self._greetings:
            raise ValueError(f"policy has no rules for default language '{default_language}'")

    @staticmethod
    def _compile(rule: dict, full: bool) -> _Rule:
        wrap = (lambda p: rf"^(?:{p})$") if full else (lambda p: p)
        return _Rule(rule["id"], tuple(re.compile(wrap(p), re.IGNORECASE) for p in rule["patterns"]))

    @classmethod
    def from_file(cls, path: str | Path, default_language: str = "en") -> "Router":
        return cls(_load_json(path), default_language)

    def route(self, text: str, language: str = "en") -> RouteDecision:
        lang = language if language in self._greetings else self._default
        norm = normalize(text)
        for rule in self._greetings[lang]:
            if any(p.match(norm) for p in rule.patterns):
                return RouteDecision(Route.GREETING, rule.id)
        for rule in self._denied[lang]:
            if any(p.search(norm) for p in rule.patterns):
                return RouteDecision(Route.OUT_OF_SCOPE, rule.id)
        return RouteDecision(Route.AIRPORT_QUESTION)


class ResponseBook:
    """Approved wording per language. Falls back to the default language per key."""

    def __init__(self, responses: dict, default_language: str = "en") -> None:
        self._langs: dict = responses.get("languages", {})
        self._default = default_language
        if default_language not in self._langs:
            raise ValueError(f"responses has no entry for default language '{default_language}'")

    @classmethod
    def from_file(cls, path: str | Path, default_language: str = "en") -> "ResponseBook":
        return cls(_load_json(path), default_language)

    def _block(self, language: str) -> dict:
        return self._langs.get(language) or self._langs[self._default]

    def greeting(self, category: str, language: str = "en") -> str:
        block = self._block(language)
        return block.get(category) or self._langs[self._default][category]

    def refusal(self, category: str, language: str = "en") -> str:
        block = self._block(language).get("refusal", {})
        return block.get(category) or self._langs[self._default]["refusal"][category]

    def fallback(self, language: str = "en") -> str:
        return self._block(language).get("fallback") or self._langs[self._default]["fallback"]

    def error(self, language: str = "en") -> str:
        return self._block(language).get("error") or self._langs[self._default]["error"]

    def all_strings(self) -> list[str]:
        """Every approved string (used by tests that check wording against trap rules)."""
        out: list[str] = []
        for block in self._langs.values():
            for value in block.values():
                if isinstance(value, str):
                    out.append(value)
                elif isinstance(value, dict):
                    out.extend(v for v in value.values() if isinstance(v, str))
        return out
